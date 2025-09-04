# Import necessary packages
import os
import json
import time
import nibabel as nib
import torch
import argparse
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import monai.transforms as T
from monai.transforms import (AsDiscrete, Activations)
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai import data
from monai.data import decollate_batch
from swinunetr import SwinUNETR3D
from functools import partial

torch.set_float32_matmul_precision("high")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def data_loader(batch_size, data_dir, split_file, fold, roi):
    train_files, val_files = data_split(data_dir, split_file, fold)
    print(f"Found {len(train_files)} training samples and {len(val_files)} validation samples")
    train_trans, val_trans = get_transforms(roi)
    train_dataset = data.Dataset(data=train_files, transform=train_trans)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataset = data.Dataset(data=val_files, transform=val_trans)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, val_loader


def get_transforms(roi):
    train_transforms = T.Compose([  # create a sequential pipeline of preprocessing steps and augmentations
        T.LoadImaged(keys=["image", "label"]),  # Load label and image entries of the sample
        T.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),  # Convert label map to multi-channel format
        T.CropForegroundd(keys=["image", "label"], source_key="image",  # Crop out empty background
                          k_divisible=[roi[0], roi[1], roi[2]], allow_smaller=True),  # Ensure crop is divisible by roi
        # T.SpatialPadd(keys=["image", "label"], spatial_size=roi, mode="constant"),  # If crop is smaller, pad to roi
        T.RandSpatialCropd(keys=["image", "label"],  # Randomly crop a patch of size roi in image and label
                           roi_size=[roi[0], roi[1], roi[2]], random_size=False),  # Always return the given roi size
        T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),  # Flip with 50% chance in 1st spatial dimension
        T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),  # Flip with 50% chance in 2nd spatial dimension
        T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),  # Flip with 50% chance in 3rd spatial dimension
        T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),  # norm_vox = (vox - mean)/std
        T.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),  # Simulate variations in contrast
        T.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),  # Simulate variations in brightness
    ])

    val_transforms = T.Compose([  # Only deterministic preprocessing, no random augmentations
        T.LoadImaged(keys=["image", "label"]),
        T.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    return train_transforms, val_transforms


## WOULD BE NICE TO HAVE A GENERAL FUNCTION TO CONVERT LABEL MAPS TO ONE-HOT MULTICHANNEL MASKS

def data_split(data_dir, split_file, fold):
    with open(split_file) as file:
        data = json.load(file)  # Load json file's content into a Python dictionary
        data = data["training"]  # Extract the entries of "training" in the file

    train = []
    val = []
    for d in data:  # Each data entry has the keys fold, image, label
        for key in d:  # Add full path to the relative paths of the files
            if isinstance(d[key], list):  # If the key contains a list (image in different modalities)
                d[key] = [os.path.join(data_dir, image) for image in d[key]]  # Add full path to each list element
            elif isinstance(d[key], str):  # If the key contains a string (image label)
                d[key] = os.path.join(data_dir, d[key])  # Add full path to the label file

        if d["fold"] == fold:  # Add the files of the specified fold to the validation split for cross-validation
            val.append(d)
        else:
            train.append(d)

    return train, val


def visualize_data(data_dir):
    matplotlib.use("TkAgg")  # To see figure live
    # Load a random image and label to visualize
    img_path = os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_flair.nii.gz")
    label_path = os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_seg.nii.gz")
    img = nib.load(img_path).get_fdata()
    label = nib.load(label_path).get_fdata()
    print(f"Image shape: {img.shape}, label shape: {label.shape}")
    # Load whole sample dictionary (need a modality dimension for transform pipeline to work)
    image_dict = {
        "image": [
            os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_t1.nii.gz"),
            os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_t1ce.nii.gz"),
            os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_t2.nii.gz"),
            os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_flair.nii.gz")
        ],
        "label": os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_seg.nii.gz")
    }
    # Get and apply the training transformations
    preview_trans = get_transforms((128, 128, 128))
    transformed = preview_trans[0](image_dict)  # 0 for all transforms, 1 for deterministic transforms
    img_t = transformed["image"]
    label_t = transformed["label"]
    print(f"Transformed image shape: {img_t.shape}, transformed label shape: {label_t.shape}")

    # Plot raw image and label
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img[:, :, 78], cmap="gray")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.title("Original Label")
    plt.imshow(label[:, :, 78])
    plt.axis("off")
    # Plot transformed image with overlaid label
    plt.subplot(1, 3, 3)
    plt.title("Transformed Image")
    plt.imshow(img_t[3, :, :, 78], cmap="gray")
    plt.imshow(label_t.sum(axis=0)[:, :, 78], alpha=0.3)
    plt.axis("off")
    plt.show()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def save_checkpoint(model, epoch, dir_add, val_acc=0, checkpoint_type="best"):
    os.makedirs(os.path.dirname(dir_add), exist_ok=True)
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "val_acc": val_acc, "state_dict": state_dict}
    if checkpoint_type == "best":
        path = os.path.join(dir_add, "best_model.pt")
        torch.save(save_dict, path)

    elif checkpoint_type == "periodic":
        path = os.path.join(dir_add, f"model_epoch_{epoch}.pt")
        torch.save(save_dict, path)

    print(f"Saving checkpoint at {dir_add}")


def save_training_history(history, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(history, f, indent=4)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    # Use amp if a CUDA device is available
    use_amp = (device.type == "cuda")
    for idx, batch_data in enumerate(loader):
        # Move data to the appropriate device
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        if use_amp:
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(data)
                loss = loss_func(logits, target)

            # Scale the loss and perform backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training step without AMP
            logits = model(data)
            loss = loss_func(logits, target)
            loss.backward()
            optimizer.step()

        run_loss.update(loss.item(), n=batch_size)
        print(
            "Epoch {}/{} {}/{}".format(epoch+1, epochs, idx+1, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()

    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, model_inferer=None, post_sigmoid=None, post_pred=None):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()
    use_amp = (device.type == "cuda")
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            if use_amp:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model_inferer(data)
            else:
                logits = model_inferer(data)

            # Post-processing steps
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]

            # Calculate metric
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()

            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            dice_tc = run_acc.avg[0]
            dice_wt = run_acc.avg[1]
            dice_et = run_acc.avg[2]

            print(
                "Val {}/{} {}/{}".format(epoch+1, np.floor(epochs/val_every)+1, idx+1, len(loader)),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()

    return run_acc.avg


def trainer(model, train_loader, val_loader, optimizer, loss_func, acc_func, scheduler, check_dir, model_inferer=None,
            start_epoch=0, post_sigmoid=None, post_pred=None, checkpoint=True):
    val_acc_max = 0.0
    dices_tc, dices_wt, dices_et, dices_avg = [], [], [], []
    loss_epochs, trains_epoch = [], []
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    for epoch in range(start_epoch, epochs):  # Training Loop
        print(time.ctime(), "Epoch:", epoch+1)
        epoch_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, epoch=epoch, loss_func=loss_func)
        print(
            "Final training  {}/{}".format(epoch+1, epochs),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )

        if (epoch + 1) % val_every == 0 or epoch == 0:  # Validation Loop
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            val_acc = val_epoch(model, val_loader, epoch=epoch, acc_func=acc_func, model_inferer=model_inferer,
                                post_sigmoid=post_sigmoid, post_pred=post_pred)
            dice_tc, dice_wt, dice_et = val_acc
            val_avg_acc = np.mean(val_acc)
            print(
                "Final validation stats {}/{}".format(epoch+1, epochs),
                ", dice_tc:", dice_tc, ", dice_wt:", dice_wt, ", dice_et:", dice_et, ", Dice_Avg:", val_avg_acc,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            dices_tc.append(dice_tc)
            dices_wt.append(dice_wt)
            dices_et.append(dice_et)
            dices_avg.append(val_avg_acc)
            if checkpoint and val_avg_acc > val_acc_max:  # Save best model
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                save_checkpoint(model, epoch, dir_add=check_dir, val_acc=val_acc_max, checkpoint_type="best")

            if checkpoint:  # Save current state and metrics every validation loop
                save_checkpoint(model, epoch, dir_add=check_dir, val_acc=val_avg_acc, checkpoint_type="periodic")
                history_to_save = {
                    "val_acc_max": float(val_acc_max),
                    "dices_tc": [float(x) for x in dices_tc],
                    "dices_wt": [float(x) for x in dices_wt],
                    "dices_et": [float(x) for x in dices_et],
                    "dices_avg": [float(x) for x in dices_avg],
                    "loss_epochs": [float(x) for x in loss_epochs],
                    "trains_epoch": trains_epoch,
                }
                save_training_history(history_to_save, os.path.join(check_dir, "training_history.json"))

            scheduler.step()

    print("Training Finished. Best Accuracy: ", val_acc_max)
    return val_acc_max, dices_tc, dices_wt, dices_et, dices_avg, loss_epochs, trains_epoch


if __name__ == '__main__':
    # Set device to be used (gpu is faster)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    # Set parameters that can be passed by the user
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=48)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--roi', type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument('--val_every', type=int, default=10)
    parser.add_argument('--data', type=str, default='brats')
    parser.add_argument("--checkpoint", dest="checkpoint", action="store_true")
    parser.add_argument("--no-checkpoint", dest="checkpoint", action="store_false")
    parser.set_defaults(checkpoint=True)
    args = parser.parse_args()

    # Define hyperparameters
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch
    fold = args.fold
    roi = tuple(args.roi)
    feature_size = args.embed_dim
    val_every = args.val_every
    num_heads = [3, 6, 12, 24]
    window_size = (7, 7, 7)
    patch_size = (2, 2, 2)
    embed_dims = [feature_size, feature_size * 2, feature_size * 4, feature_size * 8, feature_size * 16]
    sw_batch_size = 4
    infer_overlap = 0.5

    # Set relevant paths
    if args.data == 'brats':
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_brats")  # Path is current directory + data_brats
        split_file = os.path.join(data_dir, "data_split.json")  # Json path is data directory + json filename

    if args.data == 'selma':
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_selma")  # Path is current directory + data_selma

    # Load data
    train_loader, val_loader = data_loader(batch_size, data_dir, split_file, fold, roi)

    # Visualize data
    # visualize_data(data_dir)

    # Create checkpoint folder unique for each run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    check_dir = os.path.join(os.getcwd(), "checkpoints", timestamp)
    os.makedirs(check_dir, exist_ok=True)

    # Initialize model
    model = SwinUNETR3D(
        in_channels=4,
        patch_size=patch_size,
        embed_dims=embed_dims,
        num_heads=num_heads,
        window_size=window_size,
        num_classes=3
    ).to(device)

    # Monai's model to see if training pipeline works
    # model = SwinUNETR(
    #     in_channels=4,
    #     out_channels=3,
    #     feature_size=48,
    #     drop_rate=0.0,
    #     attn_drop_rate=0.0,
    #     dropout_path_rate=0.0,
    #     use_checkpoint=True,
    # ).to(device)

    # Optimizer and loss function
    dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)  # Adam with weight decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Decrease following cosine curve
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
    )
    start_epoch = 0
    (val_acc_max, dices_tc, dices_wt, dices_et, dices_avg, loss_epochs, trains_epoch) = trainer(
        model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
        loss_func=dice_loss, acc_func=dice_acc, scheduler=scheduler, model_inferer=model_inferer,
        start_epoch=start_epoch, post_sigmoid=post_sigmoid, post_pred=post_pred,
        checkpoint=args.checkpoint, check_dir=check_dir)

    print(f"Training completed, best average dice: {val_acc_max:.4f} ")

    # Save recorded evaluation metrics at the end of the training
    history_to_save = {
        "val_acc_max": float(val_acc_max),
        "dices_tc": [float(x) for x in dices_tc],
        "dices_wt": [float(x) for x in dices_wt],
        "dices_et": [float(x) for x in dices_et],
        "dices_avg": [float(x) for x in dices_avg],
        "loss_epochs": [float(x) for x in loss_epochs],
        "trains_epoch": trains_epoch
    }
    save_training_history(history_to_save, os.path.join(check_dir, "training_history.json"))
