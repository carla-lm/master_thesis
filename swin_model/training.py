# Import necessary packages
import os
import json
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from functools import partial
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib

import monai.transforms as T
from monai.transforms import (AsDiscrete, Activations)
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai import data
from swinunetr import SwinUNETR3D


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


def data_loader(batch_size, data_dir, split_file, fold, roi):
    train_files, val_files = data_split(data_dir, split_file, fold)
    print(f"Found {len(train_files)} training samples and {len(val_files)} validation samples")
    train_trans, val_trans = get_transforms(roi)
    train_dataset = data.Dataset(data=train_files, transform=train_trans)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_dataset = data.Dataset(data=val_files, transform=val_trans)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
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


# WOULD BE NICE TO HAVE A GENERAL FUNCTION TO CONVERT LABEL MAPS TO ONE-HOT MULTICHANNEL MASKS

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


class LitSwinUNETR(pl.LightningModule):
    def __init__(self, model, lr, epochs, roi, sw_batch_size, infer_overlap):
        super().__init__()
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.loss_func = DiceLoss(to_onehot_y=False, sigmoid=True)
        self.post_sigmoid = Activations(sigmoid=True)
        self.post_pred = AsDiscrete(argmax=False, threshold=0.5)
        self.dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
        self.model_inferer = partial(sliding_window_inference, roi_size=roi, sw_batch_size=sw_batch_size,
                                     predictor=self.model, overlap=infer_overlap)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch["image"], batch["label"]
        logits = self(data)
        loss = self.loss_func(logits, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.dice_metric.reset()

    def validation_step(self, batch, batch_idx):
        data, target = batch["image"], batch["label"]
        logits = self.model_inferer(data)
        val_labels_list = decollate_batch(target)
        val_outputs_list = decollate_batch(logits)
        val_output_convert = [self.post_pred(self.post_sigmoid(val_tensor)) for val_tensor in val_outputs_list]
        self.dice_metric(y_pred=val_output_convert, y=val_labels_list)
        return logits

    def on_validation_epoch_end(self):
        acc, _ = self.dice_metric.aggregate()
        dice_tc, dice_wt, dice_et = acc.cpu().numpy()
        mean_dice = np.mean(acc.cpu().numpy())
        self.log("val_dice_tc", dice_tc)
        self.log("val_dice_wt", dice_wt)
        self.log("val_dice_et", dice_et)
        self.log("val_dice_avg", mean_dice)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')  # Trade off precision for performance

    # Set parameters that can be passed by the user
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=48)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--roi', type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument('--val_every', type=int, default=10)
    parser.add_argument('--experiment', type=int, default=1)
    parser.add_argument('--data', type=str, default='brats')
    parser.add_argument("--load_checkpoint", dest="load_checkpoint", action="store_true")
    parser.add_argument("--no-load_checkpoint", dest="load_checkpoint", action="store_false")
    parser.add_argument("--monai", dest="monai", action="store_true")
    parser.set_defaults(monai=False)
    parser.set_defaults(load_checkpoint=False)
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
    embed_dims = [args.embed_dim * (2 ** i) for i in range(5)]
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

    # Initialize model
    if args.monai:
        print("Using MONAI's SwinUNETR model")
        model = SwinUNETR(
            in_channels=4,
            out_channels=3,
            feature_size=feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
        )
    else:
        print("Using custom SwinUNETR model")
        model = SwinUNETR3D(
            in_channels=4,
            patch_size=patch_size,
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            num_classes=3
        )

    # Set up lightning modules
    lit_model = LitSwinUNETR(model, lr, epochs, roi, sw_batch_size, infer_overlap)

    # Create checkpoint folder
    experiment = args.experiment
    run_name = f"Experiment_{experiment}"
    logger = CSVLogger(save_dir="checkpoints", name=run_name)
    check_dir = logger.log_dir
    print("Saving at {}".format(check_dir))
    os.makedirs(check_dir, exist_ok=True)

    # Save best model
    best_check = pl.callbacks.ModelCheckpoint(
        dirpath=check_dir,
        filename="best-model-{epoch:02d}-{val_dice_avg:.4f}",
        monitor="val_dice_avg",   # Metric used to decide best model
        save_top_k=1,   # Keep only best checkpoint
        mode="max"
    )

    # Save a periodic checkpoint every 5 epochs
    last_check = pl.callbacks.ModelCheckpoint(
        dirpath=check_dir,
        filename="last",
        save_top_k=1,  # Overwrite previous checkpoint
        every_n_epochs=5  # Save model every 5 epochs
    )

    # If the load_checkpoint flag is activated, resume training from last.ckpt
    resume_ckpt = None
    if args.load_checkpoint:
        last_ckpt = os.path.join(check_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            print(f"Resuming training from {last_ckpt}")
            resume_ckpt = last_ckpt
        else:
            print("No last.ckpt found, starting from scratch")
    else:
        print("Starting training from scratch")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[best_check, last_check],
        logger=logger,
        check_val_every_n_epoch=val_every,
        precision="16-mixed",  # Automatic mixed precision for faster training
    )

    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=resume_ckpt)
