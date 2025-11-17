import os
import argparse
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from training import LitSwinUNETR
from data_loading import eval_data_loader
from monai.inferers import sliding_window_inference
from functools import partial

matplotlib.use("TkAgg")  # To see figure live


def visualize_prediction(dataset, image, ground_truth, prediction, idx):
    # Make sure everything is CPU numpy arrays
    image = image.detach().cpu().numpy()
    ground_truth = ground_truth.detach().cpu().numpy()
    prediction = prediction.detach().cpu().numpy()

    mid_slice = image.shape[-1] // 2
    if dataset == "brats":
        img_slice = image[3, :, :, mid_slice]  # Use Flair modality as original image
        lbl_slice = ground_truth[:, :, mid_slice]
        pred_slice = prediction[:, :, mid_slice]
    elif dataset == "numorph":
        img_slice = image[:, :, mid_slice]
        lbl_slice = ground_truth[:, :, mid_slice]
        pred_slice = prediction[:, :, mid_slice]

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title(f"Input (sample {idx})")
    axes[1].imshow(lbl_slice, cmap="viridis")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_slice, cmap="viridis")
    axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')  # Trade off precision for performance

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="brats")
    parser.add_argument("--roi", type=int, nargs=3, default=[128, 128, 64])
    parser.add_argument('--fold', type=int, default=1)
    # parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--monai", dest="monai", action="store_true")
    parser.set_defaults(monai=False)
    args = parser.parse_args()

    fold = args.fold
    roi = tuple(args.roi)

    # Load the trained model from checkpoint
    # model = LitSwinUNETR.load_from_checkpoint(args.ckpt_path)
    model = LitSwinUNETR.load_from_checkpoint(r"D:\Master_Thesis\master_thesis\swin_model\checkpoints\Finetune_Brats"
                                              r"\version_0\best-model-epoch=89-val_dice_avg=0.8481.ckpt")
    model.eval()
    model.to(device)
    model_inferer = partial(sliding_window_inference, roi_size=roi, sw_batch_size=1,
                            predictor=model, overlap=0.5)  # Set the model inferer

    if args.data == "brats":
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_brats")
        split_file = os.path.join(data_dir, "data_split.json")
        test_loader = eval_data_loader(dataset_type=args.data, roi=roi, data_dir=data_dir, split_file=split_file,
                                       fold=fold)  # Load the test data (same as validation data)

    elif args.data == "numorph":
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_numorph")
        test_loader = eval_data_loader(dataset_type=args.data, roi=roi, data_dir=data_dir)

    else:
        raise ValueError("Unknown dataset")

    # Take just a few random samples
    num_samples_total = len(test_loader.dataset)
    num_samples = min(args.num_samples, num_samples_total)
    np.random.seed(100)  # For reproducibility
    selected_indices = np.random.choice(num_samples_total, size=num_samples, replace=False)
    print(f"Visualizing {num_samples} randomly selected test samples...")

    # Run testing
    with torch.no_grad():
        for idx, sample_idx in enumerate(selected_indices, start=1):
            batch = test_loader.dataset[sample_idx]
            img = torch.as_tensor(batch["image"]).to(device)
            label = torch.as_tensor(batch["label"]).to(device)
            prediction = model_inferer(img.unsqueeze(0))
            if args.data == "brats":
                prediction = torch.sigmoid(prediction)  # convert logits to [0, 1]
                prediction = (prediction > 0.5).float()  # threshold to binary mask
                prediction = prediction.squeeze(0)  # remove batch dimension from inference
                label = label.float()  # convert bool -> float
                # Convert multichannel label and prediction to a single channel multilabel mask
                pred_mask = torch.zeros_like(prediction[0])
                pred_mask[prediction[1] == 1] = 2
                pred_mask[prediction[0] == 1] = 1
                pred_mask[prediction[2] == 1] = 4
                label_mask = torch.zeros_like(label[0])
                label_mask[label[1] == 1] = 2
                label_mask[label[0] == 1] = 1
                label_mask[label[2] == 1] = 4

            elif args.data == "numorph":
                prediction = torch.sigmoid(prediction)  # convert logits to [0, 1]
                prediction = (prediction > 0.5).float()  # threshold to binary mask
                prediction = prediction.squeeze(0).squeeze(0)  # remove channel and batch dim
                label = label.squeeze(0)  # remove channel dim
                img = img.squeeze(0)  # remove channel dim

            # Visualize each sample in batch
            if args.data == "brats":
                visualize_prediction(args.data, img, label_mask, pred_mask, idx)
            else:
                visualize_prediction(args.data, img, label, prediction, idx)
