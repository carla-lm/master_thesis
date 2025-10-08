import os
import argparse
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from training import LitSwinUNETR
from data_loading import eval_data_loader
from swinunetr import SwinUNETR3D

matplotlib.use("TkAgg")  # To see figure live


def visualize_prediction(dataset, image, ground_truth, prediction, idx):
    # Convert to CPU numpy arrays
    image = image.detach().cpu().numpy().squeeze()  # -> (H, W, D)
    ground_truth = ground_truth.detach().cpu().numpy().squeeze()  # -> (H, W, D)
    prediction = prediction.detach().cpu().numpy().squeeze()  # -> (H, W, D)
    mid_slice = image.shape[-1] // 2

    if dataset == "brats":
        img_slice = image[3, :, :, mid_slice]  # Use Flair modality as original image
        lbl_slice = ground_truth[:, :, mid_slice]
        pred_slice = prediction[:, :, mid_slice]
    elif dataset == "numorph":
        img_slice = image[:, :, mid_slice]
        lbl_slice = ground_truth[:, :, mid_slice]
        pred_slice = prediction[:, :, mid_slice]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
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
    parser.add_argument("--data", type=str, default="numorph")
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
    model = LitSwinUNETR.load_from_checkpoint(
        r'D:\Master_Thesis\master_thesis\swin_model\checkpoints\Custom_Swin_Lit_Numorph\version_3\best-model-epoch'
        r'=169-val_dice_avg=0.7541.ckpt')
    model.eval()
    model.to(device)

    # Load the test data (same as validation data)
    if args.data == "brats":
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_brats")
        split_file = os.path.join(data_dir, "data_split.json")
        test_loader = eval_data_loader(dataset_type=args.data, roi=roi, data_dir=data_dir, split_file=split_file,
                                       fold=fold)

    elif args.data == "numorph":
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_numorph")
        test_loader = eval_data_loader(dataset_type=args.data, roi=roi, data_dir=data_dir)

    # Convert DataLoader to list so that we can take just a few random samples
    test_data = list(test_loader)
    num_samples = min(args.num_samples, len(test_data))
    np.random.seed(42)  # For reproducibility
    selected_indices = np.random.choice(len(test_data), size=num_samples, replace=False)
    print(f"\nVisualizing {num_samples} randomly selected test samples...\n")

    # Run testing
    with torch.no_grad():
        for idx, i in enumerate(selected_indices, start=1):
            batch = test_data[i]
            imgs = torch.as_tensor(batch["image"]).to(device)
            labels = torch.as_tensor(batch["label"]).to(device)
            predictions = model(imgs)
            if args.data == "brats":
                predictions = torch.sigmoid(predictions)  # convert logits to [0, 1]
                predictions = (predictions > 0.5).float()  # threshold to binary mask
                predictions = torch.argmax(predictions, dim=1, keepdim=False)  # single-channel mask
            else:
                predictions = torch.sigmoid(predictions)  # convert logits to [0, 1]
                predictions = (predictions > 0.5).float()  # threshold to binary mask
                predictions = predictions.squeeze(1)  # remove channel dim

            # Visualize each sample in batch
            for j in range(imgs.shape[0]):
                visualize_prediction(args.data, imgs[j], labels[j], predictions[j], idx)
