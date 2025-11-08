import os
import torch
import argparse
import numpy as np
from ssl_training import SSLLitSwinUNETR
from ssl_data_loading import ssl_data_loader
from utils import visualize_mask_overlay

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')  # Trade off precision for performance

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="brats")
    parser.add_argument("--roi", type=int, nargs=3, default=[128, 128, 64])
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()

    fold = args.fold
    roi = tuple(args.roi)
    patch_size = (2, 2, 2)
    ckpt_path = (r"D:\Master_Thesis\master_thesis\swin_model\checkpoints_ssl\Brats\version_0\best-model-epoch=95"
                 r"-train_loss=0.1600.ckpt")

    if args.data == "brats":
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_brats")
        split_file = os.path.join(data_dir, "data_split.json")
        _, _, test_loader = ssl_data_loader(dataset_type=args.data, batch_size=1, roi=roi, data_dir=data_dir,
                                      split_file=split_file, fold=fold)  # Load the test data (same as validation data)

    elif args.data == "numorph":
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_numorph")
        _, _, test_loader = ssl_data_loader(dataset_type=args.data, batch_size=1, roi=roi, data_dir=data_dir)

    else:
        raise ValueError("Unknown dataset")

    # Load the model checkpoint
    model = SSLLitSwinUNETR.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    # Take just a few random samples
    num_samples_total = len(test_loader.dataset)
    num_samples = min(args.num_samples, num_samples_total)
    np.random.seed(100)  # For reproducibility
    selected_indices = np.random.choice(num_samples_total, size=num_samples, replace=False)
    print(f"Visualizing {num_samples} randomly selected test samples...")

    # Do a forward pass
    with torch.no_grad():
        for idx, sample_idx in enumerate(selected_indices, start=1):
            batch = test_loader.dataset[sample_idx]
            x = torch.as_tensor(batch["image"]).unsqueeze(0).to(device)  # (B, C, D, H, W)
            print(x.shape)
            recon, mask = model(x)
            print(recon.shape)
            print(mask.shape)
            visualize_mask_overlay(x, mask, recon)


