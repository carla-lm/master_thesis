import os
import torch
import argparse
import numpy as np
from ssl_training import SSLLitSwinUNETR
from ssl_data_loading import ssl_data_loader
from utils import visualize_mask_overlay


def pad_to_window(x, window_size=(8, 8, 8)):
    B, C, D, H, W = x.shape
    pad_d = (window_size[0] - D % window_size[0]) % window_size[0]
    pad_h = (window_size[1] - H % window_size[1]) % window_size[1]
    pad_w = (window_size[2] - W % window_size[2]) % window_size[2]
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
    return x, (pad_d, pad_h, pad_w)


def unpad(x, pads):
    pad_d, pad_h, pad_w = pads
    if pad_d: x = x[:, :, :-pad_d, :, :]
    if pad_h: x = x[:, :, :, :-pad_h, :]
    if pad_w: x = x[:, :, :, :, :-pad_w]
    return x


if __name__ == '__main__':
    device = torch.device("cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')  # Trade off precision for performance

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="brats")
    parser.add_argument("--roi", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()

    fold = args.fold
    roi = tuple(args.roi)
    patch_size = (2, 2, 2)
    ckpt_path = (r"D:\Master_Thesis\master_thesis\swin_model\checkpoints_ssl\Brats\L1_SSIM_E1v3_Latest\best-model"
                 r"-epoch=74-val_loss=0.1974.ckpt")

    if args.data == "brats":
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_brats")
        split_file = os.path.join(data_dir, "data_split.json")
        _, test_loader, _, _ = ssl_data_loader(dataset_type=args.data, batch_size=1, roi=roi, data_dir=data_dir,
                                               split_file=split_file,
                                               fold=fold)  # Load the test data (same as validation data)

    elif args.data == "numorph":
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_numorph")
        _, test_loader, _, _ = ssl_data_loader(dataset_type=args.data, batch_size=1, roi=roi, data_dir=data_dir)

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
            x = batch[0]["image"].unsqueeze(0).to(device)  # Add batch dimension (B, C, D, H, W)
            print(x.shape)
            x_pad, pads = pad_to_window(x, window_size=(6, 6, 6))
            recon, mask = model(x_pad)
            recon = unpad(recon, pads)
            mask = unpad(mask, pads)
            visualize_mask_overlay(x, mask, recon)
