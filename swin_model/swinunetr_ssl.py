import torch
import torch.nn as nn
import numpy as np
from swinunetr import SwinUNETREncoder3D


def mask_3d(img, window_size, input_size, mask_ratio):
    B, C, D, H, W = img.shape
    wh, ww, wd = window_size
    in_d, in_h, in_w = input_size
    if any(dim % w != 0 for dim, w in zip((in_d, in_h, in_w), (wd, wh, ww))):
        raise ValueError(f"Input {input_size} not divisible by window {window_size}")

    # Define the mask grid shape (number of windows along each axis)
    mask_shape = [in_d // wd, in_h // wh, in_w // ww]
    num_windows = np.prod(mask_shape).item()

    # Create a random boolean mask grid
    mask_flat = np.zeros(num_windows, dtype=bool)
    masked_indices = np.random.choice(num_windows,
                                      round(num_windows * mask_ratio),
                                      replace=False)  # Select windows to mask
    mask_flat[masked_indices] = True  # False = visible, True = masked
    mask_grid = mask_flat.reshape(mask_shape)  # Convert grid from 1D to 3D

    # Propagate the window mask to all voxels in that window
    mask = np.logical_or(
        mask_grid[:, None, :, None, :, None],
        np.zeros([1, wd, 1, wh, 1, ww], dtype=bool)
    ).reshape(in_d, in_h, in_w)  # (num_winD, num_winH, num_winW) --> (D, H, W)

    mask = torch.from_numpy(mask).to(img.device)  # Convert to tensor and move to the same device as input

    # Apply the mask to the input
    x_masked = img.detach().clone()  # Copy the input
    x_masked[:, :, mask] = -1  # Mask voxels where mask = 1. I put -1, so it does not get confused with the background 0
    mask = mask.unsqueeze(0).unsqueeze(1).repeat(B, 1, 1, 1, 1)  # (B, 1, D, H, W), all samples in B have the same mask

    return x_masked, mask


class SSLDecoder3D(nn.Module):  # Decoder to reconstruct masked voxels from the encoded visible tokens
    def __init__(self, decoder_dim, out_channels):
        super().__init__()
        self.decoding_blocks = nn.Sequential(
            nn.Conv3d(decoder_dim, decoder_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(decoder_dim // 2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(decoder_dim // 2, decoder_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(decoder_dim // 4),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(decoder_dim // 4, decoder_dim // 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(decoder_dim // 8),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(decoder_dim // 8, decoder_dim // 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(decoder_dim // 16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(decoder_dim // 16, decoder_dim // 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(decoder_dim // 16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(decoder_dim // 16, out_channels, kernel_size=1, stride=1),
        )

    def forward(self, x):
        reconstruction = self.decoding_blocks(x)
        reconstruction = torch.sigmoid(reconstruction)  # Normalize to same scale as input in [0, 1]
        return reconstruction


class SSLSwinUNETR3D(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dims, num_heads, window_size, mask_ratio):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.mask_ratio = mask_ratio  # Ratio of image to be kept: higher = more visible patches
        self.encoder = SwinUNETREncoder3D(in_channels=self.in_channels, patch_size=self.patch_size,
                                          embed_dims=self.embed_dims, num_heads=self.num_heads,
                                          window_size=self.window_size)
        self.decoder = SSLDecoder3D(decoder_dim=embed_dims[-1], out_channels=in_channels)

    def forward(self, x):
        # print("Raw Input Shape:", x.shape)
        # Mask the input at voxel-level
        x_masked, mask = mask_3d(img=x, window_size=self.window_size, input_size=x.shape[2:],
                                 mask_ratio=self.mask_ratio)
        # Masked input has shape (B, C, D, H, W), make it (B, H, W, D, C)
        x_masked = x_masked.permute(0, 3, 4, 2, 1).contiguous()
        # print("Encoder-ready Input Shape:", x_masked.shape)
        latent, _ = self.encoder(x_masked)  # (B, H_lat, W_lat, D_lat, C)
        # latent = self.encoder_monai(x_masked)[4]
        # print("Encoder Output Shape:", latent.shape)
        latent = latent.permute(0, 4, 3, 1, 2).contiguous()  # (B, C, D_lat, H_lat, W_lat)
        # print("Decoder Input Shape:", latent.shape)
        recon = self.decoder(latent)
        recon = recon[:, :, :x.shape[2], :x.shape[3], :x.shape[4]]  # Trim padding to match original
        # print("Decoder Output Shape:", recon.shape)
        return recon, mask


# TESTING SECTION
if __name__ == "__main__":
    import os
    import nibabel as nib
    from utils import visualize_mask_overlay, compute_intensity_range

    # Load an example image
    data_dir = os.path.join(os.getcwd(), "TrainingData", "data_brats")  # Path is current directory + data_brats
    img_path = os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_flair.nii.gz")
    img = nib.load(img_path).get_fdata()  # (H, W, D)
    img = torch.tensor(img, dtype=torch.float32)
    img = img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions (1, 1, H, W, D)
    # Define variables
    patch_size = (2, 2, 2)
    embed_dims = [48, 96, 192, 384, 768]
    num_heads = [3, 6, 12, 24]
    window_size = (5, 5, 5)
    num_classes = 3

    # Instantiate and run model
    model = SSLSwinUNETR3D(in_channels=img.shape[1],
                           patch_size=patch_size,
                           embed_dims=embed_dims,
                           num_heads=num_heads,
                           window_size=window_size,
                           mask_ratio=0.75)
    recon, mask = model(img)
    # Normalize image for display
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
    visualize_mask_overlay(img, mask, recon)
