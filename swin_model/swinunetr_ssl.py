import torch
import torch.nn as nn
import torch.nn.functional as F
from swinunetr import PatchEmbedding3D, SwinTransformerBlock3D, PatchMerging3D


class SSLMaskingLayer3D(nn.Module):  # Apply a window-based mask to the input
    def __init__(self, mask_ratio, window_size):
        super().__init__()
        self.mask_ratio = mask_ratio  # Fraction of the windows to mask
        self.window_size = window_size

    def forward(self, x):
        B, H, W, D, C = x.shape  # (B, H', W', D', C)
        wh, ww, wd = self.window_size
        # Pad the input if needed to match window size
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww
        pad_d = (wd - D % wd) % wd
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))
            H, W, D = x.shape[1:4]

        N = H * W * D  # Number of tokens in the 3D volume

        # Calculate the number of non-overlapping windows in each dimension
        assert H % wh == 0 and W % ww == 0 and D % wd == 0, \
            "Input dimensions not divisible by window size"  # Make sure the windows fit the image
        num_win_h = H // wh
        num_win_w = W // ww
        num_win_d = D // wd
        num_windows = num_win_h * num_win_w * num_win_d  # Total number of windows

        # Randomly select which windows to keep
        len_keep = int(num_windows * (1 - self.mask_ratio))  # Number of windows to keep visible
        noise = torch.rand(B, num_windows, device=x.device)  # Get random noise values for each window
        ids_shuffle = torch.argsort(noise, dim=1)  # Sort window indices by their noise value (shuffle them)
        ids_keep = ids_shuffle[:, :len_keep]  # Indices of windows that are kept visible

        # Save the 3D coordinates of each window in a grid of shape (num_windows, 3)
        win_coords = torch.stack(torch.meshgrid(
            torch.arange(num_win_h, device=x.device),
            torch.arange(num_win_w, device=x.device),
            torch.arange(num_win_d, device=x.device),
            indexing='ij'
        ), dim=-1).reshape(-1, 3)  # Each row of the grid represents the ijk coordinates of a window in the grid

        # Map window coordinates into window indices
        patch_indices = []
        for b in range(B):  # Go through the batches
            keep_coords = win_coords[ids_keep[b]]  # Get the grid rows with the coords of the windows to keep
            patch_ids = []
            for (i, j, k) in keep_coords:  # Get the indices of the patches inside the kept windows using their coords
                h_idx = torch.arange(i * wh, (i + 1) * wh, device=x.device)
                w_idx = torch.arange(j * ww, (j + 1) * ww, device=x.device)
                d_idx = torch.arange(k * wd, (k + 1) * wd, device=x.device)
                # Save the 3D coordinates of each token in the window in a grid
                grid = torch.stack(torch.meshgrid(h_idx, w_idx, d_idx, indexing='ij'), dim=-1)
                patch_ids.append(grid.reshape(-1, 3))  # (wh, ww, wd, 3) --> (wh*ww*wd, 3)
            patch_ids = torch.cat(patch_ids, dim=0)  # Concatenate token coordinates of all visible windows
            # Flatten (h,w,d) token indices into a 1D index array
            flat_ids = patch_ids[:, 0] * (W * D) + patch_ids[:, 1] * D + patch_ids[:, 2]
            patch_indices.append(flat_ids)
        patch_indices = torch.stack(patch_indices, dim=0)  # Array of token indices to keep per sample: (B, N_visible)

        # Create the binary mask
        mask = torch.ones([B, N], device=x.device)
        mask.scatter_(1, patch_indices, 0)  # Fill mask of 1s (hidden) with 0s (visible) at the given indices
        mask = mask.view(B, H, W, D)  # (B, N) --> (B, H', W', D')
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, 1, C)  # Add channel dim --> (B, H', W', D', C)
        # Apply the mask to the input to zero out the masked tokens
        x_masked = x * (1.0 - mask)
        return x_masked, mask


class SSLEncoder3D(nn.Module):
    def __init__(self, in_channels, embed_dims, num_heads, patch_size, window_size, mask_ratio=0.75):
        super().__init__()
        # Build the encoder pipeline with the SwinUNETR classes and the new masking layer
        self.window_size = window_size
        self.embed_dims = embed_dims
        self.num_stages = len(num_heads)
        self.patch_embed = PatchEmbedding3D(in_channels=in_channels, patch_size=patch_size, embed_dim=embed_dims[0])
        self.trans_blocks = nn.ModuleList()
        self.merge_layers = nn.ModuleList()
        shift = tuple(w // 2 for w in window_size)

        for i in range(self.num_stages):
            stage_blocks = nn.Sequential(
                SwinTransformerBlock3D(embed_dim=embed_dims[i], num_heads=num_heads[i],
                                       window_size=window_size, shift_size=(0, 0, 0)),
                SwinTransformerBlock3D(embed_dim=embed_dims[i], num_heads=num_heads[i],
                                       window_size=window_size, shift_size=shift)
            )
            self.trans_blocks.append(stage_blocks)
            self.merge_layers.append(PatchMerging3D(embed_dims[i]))

        self.mask_layer = SSLMaskingLayer3D(mask_ratio, window_size)

    def forward(self, x):
        # Embed the input
        # print("Encoder-ready Input Shape:", x.shape)
        _, patch_embed = self.patch_embed(x)  # (B, H', W', D', C)
        # print("Embedded Input Shape:", patch_embed.shape)
        # Apply the masking
        x_masked, mask = self.mask_layer(patch_embed)
        # print("Masked Embedded Input Shape:", x_masked.shape)
        # Encode the masked input
        latent = x_masked
        for i in range(self.num_stages):
            latent = self.trans_blocks[i](latent)
            latent = self.merge_layers[i](latent)

        return latent, mask


class SSLDecoder3D(nn.Module):  # Decoder to reconstruct masked voxels from the encoded visible tokens
    def __init__(self, embed_dim, decoder_dim, out_channels):
        super().__init__()
        self.proj = nn.Linear(embed_dim, decoder_dim)  # Project the embedded tokens to the decoder layer dimension
        self.decoding_blocks = nn.Sequential(  # Upsampling stages --> each conv doubles the spatial resolution
            nn.ConvTranspose3d(decoder_dim, decoder_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm3d(decoder_dim // 2),
            nn.GELU(),

            nn.ConvTranspose3d(decoder_dim // 2, decoder_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm3d(decoder_dim // 4),
            nn.GELU(),

            nn.ConvTranspose3d(decoder_dim // 4, decoder_dim // 8, kernel_size=2, stride=2),
            nn.BatchNorm3d(decoder_dim // 8),
            nn.GELU(),

            nn.ConvTranspose3d(decoder_dim // 8, decoder_dim // 16, kernel_size=2, stride=2),
            nn.BatchNorm3d(decoder_dim // 16),
            nn.GELU(),

            nn.ConvTranspose3d(decoder_dim // 16, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x, H, W, D):
        B, N, C = x.shape
        x = self.proj(x)
        x = x.view(B, H, W, D, -1).permute(0, 4, 3, 1, 2).contiguous()  # (B, N, C) --> (B, C, D_lat, H_lat, W_lat)
        reconstruction = self.decoding_blocks(x)
        return reconstruction


class SSLSwinUNETR3D(nn.Module):  # Full SSL Swin architecture
    def __init__(self, in_channels, out_channels, embed_dims, num_heads, patch_size, window_size, mask_ratio=0.75):
        super().__init__()
        # Build the SSL SwinUNETR pipeline
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.window_size = window_size
        self.mask_ratio = mask_ratio
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.encoder = SSLEncoder3D(in_channels=self.in_channels, embed_dims=self.embed_dims, num_heads=self.num_heads,
                                    patch_size=self.patch_size, window_size=self.window_size,
                                    mask_ratio=self.mask_ratio)
        self.decoder = SSLDecoder3D(embed_dim=embed_dims[-1], decoder_dim=embed_dims[-1], out_channels=out_channels)

    def forward(self, x):
        # Prepare and embed the input
        # print("Raw Input Shape:", x.shape)
        x = x.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, D, C)
        # Encode the masked input
        latent, mask = self.encoder(x)
        # print("Encoder Output Shape:", latent.shape)
        # Decode the latent representation to reconstruct the image
        latent_shape = latent.shape[1:4]  # (H_latent, W_latent, D_latent)
        latent_flat = latent.view(latent.shape[0], -1, self.embed_dims[-1])  # (B, N, C)
        reconstruction = self.decoder(latent_flat, *latent_shape)
        reconstruction = reconstruction[:, :, :x.shape[3], :x.shape[1], :x.shape[2]]  # Trim padding to match original
        # print("Decoder Output Shape:", reconstruction.shape)

        return reconstruction, mask


# TESTING SECTION
if __name__ == "__main__":
    import os
    import nibabel as nib
    from utils import visualize_mask_overlay

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
    window_size = (6, 6, 6)
    num_classes = 3

    # Instantiate and run model
    model = SSLSwinUNETR3D(in_channels=img.shape[1],
                           patch_size=patch_size,
                           embed_dims=embed_dims,
                           num_heads=num_heads,
                           window_size=window_size,
                           out_channels=num_classes,
                           )
    recon, mask = model(img)
    visualize_mask_overlay(img, mask, recon, patch_size=patch_size)
