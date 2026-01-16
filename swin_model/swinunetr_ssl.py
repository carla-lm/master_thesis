import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from swinunetr import SwinUNETREncoder3D


def voxel_level_mask(img, window_size, input_size, mask_ratio):
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


def patch_level_mask(B, H, W, D, mask_ratio, device):
    num_patches = H * W * D  # H, W, D here is pH, pW, pD
    num_masked = int(num_patches * mask_ratio)
    noise = torch.rand(B, num_patches, device=device)  # Random noise matrix per sample in batch
    ids_shuffle = torch.argsort(noise, dim=1)  # Sort indices according to the random matrix, shuffling them randomly
    mask = torch.zeros(B, num_patches, device=device, dtype=torch.bool)
    mask.scatter_(1, ids_shuffle[:, :num_masked], True)  # Mask the defined amount of patches
    mask = mask.view(B, H, W, D)  # Reshape flat mask to 3D mask
    return mask  # True = masked, False = visible


def upsample_mask_to_voxels(patch_mask, original_shape):
    D, H, W = original_shape[2], original_shape[3], original_shape[4]
    mask = patch_mask.float().unsqueeze(1)  # (B, H', W', D') -> (B, 1, H', W', D')
    mask = mask.permute(0, 1, 4, 2, 3)  # (B, 1, D', H', W')
    mask = F.interpolate(mask, size=(D, H, W), mode='nearest')
    return mask.bool()  # (B, 1, D, H, W)


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


class MAESwinUNETR3D(nn.Module):
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
        # Learnable mask token in embedding space (more expressive than voxel-level masking)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1, embed_dims[0]))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x):
        original_shape = x.shape  # (B, C, D, H, W)
        x = x.permute(0, 3, 4, 2, 1).contiguous()  # (B, C, D, H, W) -> (B, H, W, D, C)
        skip_embed, x = self.encoder.patch_embed(x)  # x: (B, H', W', D', embed_dim)
        B, pH, pW, pD, C = x.shape
        patch_mask = patch_level_mask(B, pH, pW, pD, self.mask_ratio, x.device)  # Create mask at patch level
        mask_tokens = self.mask_token.expand(B, pH, pW, pD, -1)  # Expand mask token matrix
        x = torch.where(patch_mask.unsqueeze(-1), mask_tokens, x)  # Replace masked patches with learnable mask token
        for i in range(self.encoder.num_stages):
            x = self.encoder.trans_blocks[i](x)
            x = self.encoder.merge_layers[i](x)

        latent = x.permute(0, 4, 3, 1, 2).contiguous()  # (B, C, D_lat, H_lat, W_lat)
        recon = self.decoder(latent)
        recon = recon[:, :, :original_shape[2], :original_shape[3], :original_shape[4]]
        voxel_mask = upsample_mask_to_voxels(patch_mask, original_shape)  # Upsample mask to voxel level for loss
        return recon, voxel_mask


class BYOLOnlineProjectionMLP(nn.Module):  # Project encoder output to lower-dimensional space
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.proj_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # LayerNorm instead of BatchNorm for small batch sizes
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.proj_layer(x)


class BYOLOnlinePredictorMLP(nn.Module):  # Predict target projection from online projection
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.pred_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # LayerNorm instead of BatchNorm for small batch sizes
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.pred_layer(x)


class BYOLExponentialMovingAvg:
    def __init__(self, beta):
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class BYOLSwinUNETR3D(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dims, num_heads, window_size,
                 proj_hidden=4096, proj_out=256, pred_hidden=4096, ma_decay=0.99):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.ma_decay = ma_decay
        # Online network
        self.encoder = SwinUNETREncoder3D(in_channels=in_channels, patch_size=patch_size,
                                          embed_dims=embed_dims, num_heads=num_heads,
                                          window_size=window_size)
        self.predictor = BYOLOnlinePredictorMLP(proj_out, pred_hidden, proj_out)
        self.projector = BYOLOnlineProjectionMLP(embed_dims[-1], proj_hidden, proj_out)
        # Target network (momentum updated, no gradients)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.projector)
        self.target_ema_updater = BYOLExponentialMovingAvg(ma_decay)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_moving_average(self):
        assert self.target_encoder is not None, 'Target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.encoder)
        update_moving_average(self.target_ema_updater, self.target_projector, self.projector)

    def _encode(self, x, encoder):
        x = x.permute(0, 3, 4, 2, 1).contiguous()  # (B, C, D, H, W) -> (B, H, W, D, C)
        _, x = encoder.patch_embed(x)
        for i in range(encoder.num_stages):
            x = encoder.trans_blocks[i](x)
            x = encoder.merge_layers[i](x)

        B, H, W, D, C = x.shape
        # Global average pool: flatten the 3D feature map into a single 1D vector per image
        x = x.view(B, H * W * D, C)  # keep tokens
        x = F.layer_norm(x, (C,))  # normalize tokens
        x = x.mean(dim=1)  # pool after normalization
        return x  # (B, C)

    def forward(self, view1, view2):
        # Online
        z1 = self._encode(view1, self.encoder)
        z2 = self._encode(view2, self.encoder)
        pred1 = self.predictor(self.projector(z1))
        pred2 = self.predictor(self.projector(z2))

        # Target
        with torch.no_grad():
            target1 = self.target_projector(self._encode(view1, self.target_encoder))
            target2 = self.target_projector(self._encode(view2, self.target_encoder))

        return pred1, pred2, target1.detach(), target2.detach()


# TESTING SECTION
if __name__ == "__main__":
    import os
    import nibabel as nib
    from utils import visualize_mask_overlay, visualize_byol_augmentations

    # Load an example image
    data_dir = os.path.join(os.getcwd(), "TrainingData", "data_numorph")  # Path is current directory + data_brats
    # img_path = os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_flair.nii.gz")
    img_path = os.path.join(data_dir, "c075_images_final_224_64/c0202_Training-Top3-[00x02].nii")
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
    model = MAESwinUNETR3D(in_channels=img.shape[1],
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
    visualize_byol_augmentations(dataset_type="numorph", data_dir="TrainingData/data_numorph", roi=(64, 64, 64))
