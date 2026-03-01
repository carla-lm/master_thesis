import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from swinunetr import SwinUNETREncoder3D, SwinUNETRDecoder3D, EncoderSkipsProcessor
from transforms import get_ssl_transforms, get_byol_transforms


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
    D, H, W = original_shape[4], original_shape[2], original_shape[3]
    mask = patch_mask.float().unsqueeze(1)  # (B, H', W', D') -> (B, 1, H', W', D')
    mask = mask.permute(0, 1, 4, 2, 3)  # (B, 1, D', H', W')
    mask = F.interpolate(mask, size=(D, H, W), mode='nearest')
    mask = mask.permute(0, 1, 3, 4, 2)  # (B, 1, H, W, D)
    return mask.bool()


class LightDecoder3D(nn.Module):  # Light SimMIM-like decoder
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


class MIMSwinUNETR3D(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dims, num_heads, window_size, mask_ratio):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.mask_ratio = mask_ratio  # Ratio of patches to mask: higher = fewer visible patches
        self.encoder = SwinUNETREncoder3D(in_channels=self.in_channels, patch_size=self.patch_size,
                                          embed_dims=self.embed_dims, num_heads=self.num_heads,
                                          window_size=self.window_size)
        self.processor = EncoderSkipsProcessor(embed_dims)
        self.decoder = SwinUNETRDecoder3D(embed_dims)
        self.recon_head = nn.Sequential(
            nn.Conv3d(embed_dims[0], in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # Learnable mask token in embedding space (more expressive than voxel-level masking)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1, embed_dims[0]))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x):
        original_shape = x.shape  # (B, C, H, W, D)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, C, H, W, D) -> (B, H, W, D, C)
        skip_embed, x = self.encoder.patch_embed(x)
        B, pH, pW, pD, C = x.shape
        patch_mask = patch_level_mask(B, pH, pW, pD, self.mask_ratio, x.device)
        mask_tokens = self.mask_token.expand(B, pH, pW, pD, -1)
        x = torch.where(patch_mask.unsqueeze(-1), mask_tokens, x)

        skips = [skip_embed, x]
        for i in range(self.encoder.num_stages):
            x = self.encoder.trans_blocks[i](x)
            x = self.encoder.merge_layers[i](x)
            x = self.encoder.stage_norms[i](x)
            if i < self.encoder.num_stages - 1:
                skips.append(x)

        # Permute encoder output and all skips to (B, C, D, H, W) for the decoder
        encoder_output = x.permute(0, 4, 3, 1, 2).contiguous()
        skips = [s.permute(0, 4, 3, 1, 2).contiguous() for s in skips]
        # print("Encoder output shape", encoder_output.shape)
        encoder_output, skips = self.processor(encoder_output, skips)
        recon = self.recon_head(self.decoder(encoder_output, skips))
        # print("Decoder output shape", recon.shape)
        recon = recon[:, :, :original_shape[4], :original_shape[2], :original_shape[3]]
        recon = recon.permute(0, 1, 3, 4, 2).contiguous()  # (B, C, H, W, D)
        # print("Reconstruction shape", recon.shape)
        voxel_mask = upsample_mask_to_voxels(patch_mask, original_shape)
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
                 proj_hidden=1024, proj_out=256, pred_hidden=1024, ma_decay=0.9995):
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
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, C, H, W, D) -> (B, H, W, D, C)
        _, x = encoder.patch_embed(x)
        for i in range(encoder.num_stages):
            x = encoder.trans_blocks[i](x)
            x = encoder.merge_layers[i](x)
            x = encoder.stage_norms[i](x)

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
    data_dir = os.path.join(os.getcwd(), "TrainingData", "data_brats")  # Path is current directory + data_brats
    img_path = os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_flair.nii.gz")
    img = nib.load(img_path).get_fdata()  # (H, W, D)
    img = torch.tensor(img, dtype=torch.float32)
    img = img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions (1, 1, H, W, D)

    # Define variables
    patch_size = (2, 2, 2)
    embed_dims = [48, 96, 192, 384, 768]
    num_heads = [3, 6, 12, 24]
    window_size = (8, 8, 8)
    num_classes = 3
    roi = (120, 120, 96)

    img = img[:, :, :roi[0], :roi[1], :roi[2]]  # Crop to ROI size to avoid OOM
    print("Image shape: ", img.shape)

    # Get BYOL views
    base_transform = get_ssl_transforms("brats", roi, ssl_mode="byol")
    sample_dict = {"image": [img_path]}
    base_result = base_transform(copy.deepcopy(sample_dict))
    base_img = base_result["image"]
    view1 = base_result["view1"].unsqueeze(0)
    view2 = base_result["view2"].unsqueeze(0)

    # Instantiate and run models
    model = MIMSwinUNETR3D(in_channels=img.shape[1],
                           patch_size=patch_size,
                           embed_dims=embed_dims,
                           num_heads=num_heads,
                           window_size=window_size,
                           mask_ratio=0.75)

    model2 = BYOLSwinUNETR3D(in_channels=img.shape[1],
                             patch_size=patch_size,
                             embed_dims=embed_dims,
                             num_heads=num_heads,
                             window_size=window_size)
    recon, mask = model(img)
    print("Recon shape: ", recon.shape)
    print("Mask shape: ", mask.shape)
    pred1, pred2, target1, target2 = model2(view1, view2)
    print("Pred1 and Pred2 shape: ", pred1.shape, pred2.shape)
    print("Target1 and Target2 shape: ", target1.shape, target2.shape)

    # Normalize image for display
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
    visualize_mask_overlay(img, mask, recon)
    visualize_byol_augmentations(dataset_type="selma", data_dir="TrainingData/data_selma", roi=(120, 120, 96))
