import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import torch.nn.functional as F


class PatchEmbedding3D(nn.Module):
    # Patch size and embedding dimension (defined as C in the paper) are modifiable hyperparameters
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.skip_embed = nn.Sequential(nn.Conv3d(in_channels, embed_dim, kernel_size=3, padding=1),
                                        nn.InstanceNorm3d(embed_dim), nn.GELU())
        # self.skip_embed = ResBlock(in_channels, embed_dim)  # So it is consistent with other skips, and more robust
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        ph, pw, pd = self.patch_size
        B, H, W, D, S = x.shape
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        pad_d = (pd - D % pd) % pd
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))
            # print("Embedding Padded Shape:", x.shape)

        x = x.permute(0, 4, 3, 1, 2).contiguous()  # (B, S, D, H, W) as required by Conv3D
        x1 = self.skip_embed(x)  # (B, C, D, H, W) For the first skip connection (no downsampling)
        x2 = self.patch_embed(x)  # (B, C, D', H', W') For the second skip connection (downsampling)
        x1 = x1.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, D, C)
        x2 = x2.permute(0, 3, 4, 2, 1).contiguous()  # (B, H', W', D', C)
        return x1, x2


def window_partition(x, window_size):
    # x: (B, H', W', D', C) (we will not include the ' in the code for simplicity)
    B, H, W, D, C = x.shape
    wh, ww, wd = window_size
    assert H % wh == 0 and W % ww == 0 and D % wd == 0, "Input dimensions not divisible by window size"
    # Divide the patch embedding output into window-sized non-overlapping blocks
    x = x.view(B, H // wh, wh, W // ww, ww, D // wd, wd, C)  # (B, nH, wh, nW, ww, nD, wd, C)
    # Rearrange the axes and make the tensor contiguous in memory to avoid errors
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()  # (B, nH, nW, nD, wh, ww, wd, C)
    # Merge the window grid and flatten the 3D windows
    windows = x.view(-1, wh * ww * wd, C)  # (B * nH * nW * nD, wh * ww * wd, C)
    return windows


def window_reverse(windows, window_size, original_shape):
    wh, ww, wd = window_size
    B, H, W, D, C = original_shape  # (B, H', W', D', C)
    nH, nW, nD = H // wh, W // ww, D // wd

    # Reshape the windows back into the grid layout
    x = windows.view(B, nH, nW, nD, wh, ww, wd, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()  # (B, H', W', D', C)
    x = x.view(B, H, W, D, -1)
    return x


class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_rate=0.0):
        super().__init__()
        self.lin_layer1 = nn.Linear(embed_dim, hidden_dim)
        self.act_layer = nn.GELU()
        self.lin_layer2 = nn.Linear(hidden_dim, embed_dim)
        self.drop_layer = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.drop_layer(self.lin_layer2(self.act_layer(self.lin_layer1(x))))


class WindowAttention3D(nn.Module):
    def __init__(self, embed_dim, window_size, num_heads, attn_drop, proj_drop):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_scale = self.head_dim ** -0.5  # To improve training stability
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        H, W, D = self.window_size
        # Create a bias table for each possible relative position in a 3D window
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * H - 1) * (2 * W - 1) * (2 * D - 1), num_heads))
        # Create a 3D grid of coordinates in a window
        coords_h = torch.arange(H)
        coords_w = torch.arange(W)
        coords_d = torch.arange(D)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, coords_d, indexing="ij"))  # (3, H, W, D)
        coords_flatten = coords.flatten(1)  # (3, H*W*D) Each column is a (H,W,D) coordinate of a token in the window
        # Compute the pairwise relative positions between every token pair in the window
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (3, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 3)
        # Shift the relative coords to be non-negative, for indexing
        relative_coords[:, :, 0] += H - 1
        relative_coords[:, :, 1] += W - 1
        relative_coords[:, :, 2] += D - 1
        # Convert the 3D relative position (x, y, z) into a flat 1D index to look up in the bias table
        relative_coords[:, :, 0] *= (2 * W - 1) * (2 * D - 1)
        relative_coords[:, :, 1] *= (2 * D - 1)
        relative_position_index = relative_coords.sum(-1)  # (N, N)

        # Register the index as a buffer (non-trainable but saved with the model)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=0.02)  # Initialize bias weights

    def forward(self, x, mask):
        B_, N, C = x.shape  # B_ = number of windows * batch, N = tokens per window, C = embed_dim
        qkv = self.qkv(x)  # (B_, N, 3×C) --> calculate concatenated QKV for each token
        # Split QKV into Q,K,V, then split each of those in head_dim chunks across heads
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # Each has dimension (B_, num_heads, N, head_dim)
        # Each head calculates its own scaled attention scores between all N tokens of each window
        attn = (q @ k.transpose(-2, -1)) * self.attn_scale  # (B_, num_heads, N_queries, N_keys)
        # Add a learnt relative position bias per head
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1)  # (num_heads, N, N)
        attn = attn + bias.unsqueeze(0)

        # Apply attention mask if provided (for SW-MSA)
        if mask is not None:
            mask = mask.to(attn.device)
            num_windows = mask.shape[0]
            # From (B_, num_heads, N, N) to (B, num_windows, num_heads, N, N) -> separate attn by batch and window
            attn = attn.view(B_ // num_windows, num_windows, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)  # Apply mask to attn scores
            attn = attn.view(-1, self.num_heads, N, N)  # Return to (B_, num_heads, N, N)

        # Softmax is applied over the keys dimension to get a probability distribution of the keys for each query
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        # Calculate the weighted attention, concatenate the heads' outputs along the embedding dimension
        attn = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        attn = self.out_proj(attn)  # Fuse the info from all heads
        attn = self.proj_drop(attn)
        return attn


def compute_attention_mask(input_shape, window_size, shift_size):
    H, W, D = input_shape
    mask = torch.zeros((1, H, W, D, 1))
    id = 0  # Unique number for each mask region
    # Divide input into blocks of three types: before window region, partially overlapping region, shifted window region
    for h in (slice(0, -window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)):
        for w in (slice(0, -window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)):
            for d in (slice(0, -window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)):
                mask[:, h, w, d, :] = id  # Each region gets a unique label
                id += 1

    mask_windows = window_partition(mask, window_size)  # (num_windows, window_size^3, 1)
    mask_windows = mask_windows.squeeze(-1)  # Remove last dimension
    # Calculate if tokens belong to the same region (difference = 0) or not
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (num_windows, N, N)
    # If difference = 0, attention is allowed (no added value). If not, it is masked (-100 -> softmax will make it 0)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask  # One attention mask per window (num_windows, N, N)


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, shift_size, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(embed_dim)  # LayerNorm to stabilize training and improve convergence
        self.attn = WindowAttention3D(embed_dim, window_size, num_heads, attn_drop=0.0, proj_drop=0.0)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio))  # Paper design choice

    def forward(self, x):
        orig_shape = x.shape
        wh, ww, wd = self.window_size
        residual = x  # The input will be used later for a residual connection
        x = self.norm1(x)  # Apply 1st LN
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   dims=(1, 2, 3))
        else:
            shifted_x = x

        # Compute padding needed for window partitioning
        pad_h = (wh - orig_shape[1] % wh) % wh
        pad_w = (ww - orig_shape[2] % ww) % ww
        pad_d = (wd - orig_shape[3] % wd) % wd
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            shifted_x = F.pad(shifted_x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))
            # print(f"Padded for window partition: {shifted_x.shape}")

        padded_shape = shifted_x.shape
        x_windows = window_partition(shifted_x, self.window_size)  # (num_windows * B, window_size^3, C)

        # Compute attention mask only for shifted blocks
        mask = None
        if any(s > 0 for s in self.shift_size):
            mask = compute_attention_mask(padded_shape[1:4], self.window_size, self.shift_size)

        attn_windows = self.attn(x_windows, mask)
        shifted_x = window_reverse(attn_windows, self.window_size, padded_shape)  # (B, H, W, D, C)
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                           dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_h > 0 or pad_w > 0 or pad_d > 0:  # Remove padding
            x = x[:, :orig_shape[1], :orig_shape[2], :orig_shape[3], :]
            # print(f"Pad Removed: {x.shape}")

        x = residual + x  # Apply first residual connection to the attention output
        x = x + self.mlp(self.norm2(x))  # Apply 2nd LN, MLP and second residual connection
        return x


class PatchMerging3D(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Merging area of 2x2x2 affects 8 voxels that are concatenated along C (8C), we project to 2C dimension
        self.reduction = nn.Linear(8 * input_dim, 2 * input_dim, bias=False)  # Paper design choice
        self.norm = nn.LayerNorm(8 * input_dim)  # Paper design choice

    def forward(self, x):
        B, H, W, D, C = x.shape
        pad_h = H % 2
        pad_w = W % 2
        pad_d = D % 2
        if pad_h or pad_w or pad_d:
            x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))
            # print("Merging Padded Shape:", x.shape)

        # Split the input into 2x2x2 cubes, each xi has shape (B, H/2, W/2, D/2, C)
        # and contains all voxels of a relative cube position
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 0::2, 0::2, 1::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 1::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 1::2, 1::2, 0::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)  # (B, H/2, W/2, D/2, 8*C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, D/2, 2*C)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Padding = 1 so that output size = input size (needed to later fuse with skip connections)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)  # Better for CNNs than LayerNorm
        self.norm2 = nn.InstanceNorm3d(out_channels)  # Better for CNNs than LayerNorm
        self.act = nn.GELU()
        # Adjust residual connection if channels differ
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=1),
                                            nn.InstanceNorm3d(out_channels))

    def forward(self, x):
        residual = x
        out = self.norm2(self.conv2(self.act(self.norm1(self.conv1(x)))))
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.act(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # In MONAI, upsampling is done with a transposed convolutional layer
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ResBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)  # Upsample output from deeper layer
        # print("Upsampled X Shape:", x.shape)
        if skip is not None:
            # Crop x or skip so their dimensions match
            def crop_to_match(x1, x2):
                # x and skip have shape [B, C, D, H, W]
                d_diff = x1.size(2) - x2.size(2)
                h_diff = x1.size(3) - x2.size(3)
                w_diff = x1.size(4) - x2.size(4)
                return x1[:, :,
                       d_diff // 2: d_diff // 2 + x2.size(2),
                       h_diff // 2: h_diff // 2 + x2.size(3),
                       w_diff // 2: w_diff // 2 + x2.size(4)]

            if x.size(2) != skip.size(2) or x.size(3) != skip.size(3) or x.size(4) != skip.size(4):
                if x.numel() > skip.numel():
                    x = crop_to_match(x, skip)
                else:
                    skip = crop_to_match(skip, x)

            # print("Cropped X Shape:", x.shape)
            x = torch.cat([x, skip], dim=1)  # Concatenate upsampled tensor with skip connection
            # print("Concat Shape:", x.shape)
        x = self.conv(x)  # Apply the two convolution layers with normalization and activation
        return x


class SwinUNETREncoder3D(nn.Module):  # Whole encoding pipeline
    def __init__(self, in_channels, patch_size, embed_dims, num_heads, window_size):
        super().__init__()
        self.window_size = window_size
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

    def forward(self, x):
        skips = []
        skip_embed, x = self.patch_embed(x)
        skips.extend([skip_embed, x])
        print("Patch Embedding Output Shape:", x.shape)
        for i in range(self.num_stages):
            x = self.trans_blocks[i](x)
            # print(f"Stage {i + 1} Output Shape (before merging):", x.shape)
            x = self.merge_layers[i](x)
            # print(f"Stage {i + 1} Output Shape (after merging):", x.shape)
            if i < self.num_stages - 1:
                skips.append(x)  # Save the transformer output after stages 1-3 for the skip connection

        return x, skips


class EncoderSkipsProcessor(nn.Module):
    def __init__(self, embed_dims):
        super().__init__()
        # Only process skips[1:], skips[0] already has ResBlock applied during patch embedding
        self.output_block = ResBlock(embed_dims[-1], embed_dims[-1])
        self.skip_blocks = nn.ModuleList([ResBlock(embed_dims[i], embed_dims[i]) for i in range(len(embed_dims) - 1)])

    def forward(self, encoder_output, skips):
        processed_output = self.output_block(encoder_output)
        processed_skips = [skips[0]]
        for block, skip in zip(self.skip_blocks, skips[1:]):
            processed_skips.append(block(skip))
        return processed_output, processed_skips


class SwinUNETRDecoder3D(nn.Module):
    def __init__(self, embed_dims):
        super().__init__()
        self.decoder_stages = nn.ModuleList()
        for i in range(len(embed_dims)):
            self.decoder_stages.append(
                DecoderBlock(
                    in_channels=embed_dims[-(i + 1)],
                    skip_channels=embed_dims[-(i + 2)] if i < len(embed_dims) - 1 else embed_dims[0],
                    out_channels=embed_dims[-(i + 2)] if i < len(embed_dims) - 1 else embed_dims[0],
                )
            )

    def forward(self, x, skips):
        for i, stage in enumerate(self.decoder_stages):
            skip = skips[-(i + 1)]
            # print(f"Decoder Level {len(self.decoder_stages) - i}, X and Skip Shape:", x.shape, skip.shape)
            x = stage(x, skip)
        return x


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.head = nn.Conv3d(in_channels, num_classes, kernel_size=1)
        # For tasks where each voxel can only belong to one class you don't use sigmoid, you output raw logits
        # self.act = nn.Sigmoid()  # Output in [0,1] range --> only if you have one class or voxel can be multiclass

    def forward(self, x):
        return self.head(x)  # (B, num_classes, D, H, W)


class SwinUNETR3D(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dims, num_heads, window_size, num_classes):
        super().__init__()
        self.encoder = SwinUNETREncoder3D(in_channels=in_channels, patch_size=patch_size, embed_dims=embed_dims,
                                          num_heads=num_heads, window_size=window_size)
        self.processor = EncoderSkipsProcessor(embed_dims)
        self.decoder = SwinUNETRDecoder3D(embed_dims)
        self.seg_head = SegmentationHead(in_channels=embed_dims[0], num_classes=num_classes)

    def forward(self, x):
        print("Raw Input Shape:", x.shape)
        x = x.permute(0, 3, 4, 2, 1).contiguous()  # Input has shape (B, C, D, H, W), make it (B, H, W, D, C)
        print("Encoder-ready Input Shape:", x.shape)
        # Encoding Stage (Linear Embedding + Swin Transformers + Merging Layers)
        encoder_output, skips = self.encoder(x)
        print("Encoder Output Shape:", encoder_output.shape)
        # for i, skip in enumerate(skips):
        # print(f"Skip {i + 1} shape: {skip.shape}")

        # Make Skips and Output Shape (B, C, D, H, W) for Decoder
        encoder_output = encoder_output.permute(0, 4, 3, 1, 2).contiguous()  # (B, C, D, H, W)
        skips = [s.permute(0, 4, 3, 1, 2).contiguous() for s in skips]  # (B, C, D, H, W)
        encoder_output, skips = self.processor(encoder_output, skips)
        # print("Processed Encoder Output Shape:", encoder_output.shape)
        # for i, skip in enumerate(skips):
        # print(f"Processed Skip {i + 1} shape: {skip.shape}")

        # Decoding Stage (Upsampling Blocks + ResCNN Blocks)
        decoder_output = self.decoder(encoder_output, skips)
        # print("Decoder Output Shape:", decoder_output.shape)

        # Obtain Final Segmented Output
        final_output = self.seg_head(decoder_output)
        print("Final Output Shape:", final_output.shape)

        return final_output


# TESTING SECTION
if __name__ == "__main__":
    import os
    import nibabel as nib

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
    window_size = (7, 7, 7)
    num_classes = 3

    # Instantiate and run model
    model = SwinUNETR3D(
        in_channels=img.shape[1],
        patch_size=patch_size,
        embed_dims=embed_dims,
        num_heads=num_heads,
        window_size=window_size,
        num_classes=num_classes
    )
    output = model(img)
