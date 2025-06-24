import torch
import torch.nn.functional as F
import torch.nn as nn


# Notation: input is 3D image with shape (H, W, D, S) (height, width, depth, channels), there will be
# batches so the input has shape (B, H, W, D, S). For convenience, we use (B, S, D, H, W) in the decoder stage
# as that is the format that convolution functions require.
# The Swin UNETR creates non-overlapping patches of the input data and uses an embedding layer + window partition
# to create windows with a desired size for computing the self-attention.
# The encoded feature representations in the Swin transformer are fed to a CNN-decoder
# via skip connection at multiple resolutions. Final segmentation output consists of as many
# output channels as element types we want to segment.


class PatchEmbedding3D(nn.Module):
    # Patch size and embedding dimension (defined as C in the paper) are modifiable hyperparameters
    def __init__(self, in_channels, patch_size=(4, 4, 4), embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.skip_embed = nn.Sequential(nn.Conv3d(in_channels, embed_dim, kernel_size=3, padding=1),
                                        nn.InstanceNorm3d(embed_dim), nn.ReLU(inplace=True))
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        ph, pw, pd = self.patch_size
        B, H, W, D, S = x.shape
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        pad_d = (pd - D % pd) % pd
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))
            print("Embedding Padded Shape:", x.shape)

        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, S, H, W, D) as required by Conv3D
        x1 = self.skip_embed(x)  # (B, C, H, W, D) For the first skip connection (no downsampling)
        x2 = self.patch_embed(x)  # (B, C, H', W', D') For the second skip connection (downsampling)
        x1 = x1.permute(0, 2, 3, 4, 1).contiguous()  # (B, H, W, D, C)
        x2 = x2.permute(0, 2, 3, 4, 1).contiguous()  # (B, H', W', D', C)
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


class WindowAttention3D(nn.Module):  ## REMEMBER TO ADD RELATIVE POSITION BIAS AND MASKING TO ATTN
    def __init__(self, embed_dim, window_size, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_scale = self.head_dim ** -0.5  # To improve training stability
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B_, N, C = x.shape  # B_ = number of windows * batch, N = tokens per window, C = embed_dim
        qkv = self.qkv(x)  # (B_, N, 3×C) --> calculate concatenated QKV for each token
        # Split QKV into Q,K,V, then split each of those in head_dim chunks across heads
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # Each has dimension (B_, num_heads, N, head_dim)
        # Each head calculates its own scaled attention scores between all N tokens of each window
        attn = (q @ k.transpose(-2, -1)) * self.attn_scale  # (B_, num_heads, N_queries, N_keys)
        # Softmax is applied over the keys dimension to get a probability distribution of the keys for each query
        attn = attn.softmax(dim=-1)
        # Calculate the weighted attention, concatenate heads' outputs along the embedding dimension
        attn = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.out_proj(attn)  # Fuse the info from all heads


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0), mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(embed_dim)  # LayerNorm to stabilize training and improve convergence
        self.attn = WindowAttention3D(embed_dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio))  # Paper design choice

    def forward(self, x):
        orig_shape = x.shape
        residual = x  # The input will be used later for a residual connection
        x = self.norm1(x)  # Apply 1st LN
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))

        x_windows = window_partition(x, self.window_size)  # (num_windows * B, window_size^3, C)
        attn_windows = self.attn(x_windows)
        x = window_reverse(attn_windows, self.window_size, orig_shape)  # (B, H, W, D, C)
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))

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
            print("Merging Padded Shape:", x.shape)

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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            # Padding = 1 so that output size = input size (needed to later fuse with skip connections)
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),  # Standard CNN kernel size
            nn.InstanceNorm3d(out_channels),  # Better for CNNs than LayerNorm
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv_block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.skip_channels = skip_channels
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = ConvBlock(in_channels + skip_channels if skip_channels > 0 else in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)  # Upsample output from deeper layer
        print("Upsampled X Shape:", x.shape)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)  # Concatenate upsampled tensor with skip connection
            print("Concat Shape:", x.shape)
        x = self.conv(x)  # Apply the two convolution layers with normalization and activation
        return x


class SwinUNETREncoder3D(nn.Module):  # Whole encoding pipeline
    def __init__(self, in_channels, patch_size, embed_dims, num_heads, window_size):
        super().__init__()
        self.window_size = window_size
        self.num_stages = len(embed_dims)
        self.patch_embed = PatchEmbedding3D(in_channels=in_channels, patch_size=patch_size, embed_dim=embed_dims[0])
        self.trans_blocks = nn.ModuleList()
        self.merge_layers = nn.ModuleList()
        for i in range(self.num_stages):
            stage_blocks = nn.Sequential(
                SwinTransformerBlock3D(embed_dim=embed_dims[i], num_heads=num_heads[i],
                                       window_size=window_size, shift_size=(0, 0, 0)),
                SwinTransformerBlock3D(embed_dim=embed_dims[i], num_heads=num_heads[i],
                                       window_size=window_size, shift_size=(1, 1, 1))
            )
            self.trans_blocks.append(stage_blocks)
            if i < self.num_stages - 1:
                self.merge_layers.append(PatchMerging3D(embed_dims[i]))

    def forward(self, x):
        skips = []
        skip_embed, x = self.patch_embed(x)
        skips.extend([skip_embed, x])
        print("Patch Embedding Output Shape:", x.shape)
        for i in range(self.num_stages):
            x = self.trans_blocks[i](x)
            print(f"Stage {i + 1} Output Shape (before merging):", x.shape)
            if i < self.num_stages - 1:
                x = self.merge_layers[i](x)
                print(f"Stage {i + 1} Output Shape (after merging):", x.shape)
                if i < self.num_stages - 2:
                    skips.append(x)  # Save the transformer output after stages 1-3 for the skip connection

        return x, skips


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
            print(f"Decoder Level {5 - i}, X and Skip Shape:", x.shape, skip.shape)
            x = stage(x, skip)
        return x


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.head = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.head(x)  # (B, num_classes, D, H, W)


### TESTING SECTION ###
if __name__ == "__main__":
    # Dummy variables
    input_shape = (2, 191, 191, 191, 4)
    patch_size = (2, 2, 2)
    embed_dims = [48, 96, 192, 384, 768]
    num_heads = [3, 6, 12, 24, 48]
    window_size = (2, 2, 2)
    x = torch.randn(input_shape)
    print("Raw Input Shape:", x.shape)

    # Encoding Stage (Linear Embedding + Swin Transformers + Merging Layers)
    encoder = SwinUNETREncoder3D(in_channels=x.shape[4], patch_size=patch_size, embed_dims=embed_dims,
                                 num_heads=num_heads, window_size=window_size)
    encoder_output, skips = encoder(x)
    print("Encoder Output Shape:", encoder_output.shape)
    for i, skip in enumerate(skips):
        print(f"Skip {i + 1} shape: {skip.shape}")

    # Make Skips and Output Shape (B, C, D, H, W) for Decoder
    encoder_output = encoder_output.permute(0, 4, 3, 1, 2).contiguous()  # (B, C, D, H, W)
    skips = [s.permute(0, 4, 3, 1, 2).contiguous() for s in skips]  # (B, C, D, H, W)

    # Decoding Stage (CNN + Upsampling Blocks)
    decoder = SwinUNETRDecoder3D(embed_dims)
    decoder_output = decoder(encoder_output, skips)
    print("Decoder Output Shape:", decoder_output.shape)  # Should approach patch-embedded input resolution

    # Obtain Final Segmented Output
    segmentation_head = SegmentationHead(in_channels=embed_dims[0], num_classes=3)  # Adjust num_classes as needed
    output = segmentation_head(decoder_output)
    print("Final Output Shape:", output.shape)

