import torch
import torch.nn.functional as F
import torch.nn as nn

# Notation: input is 3D image with shape (H, W, D, S) (height, width, depth, channels), there will be
# batches so the input has shape (B, H, W, D, S)
# The Swin UNETR creates non-overlapping patches of the input data and uses an embedding layer + window partition
# to create windows with a desired size for computing the self-attention.
# The encoded feature representations in the Swin transformer are fed to a CNN-decoder
# via skip connection at multiple resolutions. Final segmentation output consists of as many
# output channels as element types we want to segment.

## ADD PADDING TO THE FUNCTIONS

class PatchEmbedding3D(nn.Module):
    # Patch size and embedding dimension (defined as C in the paper) are modifiable hyperparameters
    def __init__(self, in_channels, patch_size=(4, 4, 4), embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.embed_layer = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, H, W, D, S)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # -> (B, S, H, W, D) as required by Conv3D
        x = self.embed_layer(x)  # -> (B, C, H', W', D')
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # -> (B, H', W', D', C)
        return x


def window_partition(x, window_size):
    # x: (B, H', W', D', C) (we will not include the ' in the code for simplicity)
    B, H, W, D, C = x.shape
    wh, ww, wd = window_size
    # Divide the patch embedding output into window-sized non-overlapping blocks
    x = x.view(
        B,
        H // wh, wh,
        W // ww, ww,
        D // wd, wd,
        C
    )  # (B, nH, wh, nW, ww, nD, wd, C)
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
        assert H % 2 == 0 and W % 2 == 0 and D % 2 == 0, "Input dimensions must be even"
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


class SwinUNETREncoder3D(nn.Module):  # Whole encoding pipeline
    def __init__(self, embed_dims, num_heads, window_size):
        super().__init__()
        self.window_size = window_size
        self.num_stages = len(embed_dims)
        self.blocks = nn.ModuleList()
        self.mergers = nn.ModuleList()
        for i in range(self.num_stages):
            stage_blocks = nn.Sequential(
                SwinTransformerBlock3D(embed_dim=embed_dims[i], num_heads=num_heads[i],
                                       window_size=window_size, shift_size=(0, 0, 0)),
                SwinTransformerBlock3D(embed_dim=embed_dims[i], num_heads=num_heads[i],
                                       window_size=window_size, shift_size=(1, 1, 1))
            )
            self.blocks.append(stage_blocks)
            if i < self.num_stages - 1:
                self.mergers.append(PatchMerging3D(embed_dims[i]))

    def forward(self, x):
        skips = []
        for i in range(self.num_stages):
            x = self.blocks[i](x)
            print(f"Stage {i + 1} Output Shape (before merging):", x.shape)
            skips.append(x)  # Save the transformer output before merging for the skip connection
            if i < self.num_stages - 1:
                x = self.mergers[i](x)
                print(f"Stage {i + 1} Output Shape (after merging):", x.shape)
        return x, skips


### TESTING SECTION ###
if __name__ == "__main__":
    # Dummy variables
    input_shape = (2, 96, 96, 96, 4)
    patch_size = (2, 2, 2)
    embed_dims = [48, 96, 192, 384]
    num_heads = [3, 6, 12, 24]
    window_size = (2, 2, 2)
    x = torch.randn(input_shape)
    print("Raw Input Shape:", x.shape)

    # Patch embedding
    patch_embed = PatchEmbedding3D(in_channels=x.shape[4], patch_size=patch_size, embed_dim=embed_dims[0])
    x_embed = patch_embed(x)
    print("Patch Embedding Output Shape:", x_embed.shape)

    # Swin Transformers
    encoder = SwinUNETREncoder3D(embed_dims=embed_dims, num_heads=num_heads, window_size=window_size)
    encoder_output, skips = encoder(x_embed)
    print("Output Shape:", encoder_output.shape)
