import torch
import numpy as np
import torch.nn as nn

# Notation: input is 3D image with shape (H, W, D, S) (height, width, depth, channels), there will be
# batches so the input has shape (B, H, W, D, S)
# The Swin UNETR creates non-overlapping patches of the input data and uses an embedding layer + window partition
# to create windows with a desired size for computing the self-attention.
# The encoded feature representations in the Swin transformer are fed to a CNN-decoder
# via skip connection at multiple resolutions. Final segmentation output consists of as many
# output channels as element types we want to segment.


class PatchEmbedding(nn.Module):
    # Patch size and embedding dimension (defined as C in the paper) are modifiable hyperparameters
    def __init__(self, in_channels, patch_size=(4, 4, 4), embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.embed_layer = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, H, W, D, S)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # -> (B, S, H, W, D) as required by Conv3d
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
    # Merge window grid and flatten the 3D windows
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



class SwinUNETR(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.out_channels = out_channels

        ## DEFINE SWIN TRANSFORMER
        ## DEFINE ENCODERS
        ## DEFINE DECODERS
        ## DEFINE OUTPUT
        ## DEFINE FORWARD PASS

class SwinTransformer(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads


### TESTING SECTION ###
if __name__ == "__main__":

    # Define dummy input
    B, H, W, D, S = 2, 32, 32, 32, 1  # Batch size, height, width, depth, channels
    patch_size = (4, 4, 4)
    embed_dim = 96
    window_size = (2, 2, 2)

    x = torch.randn(B, H, W, D, S)

    # Patch embedding
    patch_embed = PatchEmbedding(in_channels=S, patch_size=patch_size, embed_dim=embed_dim)
    x_embed = patch_embed(x)  # -> (B, H', W', D', C)

    print("Patch Embedded Shape:", x_embed.shape)

    # Partition into windows
    windows = window_partition(x_embed, window_size)
    print("Windows Shape:", windows.shape)

    # Reconstruct
    x_reconstructed = window_reverse(windows, window_size, x_embed.shape)
    print("Reconstructed Shape:", x_reconstructed.shape)

    # Verify reconstruction
    diff = (x_embed - x_reconstructed).abs().max()
    print("Max difference after reconstruction:", diff.item())
