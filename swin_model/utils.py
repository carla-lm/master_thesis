import os
import nibabel as nib
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from data_loading import get_transforms
# matplotlib.use("TkAgg")


def visualize_data_numorph(data_dir, roi):
    # Define samples to be visualized
    img_path = os.path.join(data_dir, "c075_images_final_224_64/c0202_Training-Top3-[00x02].nii")
    label_path = os.path.join(data_dir, "c075_cen_final_224_64/c0202_Training-Top3-[00x02].nii")
    # Load the samples
    img = nib.load(img_path).get_fdata()
    label = nib.load(label_path).get_fdata()
    print(f"Image shape: {img.shape}, Label shape: {label.shape}")

    # Apply the transforms to the sample
    sample_dict = {"image": img_path, "label": label_path}
    transforms = get_transforms(dataset_type="numorph", roi=roi)[0]  # apply the train transforms
    transformed = transforms(sample_dict)
    img_t = transformed["image"]
    label_t = transformed["label"]
    print(f"Transformed image shape: {img_t.shape}, transformed label shape: {label_t.shape}")

    # Plot raw image and label
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img[:, :, 30], cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Original Label")
    plt.imshow(label[:, :, 30])
    plt.axis("off")

    # Plot transformed image with overlaid label
    plt.subplot(1, 3, 3)
    plt.title("Transformed Image")
    plt.imshow(img_t[0, :, :, 30], cmap="gray")
    plt.imshow(label_t[0, :, :, 30], alpha=0.3)
    plt.axis("off")

    plt.show()


def visualize_data_brats(data_dir, roi):
    # Define samples to be visualized
    img_path = os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_flair.nii.gz")
    label_path = os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_seg.nii.gz")
    # Load the samples
    img = nib.load(img_path).get_fdata()
    label = nib.load(label_path).get_fdata()
    print(f"Image shape: {img.shape}, label shape: {label.shape}")  # Should be (H, W, D)
    # Load whole sample dictionary (need a modality dimension for transform pipeline to work)
    image_dict = {
        "image": [
            os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_t1.nii.gz"),
            os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_t1ce.nii.gz"),
            os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_t2.nii.gz"),
            os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_flair.nii.gz")
        ],
        "label": os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_seg.nii.gz")
    }
    # Apply the transforms to the sample
    preview_trans = get_transforms(dataset_type="brats", roi=roi)
    transformed = preview_trans[0](image_dict)  # 0 for all transforms, 1 for deterministic transforms
    img_t = transformed["image"]
    label_t = transformed["label"]
    print(f"Transformed image shape: {img_t.shape}, transformed label shape: {label_t.shape}")  # Should be (C, H, W, D)

    # Plot raw image and label
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img[:, :, 78], cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Original Label")
    plt.imshow(label[:, :, 78])
    plt.axis("off")

    # Plot transformed image with overlaid label
    plt.subplot(1, 3, 3)
    plt.title("Transformed Image")
    plt.imshow(img_t[3, :, :, 78], cmap="gray")
    plt.imshow(label_t.sum(axis=0)[:, :, 78], alpha=0.3)
    plt.axis("off")
    plt.show()


def visualize_mask_overlay(x, mask, recon, filename="overlay.png"):
    # Save figure
    save_dir = "Figures"
    os.makedirs(save_dir, exist_ok=True)
    # Convert tensors to numpy arrays and discard batch and channel dimensions
    x = x[0, 0].detach().cpu().numpy()  # (B, C, D, H, W) --> (D, H, W)
    recon = recon[0, 0].detach().cpu().numpy()  # (B, C, D, H, W) --> (D, H, W)
    mask = mask[0, 0].detach().cpu().numpy()  # (B, 1, D, H, W) --> (D, H, W)

    # Select middle z-slice
    mid_z = x.shape[2] // 2
    x_slice = x[:, :, mid_z]
    recon_slice = recon[:, :, mid_z]
    mask_slice = mask[:, :, mid_z]

    # Overlay the mask onto the image
    overlay_masked = x_slice.copy()
    overlay_masked[mask_slice > 0] = -0.1  # Set image voxels where mask is applied to value -0.1

    # Plot each image
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(x_slice, cmap='gray')
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(overlay_masked, cmap='gray')
    axs[1].set_title("Masked Image")
    axs[1].axis('off')

    axs[2].imshow(recon_slice, cmap='gray')
    axs[2].set_title("Reconstructed Image")
    axs[2].axis('off')

    plt.tight_layout()
    # plt.show()
    # Save instead of showing
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved overlay figure at: {save_path}")


def reconstruction_loss(img, recon, mask):
    # Apply mask to only compare masked voxels
    mask = mask.to(dtype=img.dtype)
    img_masked = img * mask
    recon_masked = recon * mask
    loss_fn = nn.MSELoss()
    loss = loss_fn(recon_masked, img_masked)
    return loss

