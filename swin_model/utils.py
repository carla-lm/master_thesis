import os
import nibabel as nib
import matplotlib.pyplot as plt
from data_loading import get_transforms
from ssl_data_loading import get_ssl_transforms, get_byol_transforms
import copy
import matplotlib
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
    transforms = get_transforms(dataset_type="numorph", roi=roi)[1]  # apply the val transforms
    transformed = transforms(sample_dict)
    img_t = transformed["image"]
    label_t = transformed["label"]
    print(f"Transformed image shape: {img_t.shape}, transformed label shape: {label_t.shape}")
    show_z = img.shape[2] // 2

    # Plot raw image and label
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img[:, :, show_z], cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Original Label")
    plt.imshow(label[:, :, show_z])
    plt.axis("off")

    # Plot transformed image with overlaid label
    plt.subplot(1, 3, 3)
    plt.title("Transformed Image")
    plt.imshow(img_t[0, :, :, show_z], cmap="gray")
    # plt.imshow(label_t[0, :, :, show_z], alpha=0.3)
    plt.axis("off")

    plt.show()


def visualize_data_selma(data_dir, roi):
    # Define samples to be visualized
    img_path = os.path.join(data_dir, "Annotated/Nuclei/raw_patches", "patchvolume_002_0000.nii.gz")
    label_path = os.path.join(data_dir, "Annotated/Nuclei/annotations", "patchvolume_002.nii.gz")
    # Load the samples
    img = nib.load(img_path).get_fdata()
    label = nib.load(label_path).get_fdata()
    print(f"Image shape: {img.shape}, Label shape: {label.shape}")

    # Apply the transforms to the sample
    sample_dict = {"image": img_path, "label": label_path}
    transforms = get_transforms(dataset_type="selma", roi=roi)[1]  # apply the train transforms
    transformed = transforms(sample_dict)
    img_t = transformed["image"]
    label_t = transformed["label"]
    print(f"Transformed image shape: {img_t.shape}, transformed label shape: {label_t.shape}")
    show_z = img.shape[2] // 2

    # Plot raw image and label
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img[:, :, show_z], cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Original Label")
    plt.imshow(label[:, :, show_z])
    plt.axis("off")

    # Plot transformed image with overlaid label
    plt.subplot(1, 3, 3)
    plt.title("Transformed Image")
    plt.imshow(img_t[0, :, :, show_z], cmap="gray")
    # plt.imshow(label_t[0, :, :, show_z], alpha=0.3)
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
    transformed = preview_trans[1](image_dict)  # 0 for all transforms, 1 for deterministic transforms
    img_t = transformed["image"]
    label_t = transformed["label"]
    print(f"Transformed image shape: {img_t.shape}, transformed label shape: {label_t.shape}")  # Should be (C, H, W, D)
    show_z = img.shape[2] // 2

    # Plot raw image and label
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img[:, :, show_z], cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Original Label")
    plt.imshow(label[:, :, show_z])
    plt.axis("off")

    # Plot transformed image with overlaid label
    plt.subplot(1, 3, 3)
    plt.title("Transformed Image")
    plt.imshow(img_t[3, :, :, show_z], cmap="gray")
    # plt.imshow(label_t.sum(axis=0)[:, :, show_z], alpha=0.3)
    plt.axis("off")
    plt.show()


def visualize_mask_overlay(x, mask, recon, filename="overlay.png", run_name="Run"):
    # Save figure
    save_dir = os.path.join("Figures", run_name)
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


def visualize_byol_augmentations(dataset_type, data_dir, roi):
    # Define sample path based on dataset
    if dataset_type == "selma":
        img_path = os.path.join(data_dir, "Annotated/Nuclei/raw_patches", "patchvolume_002_0000.nii.gz")
        sample_dict = {"image": [img_path]}
    elif dataset_type == "brats":
        sample_dict = {
            "image": [
                os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_t1.nii.gz"),
                os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_t1ce.nii.gz"),
                os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_t2.nii.gz"),
                os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_flair.nii.gz")
            ]
        }
        img_path = sample_dict["image"][3]  # Use flair for display
    elif dataset_type == "numorph":
        img_path = os.path.join(data_dir, "c075_images_final_224_64/c0202_Training-Top3-[00x02].nii")
        sample_dict = {"image": img_path}
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Load original image
    original = nib.load(img_path).get_fdata()
    # Get base transforms
    base_transform = get_ssl_transforms(dataset_type, roi, ssl_mode="byol")
    # Apply base transforms
    base_result = base_transform(copy.deepcopy(sample_dict))
    base_img = base_result["image"]
    # Get BYOL augmentations
    aug_transform = get_byol_transforms(roi)
    # Create two augmented views from base result
    view1_result = aug_transform(copy.deepcopy(base_result))
    view2_result = aug_transform(copy.deepcopy(base_result))
    view1 = view1_result["image"]
    view2 = view2_result["image"]

    print("Original shape: ", original.shape)
    print("Base transform shape: ", base_img.shape)
    print("View 1 shape: ", view1.shape)
    print("View 2 shape: ", view2.shape)

    if dataset_type == "brats":  # Select flair modality
        base_img = base_img[3]  # (D, H, W)
        view1 = view1[3]  # (D, H, W)
        view2 = view2[3]  # (D, H, W)

    else:    # Convert tensors to numpy arrays and discard batch and channel dimensions
        base_img = base_img[0].detach().cpu().numpy()  # (C, D, H, W) --> (D, H, W)
        view1 = view1[0].detach().cpu().numpy()  # (C, D, H, W) --> (D, H, W)
        view2 = view2[0].detach().cpu().numpy()  # (C, D, H, W) --> (D, H, W)

    # Select middle z-slice
    orig_z = original.shape[2] // 2
    trans_z = view1.shape[2] // 2

    # Plot 4 images
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(original[:, :, orig_z], cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis('off')

    axs[1].imshow(base_img[:, :, orig_z], cmap='gray')
    axs[1].set_title("After Base Transforms")
    axs[1].axis('off')

    axs[2].imshow(view1[:, :, trans_z], cmap='gray')
    axs[2].set_title("View 1 (BYOL Aug)")
    axs[2].axis('off')

    axs[3].imshow(view2[:, :, trans_z], cmap='gray')
    axs[3].set_title("View 2 (BYOL Aug)")
    axs[3].axis('off')

    plt.tight_layout()
    plt.show()
