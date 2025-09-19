import os
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
from data_loading import get_transforms


def visualize_data_numorph(data_dir):
    # Define samples to be visualized
    img_path = os.path.join(data_dir, "c075_images_final_224_64/c0202_Training-Top3-[00x02].nii")
    label_path = os.path.join(data_dir, "c075_cen_final_224_64/c0202_Training-Top3-[00x02].nii")
    # Load the samples
    img = nib.load(img_path).get_fdata()
    label = nib.load(label_path).get_fdata()
    print(f"Image shape: {img.shape}, Label shape: {label.shape}")

    # Apply the transforms to the sample
    sample_dict = {"image": img_path, "label": label_path}
    transforms = get_transforms(dataset_type="numorph", roi=(64, 64, 64))[0]  # apply the train transforms
    transformed = transforms(sample_dict)
    img_t = transformed["image"]
    label_t = transformed["label"]
    print(f"Transformed image shape: {img_t.shape}, transformed label shape: {label_t.shape}")

    # Plot raw image and label
    matplotlib.use("TkAgg")
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


def visualize_data_brats(data_dir):
    # Define samples to be visualized
    img_path = os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_flair.nii.gz")
    label_path = os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_seg.nii.gz")
    # Load the samples
    img = nib.load(img_path).get_fdata()
    label = nib.load(label_path).get_fdata()
    print(f"Image shape: {img.shape}, label shape: {label.shape}")
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
    preview_trans = get_transforms(dataset_type="brats", roi=(128, 128, 128))
    transformed = preview_trans[0](image_dict)  # 0 for all transforms, 1 for deterministic transforms
    img_t = transformed["image"]
    label_t = transformed["label"]
    print(f"Transformed image shape: {img_t.shape}, transformed label shape: {label_t.shape}")

    # Plot raw image and label
    matplotlib.use("TkAgg")  # To see figure live
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
