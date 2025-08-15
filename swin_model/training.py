# Import necessary packages
import os
import json
import nibabel as nib
import torch
import random
import argparse
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
import monai.transforms as T
from monai import data


def data_loader(batch_size, data_dir, split_file, fold, roi):
    train_files, val_files = data_split(data_dir, split_file, fold)
    print(f"Found {len(train_files)} training samples and {len(val_files)} validation samples")
    train_trans, val_trans = get_transforms(roi)
    train_dataset = data.Dataset(data=train_files, transform=train_trans)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataset = data.Dataset(data=val_files, transform=val_trans)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, val_loader


def get_transforms(roi):
    train_transforms = T.Compose([  # create a sequential pipeline of preprocessing steps and augmentations
        T.LoadImaged(keys=["image", "label"]),  # Load label and image entries of the sample
        T.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),  # Convert label map to multi-channel format
        T.CropForegroundd(keys=["image", "label"], source_key="image",  # Crop out empty background
                          k_divisible=[roi[0], roi[1], roi[2]], allow_smaller=True),  # Ensure crop is divisible by roi
        # T.SpatialPadd(keys=["image", "label"], spatial_size=roi, mode="constant"),  # If crop is smaller, pad to roi
        T.RandSpatialCropd(keys=["image", "label"],  # Randomly crop a patch of size roi in image and label
                           roi_size=[roi[0], roi[1], roi[2]], random_size=False),  # Always return the given roi size
        T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),  # Flip with 50% chance in 1st spatial dimension
        T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),  # Flip with 50% chance in 2nd spatial dimension
        T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),  # Flip with 50% chance in 3rd spatial dimension
        T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),  # norm_vox = (vox - mean)/std
        T.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),  # Simulate variations in contrast
        T.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),  # Simulate variations in brightness
    ])

    val_transforms = T.Compose([  # Only deterministic preprocessing, no random augmentations
        T.LoadImaged(keys=["image", "label"]),
        T.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    return train_transforms, val_transforms


## WOULD BE NICE TO HAVE A GENERAL FUNCTION TO CONVERT LABEL MAPS TO ONE-HOT MULTICHANNEL MASKS

def data_split(data_dir, split_file, fold):
    with open(split_file) as file:
        data = json.load(file)  # Load json file's content into a Python dictionary
        data = data["training"]  # Extract the entries of "training" in the file

    train = []
    val = []
    for d in data:  # Each data entry has the keys fold, image, label
        for key in d:  # Add full path to the relative paths of the files
            if isinstance(d[key], list):  # If the key contains a list (image in different modalities)
                d[key] = [os.path.join(data_dir, image) for image in d[key]]  # Add full path to each list element
            elif isinstance(d[key], str):  # If the key contains a string (image label)
                d[key] = os.path.join(data_dir, d[key])  # Add full path to the label file

        if d["fold"] == fold:  # Add the files of the specified fold to the validation split for cross-validation
            val.append(d)
        else:
            train.append(d)

    return train, val


def visualize_data(data_dir):
    # Load a random image and label to visualize
    img_path = os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_flair.nii.gz")
    label_path = os.path.join(data_dir, "BraTS2021_00006/BraTS2021_00006_seg.nii.gz")
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
    # Get and apply the training transformations
    preview_trans = get_transforms((128, 128, 128))
    transformed = preview_trans[0](image_dict)  # 0 for all transforms, 1 for deterministic transforms
    img_t = transformed["image"]
    label_t = transformed["label"]
    print(f"Transformed image shape: {img_t.shape}, transformed label shape: {label_t.shape}")

    # Plot raw image and label
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


if __name__ == '__main__':
    # Set parameters that can be passed by the user
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=int, default=48)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=48)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--data', type=str, default='brats')
    args = parser.parse_args()

    # Define hyperparameters
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch
    fold = 1
    roi = (128, 128, 128)

    # Set relevant paths
    if args.data == 'brats':
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_brats")  # Path is current directory + data_brats
        split_file = os.path.join(data_dir, "data_split.json")  # Json path is data directory + json filename
    if args.data == 'selma':
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_selma")  # Path is current directory + data_selma

    # Load data
    train_loader, val_loader = data_loader(batch_size, data_dir, split_file, fold, roi)

    # Visualize data
    # visualize_data(data_dir)
