import os
import glob
import json
from sklearn.model_selection import train_test_split
from monai import data, transforms as T


def get_transforms(dataset_type, roi):
    if dataset_type == "brats":
        train_transforms = T.Compose([  # create a sequential pipeline of preprocessing steps and augmentations
            T.LoadImaged(keys=["image", "label"]),  # Load label and image entries of the sample
            T.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),  # Convert label map to multi-channel format
            T.CropForegroundd(keys=["image", "label"], source_key="image",  # Crop out empty background
                              k_divisible=[roi[0], roi[1], roi[2]], allow_smaller=True), # Ensure crop is divisible by roi
            # T.SpatialPadd(keys=["image", "label"], spatial_size=roi, mode="constant"),  # If crop is smaller, pad to roi
            T.RandSpatialCropd(keys=["image", "label"],  # Randomly crop a patch of size roi in image and label
                               roi_size=[roi[0], roi[1], roi[2]], random_size=False), # Always return the given roi size
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0), # Flip with 50% chance in 1st spatial dimension
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1), # Flip with 50% chance in 2nd spatial dimension
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2), # Flip with 50% chance in 3rd spatial dimension
            T.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0,
                                              b_max=1.0, clip=True, channel_wise=True),
            T.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),  # Simulate variations in contrast
            T.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),  # Simulate variations in brightness
        ])

        val_transforms = T.Compose([  # Only deterministic preprocessing, no random augmentations
            T.LoadImaged(keys=["image", "label"]),
            T.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            T.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0,
                                              b_max=1.0, clip=True, channel_wise=True),
        ])

    elif dataset_type == "numorph":
        train_transforms = T.Compose([
            T.LoadImaged(keys=["image", "label"]),
            T.EnsureChannelFirstd(keys=["image", "label"]),  # This adds the single channel as a fourth dimension
            T.CropForegroundd(keys=["image", "label"], source_key="image",
                              k_divisible=[roi[0], roi[1], roi[2]], allow_smaller=True),
            T.RandSpatialCropd(keys=["image", "label"], roi_size=[roi[0], roi[1], roi[2]], random_size=False),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            T.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0,
                                              b_max=1.0, clip=True, channel_wise=True),
            T.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            T.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ])

        val_transforms = T.Compose([
            T.LoadImaged(keys=["image", "label"]),
            T.EnsureChannelFirstd(keys=["image", "label"]),
            T.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0,
                                              b_max=1.0, clip=True, channel_wise=True),
        ])
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return train_transforms, val_transforms


def data_split_brats(data_dir, split_file, fold):
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


def data_split_numorph(data_dir, test_size, seed):
    pairs = [("c075_images_final_224_64", "c075_cen_final_224_64"),
             ("c121_images_final_224_64", "c121_cen_final_224_64_1"), ]

    image_files, label_files = [], []
    for img_dir, label_dir in pairs:
        img_paths = sorted(glob.glob(os.path.join(data_dir, img_dir, "*.nii")))
        lbl_paths = sorted(glob.glob(os.path.join(data_dir, label_dir, "*.nii")))
        image_files.extend(img_paths)
        label_files.extend(lbl_paths)

    train_imgs, val_imgs, train_labels, val_labels = train_test_split(image_files, label_files,
                                                                      test_size=test_size, random_state=seed)

    train = [{"image": i, "label": l} for i, l in zip(train_imgs, train_labels)]
    val = [{"image": i, "label": l} for i, l in zip(val_imgs, val_labels)]
    return train, val


def data_loader(dataset_type, batch_size, roi, data_dir, split_file=None, fold=None):
    if dataset_type == "brats":
        if split_file is None or fold is None:
            raise ValueError("For the BraTS dataset you must provide a split file and a fold")
        train_files, val_files = data_split_brats(data_dir, split_file, fold)

    elif dataset_type == "numorph":
        train_files, val_files = data_split_numorph(data_dir, test_size=0.2, seed=42)

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    print(f"Found {len(train_files)} training samples and {len(val_files)} validation samples")

    train_trans, val_trans = get_transforms(dataset_type, roi)
    train_dataset = data.Dataset(data=train_files, transform=train_trans)
    val_dataset = data.Dataset(data=val_files, transform=val_trans)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                 num_workers=8, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader


def eval_data_loader(dataset_type, roi, data_dir, split_file=None, fold=None):
    # Load the validation files as test data
    if dataset_type == "brats":
        if split_file is None or fold is None:
            raise ValueError("For the BraTS dataset you must provide a split file and a fold")
        _, test_files = data_split_brats(data_dir, split_file, fold)
    elif dataset_type == "numorph":
        _, test_files = data_split_numorph(data_dir, test_size=0.2, seed=42)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Use only deterministic transforms (same as val_transforms)
    _, test_trans = get_transforms(dataset_type, roi)

    test_dataset = data.Dataset(data=test_files, transform=test_trans)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Loaded {len(test_files)} test samples")
    return test_loader
