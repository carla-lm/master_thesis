from monai import transforms as T
import copy
from itertools import combinations


def get_supervised_transforms(dataset_type, roi):
    if dataset_type == "brats":
        train_transforms = T.Compose([  # create a sequential pipeline of preprocessing steps and augmentations
            T.LoadImaged(keys=["image", "label"]),  # Load label and image entries of the sample
            T.EnsureChannelFirstd(keys=["image", "label"]),
            T.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),  # Convert label map to multi-channel format
            T.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.1, upper=99.9, b_min=0.0,
                                              b_max=1.0, clip=True, channel_wise=True),
            T.CropForegroundd(keys=["image", "label"], source_key="image",  # Crop out empty background
                              k_divisible=[roi[0], roi[1], roi[2]], allow_smaller=True),
            # Ensure crop is divisible by roi
            T.RandSpatialCropd(keys=["image", "label"],  # Randomly crop a patch of size roi in image and label
                               roi_size=[roi[0], roi[1], roi[2]], random_size=False),
            # Always return the given roi size
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            # Flip with 50% chance in 1st spatial dimension
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            # Flip with 50% chance in 2nd spatial dimension
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # Flip with 50% chance in 3rd spatial dimension
            T.RandScaleIntensityd(keys="image", factors=0.1, prob=0.8),  # Simulate variations in contrast
            T.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.8),  # Simulate variations in brightness
        ])

        val_transforms = T.Compose([  # Only deterministic preprocessing, no random augmentations
            T.LoadImaged(keys=["image", "label"]),
            T.EnsureChannelFirstd(keys=["image", "label"]),
            T.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            T.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.1, upper=99.9, b_min=0.0,
                                              b_max=1.0, clip=True, channel_wise=True),
        ])

    elif dataset_type in ["numorph", "selma"]:
        train_transforms = T.Compose([
            T.LoadImaged(keys=["image", "label"]),
            T.EnsureChannelFirstd(keys=["image", "label"]),  # This adds the single channel as a fourth dimension
            T.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.1, upper=99.9, b_min=0.0,
                                              b_max=1.0, clip=True, channel_wise=True),
            T.RandSpatialCropd(keys=["image", "label"], roi_size=[roi[0], roi[1], roi[2]], random_size=False),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            T.RandScaleIntensityd(keys="image", factors=0.1, prob=0.8),
            T.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.8),
        ])

        val_transforms = T.Compose([
            T.LoadImaged(keys=["image", "label"]),
            T.EnsureChannelFirstd(keys=["image", "label"]),
            T.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.1, upper=99.9, b_min=0.0,
                                              b_max=1.0, clip=True, channel_wise=True),
        ])

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return train_transforms, val_transforms


def get_safe_rotation_axes(roi):
    axes = []
    for i, j in combinations(range(3), 2):
        if roi[i] == roi[j]:
            axes.append((i, j))
    return axes


def get_byol_transforms(roi):
    # Get only the axis pairs with equal dimensions (safe for 90° rotation)
    safe_axes = get_safe_rotation_axes(roi)
    transform_list = [
        T.RandSpatialCropd(keys=["image"], roi_size=[roi[0], roi[1], roi[2]], random_size=False),
        T.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        T.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        T.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
    ]
    # Add rotations only for the safe axis pairs
    for axes in safe_axes:
        transform_list.append(T.RandRotate90d(keys=["image"], prob=0.5, spatial_axes=axes))

    # Intensity augmentations
    transform_list.extend([
        T.RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.8),
        T.RandShiftIntensityd(keys=["image"], offsets=0.2, prob=0.8),
        T.RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
    ])

    return T.Compose(transform_list)


def get_augmented_views(base_transform, aug_transform):
    def transform(data):  # return a callable that will be applied to each sample by MONAI
        data = base_transform(data)
        view1 = aug_transform(copy.deepcopy(data))
        view2 = aug_transform(copy.deepcopy(data))
        return {"image": data["image"], "view1": view1["image"], "view2": view2["image"]}

    return transform


def get_ssl_transforms(dataset_type, roi, ssl_mode):
    if dataset_type not in ["numorph", "brats", "selma"]:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if ssl_mode == "byol":
        base_transforms = T.Compose([
            T.LoadImaged(keys=["image"]),
            T.EnsureChannelFirstd(keys=["image"]),
            T.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.1, upper=99.9, b_min=0.0,
                                              b_max=1.0, clip=True, channel_wise=True),
            T.ToTensord(keys=["image"]),
        ])
        aug_transforms = get_byol_transforms(roi)
        return get_augmented_views(base_transform=base_transforms, aug_transform=aug_transforms)

    elif ssl_mode == "mim":
        base_transforms = T.Compose([
            T.LoadImaged(keys=["image"]),
            T.EnsureChannelFirstd(keys=["image"]),
            T.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.1, upper=99.9, b_min=0.0,
                                              b_max=1.0, clip=True, channel_wise=True),
            T.RandSpatialCropd(keys=["image"], roi_size=[roi[0], roi[1], roi[2]],
                               random_size=False),
            T.ToTensord(keys=["image"]),
        ])
        return base_transforms

    else:
        raise ValueError(f"Unknown ssl_mode: {ssl_mode}")
