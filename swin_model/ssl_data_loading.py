import os
import glob
import json
import copy
from monai import data, transforms as T
from sklearn.model_selection import train_test_split
from data_loading import get_transforms


def get_byol_transforms(roi):
    transforms = T.Compose([
        # Each view can have a different section of the sample
        T.RandSpatialCropd(keys=["image"], roi_size=[roi[0], roi[1], roi[2]], random_size=False),
        # Spatial augmentations
        T.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        T.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        T.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
        T.RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 1)),
        # Stronger intensity augmentations to prevent collapse
        T.RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.8),
        T.RandShiftIntensityd(keys=["image"], offsets=0.2, prob=0.8),
        T.RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
        T.RandGaussianSmoothd(keys=["image"], prob=0.3, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
    ])
    return transforms


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

    elif ssl_mode == "mae":
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


def data_split_brats_ssl(data_dir, split_file, fold, seed=42):
    with open(split_file) as file:
        data = json.load(file)  # Load json file's content into a Python dictionary
        data = data["training"]  # Extract the entries of "training" in the file

    ssl_files = []
    finetune_files = []

    for d in data:
        if d["fold"] != fold:  # For SSL split
            image_paths = [os.path.join(data_dir, img) for img in d["image"]]  # Add full path to the data relative path
            ssl_files.append({"image": image_paths})  # Add only image paths (lists of modalities) and not the labels
        else:  # For finetune split
            for key in d:
                if isinstance(d[key], list):  # If the key contains a list (image in different modalities)
                    d[key] = [os.path.join(data_dir, image) for image in d[key]]  # Add full path to each list element
                elif isinstance(d[key], str):  # If the key contains a string (image label)
                    d[key] = os.path.join(data_dir, d[key])  # Add full path to the label file
            finetune_files.append(d)

    # Split the SSL set and finetuning fold into train and validation
    ssl_train_files, ssl_val_files = train_test_split(ssl_files, test_size=0.1, random_state=seed)
    train_files, val_files = train_test_split(finetune_files, test_size=0.2, random_state=seed)
    return ssl_train_files, ssl_val_files, train_files, val_files


def data_split_numorph_ssl(data_dir, seed=42):
    pairs = [("c075_images_final_224_64", "c075_cen_final_224_64"),
             ("c121_images_final_224_64", "c121_cen_final_224_64_1")]

    image_files, label_files = [], []
    for img_dir, label_dir in pairs:
        img_paths = sorted(glob.glob(os.path.join(data_dir, img_dir, "*.nii")))
        lbl_paths = sorted(glob.glob(os.path.join(data_dir, label_dir, "*.nii")))
        image_files.extend(img_paths)
        label_files.extend(lbl_paths)

    # Split into SSL pretraining and fine-tuning sets
    ssl_imgs, finetune_imgs, _, finetune_labels = train_test_split(
        image_files, label_files, test_size=0.2, random_state=seed)

    # Split fine-tuning and SSL sets into training and validation sets
    ssl_train_imgs, ssl_val_imgs = train_test_split(ssl_imgs, test_size=0.1, random_state=seed)

    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        finetune_imgs, finetune_labels, test_size=0.2, random_state=seed)

    ssl_train_files = [{"image": i} for i in ssl_train_imgs]
    ssl_val_files = [{"image": i} for i in ssl_val_imgs]
    train_files = [{"image": i, "label": l} for i, l in zip(train_imgs, train_labels)]
    val_files = [{"image": i, "label": l} for i, l in zip(val_imgs, val_labels)]

    return ssl_train_files, ssl_val_files, train_files, val_files


def data_split_selma_ssl(data_dir, seed=42):
    entity_dirs = ["Cells", "Nuclei", "Vessels"]
    # Unannotated Data (for SSL pretraining)
    ssl_imgs = []
    ssl_entity_labels = []  # Track entity type for stratified splitting
    unannotated_dir = os.path.join(data_dir, "Unannotated")
    for entity_idx, entity in enumerate(entity_dirs):
        patches_dir = os.path.join(unannotated_dir, entity, "patches")
        nii_files = sorted(glob.glob(os.path.join(patches_dir, "*.nii.gz")))
        for f in nii_files:
            ssl_imgs.append({"image": [f]})
            ssl_entity_labels.append(entity_idx)

    # Stratified split ensures proportional entity distribution in SSL train/val
    ssl_train_files, ssl_val_files = train_test_split(
        ssl_imgs, test_size=0.1, random_state=seed, stratify=ssl_entity_labels
    )

    # Annotated Data
    finetune_files = []
    finetune_entity_labels = []
    annotated_dir = os.path.join(data_dir, "Annotated")
    for entity_idx, entity in enumerate(entity_dirs):
        raw_dir = os.path.join(annotated_dir, entity, "raw_patches")
        lbl_dir = os.path.join(annotated_dir, entity, "annotations")
        raw_patches = sorted(glob.glob(os.path.join(raw_dir, "*.nii.gz")))
        labels = sorted(glob.glob(os.path.join(lbl_dir, "*.nii.gz")))
        # Match raw patches to their labels
        raw_dict = {}
        for rp in raw_patches:
            base = os.path.basename(rp)  # patchvolume_XXX_0000.nii.gz or _0001.nii.gz
            key = base.split("_")[-2]  # XXX
            raw_dict.setdefault(key, []).append(rp)

        for lbl in labels:
            key = os.path.basename(lbl).split("_")[-1].split(".")[0]  # patchvolume_XXX.nii.gz get XXX
            if key not in raw_dict:
                continue

            if entity == "Vessels":
                rps = sorted(raw_dict[key])
                c00 = [rp for rp in rps if rp.endswith("_0000.nii.gz")]  # Get only C00
                if len(c00) == 1:
                    finetune_files.append({
                        "image": c00,
                        "label": lbl
                    })
                    finetune_entity_labels.append(entity_idx)
            else:  # Single-channel entities
                for rp in raw_dict[key]:
                    finetune_files.append({
                        "image": [rp],
                        "label": lbl
                    })
                    finetune_entity_labels.append(entity_idx)

    # Stratified split ensures proportional entity distribution in finetune train/val
    train_files, val_files = train_test_split(
        finetune_files, test_size=0.2, random_state=seed, stratify=finetune_entity_labels
    )
    return ssl_train_files, ssl_val_files, train_files, val_files


def ssl_data_loader(dataset_type, batch_size, roi, data_dir, ssl_mode, split_file=None, fold=None):
    if dataset_type == "brats":
        if split_file is None or fold is None:
            raise ValueError("For the BraTS dataset you must provide a split file and a fold")
        ssl_train_files, ssl_val_files, train_files, val_files = data_split_brats_ssl(data_dir, split_file,
                                                                                      fold, seed=42)

    elif dataset_type == "numorph":
        ssl_train_files, ssl_val_files, train_files, val_files = data_split_numorph_ssl(data_dir, seed=42)

    elif dataset_type == "selma":
        ssl_train_files, ssl_val_files, train_files, val_files = data_split_selma_ssl(data_dir, seed=42)

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    print(f"SSL pretrain training samples: {len(ssl_train_files)} "
          f"| SSL pretrain validation samples: {len(ssl_val_files)}"
          f"| Fine-tune training samples: {len(train_files)} | Fine-tune "
          f"validation samples: {len(val_files)}")

    # Apply transforms and create DataLoaders for each set
    ssl_train_dataloader = None
    ssl_val_dataloader = None

    if ssl_mode is not None:  # Only fill the ssl dataloaders in ssl training, not in finetuning
        ssl_transforms = get_ssl_transforms(dataset_type=dataset_type, roi=roi, ssl_mode=ssl_mode)
        ssl_train_dataset = data.Dataset(data=ssl_train_files, transform=ssl_transforms)
        ssl_train_dataloader = data.DataLoader(ssl_train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=8, pin_memory=True, persistent_workers=True)

        ssl_val_dataset = data.Dataset(data=ssl_val_files, transform=ssl_transforms)
        ssl_val_dataloader = data.DataLoader(ssl_val_dataset, batch_size=1, shuffle=False,
                                             num_workers=8, pin_memory=True, persistent_workers=True)

    train_transforms, val_transforms = get_transforms(dataset_type=dataset_type, roi=roi)
    train_dataset = data.Dataset(data=train_files, transform=train_transforms)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=8, pin_memory=True, persistent_workers=True)

    val_dataset = data.Dataset(data=val_files, transform=val_transforms)
    val_dataloader = data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                     num_workers=8, pin_memory=True, persistent_workers=True)

    return ssl_train_dataloader, ssl_val_dataloader, train_dataloader, val_dataloader
