import os
import glob
import json
from monai import data, transforms as T
from sklearn.model_selection import train_test_split
from data_loading import get_transforms


def extract_first_channel(x):  # Get only C00 channel from vessels
    return x[0:1] if x.shape[0] > 1 else x


def get_ssl_transforms(dataset_type, roi):  # No geometric transforms for MAE SSL
    if dataset_type in ["numorph", "brats"]:
        transforms = T.Compose([T.LoadImaged(keys=["image"]),
                                T.EnsureChannelFirstd(keys=["image"]),
                                T.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.1, upper=99.9, b_min=0.0,
                                                                  b_max=1.0, clip=True, channel_wise=True),
                                T.CropForegroundd(keys=["image"], source_key="image",
                                                  k_divisible=[roi[0], roi[1], roi[2]], allow_smaller=True),
                                # T.SpatialPadd(keys="image", spatial_size=[roi[0], roi[1], roi[2]]),
                                T.RandSpatialCropd(keys=["image"], roi_size=[roi[0], roi[1], roi[2]],
                                                   random_size=False),
                                T.ToTensord(keys=["image"]),
                                ])
    elif dataset_type == "selma":
        transforms = T.Compose([T.LoadImaged(keys=["image"]),
                                T.EnsureChannelFirstd(keys=["image"]),
                                T.Lambdad(
                                    keys=["image"],
                                    func=extract_first_channel
                                ),
                                T.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.1, upper=99.9, b_min=0.0,
                                                                  b_max=1.0, clip=True, channel_wise=True),
                                T.CropForegroundd(keys=["image"], source_key="image",
                                                  k_divisible=[roi[0], roi[1], roi[2]], allow_smaller=True),
                                # T.SpatialPadd(keys="image", spatial_size=[roi[0], roi[1], roi[2]]),
                                T.RandSpatialCropd(keys=["image"], roi_size=[roi[0], roi[1], roi[2]],
                                                   random_size=False),
                                T.ToTensord(keys=["image"]),
                                ])
    return transforms


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
    entity_dirs = ["Alzheimer", "Cells", "Nuclei", "Vessels"]
    # Unannotated Data (for SSL pretraining)
    ssl_imgs = []
    ssl_entity_labels = []  # Track entity type for stratified splitting
    unannotated_dir = os.path.join(data_dir, "Unannotated")
    for entity_idx, entity in enumerate(entity_dirs):
        all_patches_dir = os.path.join(unannotated_dir, entity, "all_patches")
        nii_files = sorted(glob.glob(os.path.join(all_patches_dir, "*.nii.gz")))
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


def ssl_data_loader(dataset_type, batch_size, roi, data_dir, split_file=None, fold=None):
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

    # Apply transforms to each set
    ssl_transforms = get_ssl_transforms(dataset_type, roi)
    train_transforms, val_transforms = get_transforms(dataset_type, roi)

    # Create DataLoaders for each set
    ssl_train_dataset = data.Dataset(data=ssl_train_files, transform=ssl_transforms)
    ssl_train_dataloader = data.DataLoader(ssl_train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=8, pin_memory=True, persistent_workers=True)

    ssl_val_dataset = data.Dataset(data=ssl_val_files, transform=ssl_transforms)
    ssl_val_dataloader = data.DataLoader(ssl_val_dataset, batch_size=1, shuffle=False,
                                         num_workers=8, pin_memory=True, persistent_workers=True)

    train_dataset = data.Dataset(data=train_files, transform=train_transforms)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=8, pin_memory=True, persistent_workers=True)

    val_dataset = data.Dataset(data=val_files, transform=val_transforms)
    val_dataloader = data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                     num_workers=8, pin_memory=True, persistent_workers=True)

    return ssl_train_dataloader, ssl_val_dataloader, train_dataloader, val_dataloader
