import os
import glob
import json
from monai import data, transforms as T
from sklearn.model_selection import train_test_split
from data_loading import get_transforms


def get_ssl_transforms(dataset_type, roi):  # No geometric transforms for MAE SSL
    train_transforms = T.Compose([T.LoadImaged(keys=["image"]),
                                  T.EnsureChannelFirstd(keys=["image"]),
                                  T.CropForegroundd(keys=["image"], source_key="image",
                                                    k_divisible=[roi[0], roi[1], roi[2]], allow_smaller=True),
                                  T.RandSpatialCropd(keys=["image"], roi_size=[roi[0], roi[1], roi[2]],
                                                     random_size=False),
                                  T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                                  T.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                                  T.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                                  ])

    return train_transforms


def data_split_brats_ssl(data_dir, split_file, fold, finetune_val_ratio=0.2, seed=42):
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

    # Split the finetuning fold into train and validation sets
    train_files, val_files = train_test_split(
        finetune_files, test_size=finetune_val_ratio, random_state=seed)
    return ssl_files, train_files, val_files


def data_split_numorph_ssl(data_dir, pretrain_ratio=0.8, finetune_val_ratio=0.2, seed=42):
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
        image_files, label_files, test_size=(1 - pretrain_ratio), random_state=seed)

    # Split fine-tuning set into training and validation sets
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        finetune_imgs, finetune_labels, test_size=finetune_val_ratio, random_state=seed)

    ssl_files = [{"image": i} for i in ssl_imgs]
    train_files = [{"image": i, "label": l} for i, l in zip(train_imgs, train_labels)]
    val_files = [{"image": i, "label": l} for i, l in zip(val_imgs, val_labels)]

    return ssl_files, train_files, val_files


def ssl_data_loader(dataset_type, batch_size, roi, data_dir, split_file=None, fold=None):
    if dataset_type == "brats":
        if split_file is None or fold is None:
            raise ValueError("For the BraTS dataset you must provide a split file and a fold")
        ssl_files, train_files, val_files = data_split_brats_ssl(data_dir, split_file, fold,
                                                                 finetune_val_ratio=0.2, seed=42)

    elif dataset_type == "numorph":
        ssl_files, train_files, val_files = data_split_numorph_ssl(data_dir, pretrain_ratio=0.8,
                                                                   finetune_val_ratio=0.2, seed=42)

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    print(f"SSL pretrain samples: {len(ssl_files)} | Fine-tune training samples: {len(train_files)} | Fine-tune "
          f"validation samples: {len(val_files)}")

    # Apply transforms to each set
    ssl_transforms = get_ssl_transforms(dataset_type, roi)
    train_transforms, val_transforms = get_transforms(dataset_type, roi)

    # Create DataLoaders for each set
    ssl_dataset = data.Dataset(data=ssl_files, transform=ssl_transforms)
    ssl_dataloader = data.DataLoader(ssl_dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=8, pin_memory=True, persistent_workers=True)

    train_dataset = data.Dataset(data=train_files, transform=train_transforms)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=8, pin_memory=True, persistent_workers=True)

    val_dataset = data.Dataset(data=val_files, transform=val_transforms)
    val_dataloader = data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                     num_workers=8, pin_memory=True, persistent_workers=True)

    return ssl_dataloader, train_dataloader, val_dataloader

