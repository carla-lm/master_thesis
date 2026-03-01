import os
import glob
import json
from monai import data
from sklearn.model_selection import train_test_split
from transforms import get_supervised_transforms, get_ssl_transforms


def data_split_brats_ssl(data_dir, split_file, fold, seed=42, train_fraction=1.0):
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

    # All SSL files used for training (no validation during pretraining)
    ssl_train_files = ssl_files
    # Split finetuning fold into 70% train, 15% val, 15% test
    train_files, remaining = train_test_split(finetune_files, test_size=0.3, random_state=seed)
    val_files, test_files = train_test_split(remaining, test_size=0.5, random_state=seed)
    if train_fraction < 1.0:
        train_files, _ = train_test_split(train_files, train_size=train_fraction, random_state=seed)
    return ssl_train_files, train_files, val_files, test_files


def data_split_numorph_ssl(data_dir, seed=42, train_fraction=1.0):
    pairs = [("c075_images_final_224_64", "c075_cen_final_224_64"),
             ("c121_images_final_224_64", "c121_cen_final_224_64_1")]

    image_files, label_files = [], []
    for img_dir, label_dir in pairs:
        img_paths = sorted(glob.glob(os.path.join(data_dir, img_dir, "*.nii")))
        lbl_paths = sorted(glob.glob(os.path.join(data_dir, label_dir, "*.nii")))
        image_files.extend(img_paths)
        label_files.extend(lbl_paths)

    # Split into 80% SSL pretraining and 20% fine-tuning sets
    ssl_imgs, finetune_imgs, _, finetune_labels = train_test_split(
        image_files, label_files, test_size=0.2, random_state=seed)

    # All SSL files are used for training (no validation during pretraining)
    ssl_train_files = [{"image": i} for i in ssl_imgs]

    # Split fine-tuning into 70% train, 15% val, 15% test
    train_imgs, remaining_imgs, train_labels, remaining_labels = train_test_split(
        finetune_imgs, finetune_labels, test_size=0.3, random_state=seed)
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        remaining_imgs, remaining_labels, test_size=0.5, random_state=seed)

    if train_fraction < 1.0:
        train_imgs, _, train_labels, _ = train_test_split(
            train_imgs, train_labels, train_size=train_fraction, random_state=seed)

    train_files = [{"image": i, "label": l} for i, l in zip(train_imgs, train_labels)]
    val_files = [{"image": i, "label": l} for i, l in zip(val_imgs, val_labels)]
    test_files = [{"image": i, "label": l} for i, l in zip(test_imgs, test_labels)]

    return ssl_train_files, train_files, val_files, test_files


def data_split_selma_ssl(data_dir, seed=42, train_fraction=1.0):
    entity_dirs = ["Cells", "Nuclei", "Vessels"]
    # Unannotated data (for SSL pretraining)
    ssl_imgs = []
    unannotated_dir = os.path.join(data_dir, "Unannotated")
    for entity_idx, entity in enumerate(entity_dirs):
        patches_root = os.path.join(unannotated_dir, entity, "patches")
        nii_files = sorted(glob.glob(os.path.join(patches_root, "**", "*.nii.gz"), recursive=True))
        for f in nii_files:
            ssl_imgs.append({"image": [f]})

    # All unannotated data used for SSL training (no validation during pretraining)
    ssl_train_files = ssl_imgs

    # Annotated data
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

            for rp in raw_dict[key]:
                finetune_files.append({
                    "image": [rp],
                    "label": lbl
                })
                finetune_entity_labels.append(entity_idx)

    # Stratified split into 70% train, 15% val, 15% test
    train_files, remaining, train_entity_labels, remaining_entity_labels = train_test_split(
        finetune_files, finetune_entity_labels, test_size=0.3, random_state=seed, stratify=finetune_entity_labels
    )
    val_files, test_files, _, _ = train_test_split(
        remaining, remaining_entity_labels, test_size=0.5, random_state=seed, stratify=remaining_entity_labels
    )
    if train_fraction < 1.0:
        train_files, _, train_entity_labels, _ = train_test_split(
            train_files, train_entity_labels, train_size=train_fraction, random_state=seed,
            stratify=train_entity_labels)
    return ssl_train_files, train_files, val_files, test_files


def ssl_data_loader(dataset_type, batch_size, roi, data_dir, ssl_mode=None, split_file=None, fold=None,
                    train_fraction=1.0):
    if dataset_type == "brats":
        if split_file is None or fold is None:
            raise ValueError("For the BraTS dataset you must provide a split file and a fold")
        ssl_train_files, train_files, val_files, test_files = data_split_brats_ssl(data_dir, split_file,
                                                                                   fold, seed=42,
                                                                                   train_fraction=train_fraction)

    elif dataset_type == "numorph":
        ssl_train_files, train_files, val_files, test_files = data_split_numorph_ssl(data_dir, seed=42,
                                                                                     train_fraction=train_fraction)

    elif dataset_type == "selma":
        ssl_train_files, train_files, val_files, test_files = data_split_selma_ssl(data_dir, seed=42,
                                                                                   train_fraction=train_fraction)

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    print(f"SSL pretrain samples: {len(ssl_train_files)} "
          f"| Fine-tune train samples: {len(train_files)} | Fine-tune val samples: {len(val_files)} "
          f"| Fine-tune test samples: {len(test_files)}")

    # Apply transforms and create DataLoaders for each set
    ssl_train_dataloader = None
    train_dataloader = None
    val_dataloader = None
    test_dataloader = None

    if ssl_mode is not None:  # Only fill the ssl dataloader in ssl training, not in finetuning
        ssl_transforms = get_ssl_transforms(dataset_type=dataset_type, roi=roi, ssl_mode=ssl_mode)
        ssl_train_dataset = data.Dataset(data=ssl_train_files, transform=ssl_transforms)
        ssl_train_dataloader = data.DataLoader(ssl_train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=8, pin_memory=True, persistent_workers=True)
    else:  # Only fill the finetuning dataloader when doing supervised training
        train_transforms, val_transforms = get_supervised_transforms(dataset_type=dataset_type, roi=roi)
        train_dataset = data.Dataset(data=train_files, transform=train_transforms)
        train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=8, pin_memory=True, persistent_workers=True)

        val_dataset = data.Dataset(data=val_files, transform=val_transforms)
        val_dataloader = data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                         num_workers=2, pin_memory=True, persistent_workers=False)

        test_dataset = data.Dataset(data=test_files, transform=val_transforms)
        test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                          num_workers=2, pin_memory=True, persistent_workers=False)

    return ssl_train_dataloader, train_dataloader, val_dataloader, test_dataloader
