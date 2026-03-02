# SwinUNETR for 3D Medical Image Segmentation

Custom 3D SwinUNETR implementation with self-supervised pre-training (MIM and BYOL) and supervised fine-tuning for volumetric medical image segmentation. Developed as part of a master's thesis analyzing the impact of SSL pre-training on LSFM data.

Supports the BraTS 2021 brain tumor segmentation dataset and the Selma3D light-sheet fluorescence microscopy dataset.

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Lightning
- MONAI
- nibabel
- scikit-learn
- matplotlib
- pandas

## Project Structure

| File | Description |
|---|---|
| `swinunetr.py` | Custom 3D SwinUNETR architecture: patch embedding, Swin Transformer encoder, U-Net decoder, segmentation head |
| `swinunetr_ssl.py` | SSL model wrappers: MIM (masked image modeling with reconstruction head) and BYOL (online/target networks with projection and prediction heads) |
| `training.py` | Supervised training pipeline using PyTorch Lightning |
| `ssl_training.py` | Self-supervised pre-training pipeline (MIM and BYOL) using PyTorch Lightning |
| `testing.py` | Model evaluation: qualitative visualization and quantitative metrics (Dice, Jaccard) |
| `testing_ssl.py` | MIM reconstruction visualization on SSL pre-trained models |
| `data_loading.py` | Data splitting (SSL/train/val/test) and DataLoader creation for all datasets |
| `transforms.py` | MONAI transform pipelines for supervised training, MIM pre-training, and BYOL augmentations |
| `subvolumes.py` | Patch extraction utility for creating 3D subvolumes from large LSFM image stacks |
| `utils.py` | Visualization utilities for data samples, MIM reconstructions, and BYOL augmentations |
| `plot.py` | Plotting script for training/validation losses and metrics from CSV logs |

## Data Setup

Place your datasets under `TrainingData/` in the working directory:

```
TrainingData/
├── data_brats/
│   ├── BraTS2021_XXXXX/
│   │   ├── BraTS2021_XXXXX_t1.nii.gz
│   │   ├── BraTS2021_XXXXX_t1ce.nii.gz
│   │   ├── BraTS2021_XXXXX_t2.nii.gz
│   │   ├── BraTS2021_XXXXX_flair.nii.gz
│   │   └── BraTS2021_XXXXX_seg.nii.gz
│   └── data_split.json
└── data_selma/
    ├── Unannotated/
    │   ├── Cells/patches/
    │   ├── Nuclei/patches/
    │   └── Vessels/patches/
    └── Annotated/
        ├── Cells/
        │   ├── raw_patches/
        │   └── annotations/
        ├── Nuclei/
        └── Vessels/
```

## Usage

### 1. Self-Supervised Pre-Training

Pre-train the SwinUNETR encoder (and decoder for MIM) on unannotated data.

**MIM pre-training:**
```bash
python ssl_training.py --data brats --ssl_mode mim --experiment 1
```

**BYOL pre-training:**
```bash
python ssl_training.py --data selma --ssl_mode byol --experiment 2
```

| Argument | Default | Description |
|---|---|---|
| `--data` | *required* | Dataset: `brats`, `selma`, or `numorph` |
| `--ssl_mode` | *required* | SSL method: `mim` or `byol` |
| `--experiment` | *required* | Experiment number (used for checkpoint folder naming) |
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 5e-4 | Learning rate |
| `--batch` | 2 | Batch size |
| `--accum_steps` | 8 | Gradient accumulation steps (effective batch = batch x accum_steps) |
| `--embed_dim` | 96 | Base embedding dimension |
| `--roi` | 96 96 96 | Input volume size (height, width, depth) |
| `--mask_ratio` | 0.75 | Fraction of patches to mask (MIM only) |
| `--fold` | 1 | Data fold for BraTS |
| `--resume_dir` | None | Path to checkpoint directory to resume training |

Checkpoints are saved to `checkpoints_ssl/<experiment>/`.

### 2. Supervised Training / Fine-Tuning

Train from scratch or fine-tune from a pre-trained SSL checkpoint.

**Training from scratch:**
```bash
python training.py --data brats --experiment 1
```

**Fine-tuning from SSL checkpoint:**
```bash
python training.py --data brats --experiment 2 --pretrain_ckpt checkpoints_ssl/.../best-model.ckpt
```

**Training with reduced data:**
```bash
python training.py --data selma --experiment 3 --train_fraction 0.5
```

| Argument | Default | Description |
|---|---|---|
| `--data` | *required* | Dataset: `brats`, `selma`, or `numorph` |
| `--experiment` | *required* | Experiment number (used for checkpoint folder naming) |
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--batch` | 1 | Batch size |
| `--embed_dim` | 96 | Base embedding dimension |
| `--roi` | 96 96 96 | Input volume size (height, width, depth) |
| `--val_every` | 10 | Run validation every N epochs |
| `--fold` | 1 | Data fold for BraTS |
| `--pretrain_ckpt` | None | Path to SSL pre-trained checkpoint for weight transfer |
| `--train_fraction` | 1.0 | Fraction of training data to use (e.g., 0.1 for 10%) |
| `--resume_dir` | None | Path to checkpoint directory to resume training |

Checkpoints, metrics CSV, and hyperparameters YAML are saved to `checkpoints/<experiment>/`.

### 3. Testing

Evaluate a trained model on the test split.

**Quantitative metrics:**
```bash
python testing.py --data selma --test_type metrics --ckpt_path checkpoints/.../best-model.ckpt
```

**Qualitative visualization:**
```bash
python testing.py --data brats --test_type visualize --ckpt_path checkpoints/.../best-model.ckpt --num_samples 5
```

| Argument | Default | Description |
|---|---|---|
| `--data` | *required* | Dataset: `brats` or `selma` |
| `--test_type` | *required* | Test mode: `metrics` or `visualize` |
| `--ckpt_path` | *required* | Path to model checkpoint |
| `--roi` | 128 128 128 | ROI size for sliding window inference |
| `--fold` | 1 | Data fold for BraTS |
| `--num_samples` | 3 | Number of samples to visualize (visualize mode only) |

Metrics mode saves results to `testing/test_results_<dataset>.csv`.

### 4. MIM Reconstruction Visualization

Visualize MIM reconstructions from an SSL pre-trained model:

```bash
python testing_ssl.py --data brats --ckpt_path checkpoints_ssl/.../best-model.ckpt
```

### 5. Subvolume Extraction

Extract 3D patches from large LSFM image stacks (used to prepare Selma unannotated data):

```bash
python subvolumes.py
```

Configure the input/output paths and parameters directly in the script's `__main__` block.
