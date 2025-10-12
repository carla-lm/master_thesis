# Master Thesis Repository
## Overview
This repository contains a PyTorch Lightning training pipeline for 3D medical image segmentation using my or MONAI's SwinUNETR model. It supports multiple datasets (e.g., BraTS, Numorph) and is designed for flexible experimentation and checkpoint management. The Lightning module wraps the chosen SwinUNETR model and handles:
* Forward pass and loss computation
* Sliding window inference for validation
* Dice and IoU metrics
* Checkpointing, logging and optimizer scheduling

Training runs are reproducible and automatically logged under a checkpoints/ directory.
It also contains a script for testing a pretrained model on some random samples: it performs a forward pass on a number of samples defined by the user and plots the original image vs the ground truth segmentation vs the predicted segmentation.

## Supported datasets
* BraTS: 4 input MRI modalities → 3 tumor class outputs
* Numorph: 1 input channel → 1 binary segmentation output

## Checkpointing
Two checkpoints are saved automatically:
* Best model: based on the highest validation Dice score
* Last model: every 10 epochs, overwriting the previous one

Each run is saved under: checkpoints/Experiment_X/version_Y/

The training and validation metrics are logged in a .csv file, and the hyperparameters are saved in a .yaml file.
The training can be resumed from a saved checkpoint by using the --resume_dir flag.

## How to use
### Training
* Basic training command: runs the training with default parameters and custom model for the indicated dataset → python training.py --data DATASET
* Resume training command: loads the latest checkpoint from the indicated folder and resumes training from the last saved epoch → python training.py --data DATASET --resume_dir checkpoints/Experiment_X/version_Y

There are several flags available to change the training parameters
* --epochs: set number of training epochs (default = 200)
* --lr: set the initial learning rate (it follows a Cosine Annealing LR Scheduler) (default = 1e-4)
* --batch: set the batch size for training (default = 1)
* --embed_dim: set the base embedding dimension size (default = 48)
* --fold: set which BraTS samples will be part of the validation set (default = 1)
* --roi: set the size of the training samples (default = [128, 128, 64])
* --val_every: set how often the validation epoch will be performed (default = 10)
* --experiment: set the experiment number for checkpointing purposes (default = 1)
* --data: set which dataset will be used (default: 'numorph')
* --resume_dir: set from which folder the checkpoint will be loaded
* --monai: if passed, the MONAI'S SwinUNETR model will be used instead of the custom model
### Testing
* Basic testing command: runs the testing with default parameters and custom model for the indicated dataset → python testing.py --data DATASET --ckpt_path CKPT_PATH
There are several flags available to change the testing parameters:
* --data: set which dataset will be used (default: 'numorph')
* --fold: set which BraTS samples will be part of the validation set (default = 1)
* --roi: set the size of the testing samples (default = [128, 128, 64])
* --monai: if passed, the MONAI'S SwinUNETR model will be used instead of the custom model
* --ckpt_path: set from which folder the pretrained model weights will be loaded
* --num_samples: set how many samples from the validation set will be used for testing the model
## Model Architecture
![screenshot](swin_model/assets/swin_unetr.png)
