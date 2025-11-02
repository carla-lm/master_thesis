# Import necessary packages
import os
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from swinunetr_ssl import SSLEncoder3D, SSLDecoder3D
from ssl_data_loading import ssl_data_loader


class SSLLitSwinUNETR(pl.LightningModule):
    def __init__(self, in_channels, out_channels, patch_size, window_size, embed_dims, num_heads, lr, epochs, mask_ratio):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.epochs = epochs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.window_size = window_size
        self.mask_ratio = mask_ratio
        # Instantiate my custom SwinUNETR encoder
        self.encoder = SSLEncoder3D(in_channels=in_channels, embed_dims=self.embed_dims, num_heads=self.num_heads,
                                    patch_size=self.patch_size, window_size=self.window_size,
                                    mask_ratio=self.mask_ratio)
        # Instantiate my custom reconstruction decoder
        self.decoder = SSLDecoder3D(embed_dim=self.embed_dims[-1], decoder_dim=self.embed_dims[-1],
                                    out_channels=self.out_channels)
        # Define loss function for masked patch reconstruction
        self.lossfunc = nn.L1Loss()  # L1 is more stable for medical data

    def forward(self, x):
        # Prepare and embed the input
        # print("Raw Input Shape:", x.shape)
        x = x.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, D, C)
        # Encode the masked input
        latent, mask = self.encoder(x)
        # print("Encoder Output Shape:", latent.shape)
        # Decode the latent representation to reconstruct the image
        latent_shape = latent.shape[1:4]  # (H_latent, W_latent, D_latent)
        latent_flat = latent.view(latent.shape[0], -1, self.embed_dims[-1])  # (B, N, C)
        reconstruction = self.decoder(latent_flat, *latent_shape)
        reconstruction = reconstruction[:, :, :x.shape[3], :x.shape[1], :x.shape[2]]  # Trim padding to match original
        # Upsample the mask and permute it, so it has the same shape as the reconstruction
        mask = mask.repeat_interleave(self.patch_size[0], dim=1) \
            .repeat_interleave(self.patch_size[1], dim=2) \
            .repeat_interleave(self.patch_size[2], dim=3)
        mask = mask[..., 0]  # Drop embedding dimension (B, H, W, D)
        mask = mask.unsqueeze(1)  # (B, 1, H, W, D)  # Add channel dimension, so it matches reconstruction shape
        mask = mask.permute(0, 1, 4, 2, 3)  # (B, 1, D, H, W)
        mask = mask[:, :, :x.shape[3], :x.shape[1], :x.shape[2]]  # Trim padding to match original
        # print("Upsampled Mask Shape:", mask.shape)
        # print("Decoder Output Shape:", reconstruction.shape)

        return reconstruction, mask

    def training_step(self, batch, batch_idx):
        x = batch["image"].to(self.device)
        recon, mask = self.forward(x)
        # Compute loss voxel-wise
        mask_bool = mask > 0.5  # My mask is a tensor of floats 0 and 1, convert it to boolean
        loss = self.lossfunc(recon * mask_bool, x * mask_bool)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')  # Trade off precision for performance

    # Set parameters that can be passed by the user
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=48)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--roi', type=int, nargs=3, default=[128, 128, 64])
    parser.add_argument('--val_every', type=int, default=10)
    parser.add_argument('--experiment', type=int, default=1)
    parser.add_argument('--data', type=str, default="brats")
    parser.add_argument("--resume_dir", type=str, default=None)
    args = parser.parse_args()

    # Define hyperparameters
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch
    fold = args.fold
    roi = tuple(args.roi)
    num_heads = [3, 6, 12, 24]
    window_size = (6, 6, 6)
    patch_size = (2, 2, 2)
    embed_dims = [args.embed_dim * (2 ** i) for i in range(5)]

    # Set relevant paths and load data
    if args.data == "brats":
        in_channels = 4
        out_channels = in_channels  # In SSL I am reconstructing the original image, not predicting classes
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_brats")  # Path is current directory + data_brats
        split_file = os.path.join(data_dir, "data_split.json")  # Json path is data directory + json filename
        ssl_loader, train_loader, val_loader = ssl_data_loader(dataset_type=args.data, batch_size=batch_size,
                                                               data_dir=data_dir,
                                                               split_file=split_file, fold=fold, roi=roi)
    elif args.data == "numorph":
        in_channels = 1
        out_channels = in_channels  # In SSL I am reconstructing the original image, not predicting classes
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_numorph")
        ssl_loader, train_loader, val_loader = ssl_data_loader(dataset_type=args.data, batch_size=batch_size,
                                                               data_dir=data_dir,
                                                               roi=roi)
    elif args.data == "selma":
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_selma")

    else:
        raise ValueError("Unknown dataset")

    # Set up lightning module
    lit_model = SSLLitSwinUNETR(in_channels=in_channels, out_channels=out_channels, embed_dims=embed_dims,
                                num_heads=num_heads, patch_size=patch_size, window_size=window_size,
                                epochs=epochs, lr=lr, mask_ratio=0.75)

    # Create checkpoint folder
    experiment = args.experiment
    run_name = f"Experiment_{experiment}"
    if args.resume_dir is not None:  # Create a new folder for continuation so that things don't get overwritten
        exp_name = os.path.basename(os.path.dirname(args.resume_dir))  # Experiment_X
        old_version = os.path.basename(args.resume_dir)  # version_Y
        new_version = f"{old_version}_cont"  # Specify that it is a continuation
        logger = CSVLogger(save_dir=os.path.dirname(os.path.dirname(args.resume_dir)),  # checkpoints
                           name=exp_name,
                           version=new_version)
        check_dir = logger.log_dir
    else:
        logger = CSVLogger(save_dir="checkpoints_ssl", name=run_name)
        check_dir = logger.log_dir

    print(f"Saving at {check_dir}")
    os.makedirs(check_dir, exist_ok=True)

    # Save best model
    best_check = pl.callbacks.ModelCheckpoint(
        dirpath=check_dir,
        filename="best-model-{epoch:02d}-{train_loss:.4f}",
        monitor="train_loss",  # Metric used to decide best model
        save_top_k=1,  # Keep only best checkpoint
        mode="min"
    )

    # Save a periodic checkpoint every 5 epochs
    last_check = pl.callbacks.ModelCheckpoint(
        dirpath=check_dir,
        filename="last",
        save_top_k=1,  # Overwrite previous checkpoint
        every_n_epochs=5  # Save model every 5 epochs
    )

    # If resume_dir is passed, resume training from last.ckpt from specified directory
    resume_ckpt = None
    if args.resume_dir is not None:
        last_ckpt = os.path.join(args.resume_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            print(f"Resuming training from {last_ckpt}")
            resume_ckpt = last_ckpt
        else:
            print("No last.ckpt found, starting from scratch")
    else:
        print("Starting training from scratch")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[best_check, last_check],
        logger=logger,
        precision="16-mixed",  # Automatic mixed precision for faster training
    )

    trainer.fit(lit_model, ssl_loader, ckpt_path=resume_ckpt)
