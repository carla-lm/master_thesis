# Import necessary packages
import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from swinunetr_ssl import SSLSwinUNETR3D
from ssl_data_loading import ssl_data_loader
from utils import visualize_mask_overlay
from monai.losses import SSIMLoss

ssim_loss_fn = SSIMLoss(spatial_dims=3, reduction='mean')
l1_loss_fn = torch.nn.L1Loss()


def reconstruction_loss(img, recon, mask, ssim_weight=0.5):
    # Apply mask to only compare masked voxels for L1 loss
    mask = mask.to(dtype=img.dtype)
    img_masked = img * mask
    recon_masked = recon * mask
    # L1 loss on masked region only
    loss_l1 = l1_loss_fn(recon_masked, img_masked)
    # SSIM loss on entire image
    loss_ssim = ssim_loss_fn(img, recon + 1e-6)
    total_loss = loss_l1 + ssim_weight * loss_ssim
    return total_loss, loss_l1, loss_ssim


class SSLLitSwinUNETR(pl.LightningModule):
    def __init__(self, in_channels, patch_size, window_size, embed_dims, num_heads, lr, epochs, mask_ratio):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.epochs = epochs
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.window_size = window_size
        self.mask_ratio = mask_ratio
        self.model = SSLSwinUNETR3D(in_channels=self.in_channels, patch_size=self.patch_size,
                                    embed_dims=self.embed_dims, num_heads=self.num_heads,
                                    window_size=self.window_size, mask_ratio=self.mask_ratio)

    def forward(self, x):
        # Prepare and embed the input
        # print("Raw Input Shape:", x.shape)
        recon, mask = self.model(x)
        # print("Decoder Output Shape:", recon.shape)
        # print("Mask Shape:", mask.shape)
        return recon, mask

    def training_step(self, batch, batch_idx):
        x = batch["image"].to(self.device)
        recon, mask = self.forward(x)
        # Compute loss voxel-wise
        loss, loss_l1, loss_ssim = reconstruction_loss(x, recon, mask)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_l1", loss_l1, on_step=False, on_epoch=True)
        self.log("train_ssim", loss_ssim, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"].to(self.device)
        recon, mask = self.forward(x)
        val_loss, val_l1, val_ssim = reconstruction_loss(x, recon, mask)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_l1", val_l1, on_step=False, on_epoch=True)
        self.log("val_ssim", val_ssim, on_step=False, on_epoch=True)
        return val_loss

    def on_validation_epoch_end(self):  # Only test one sample per val epoch instead of all val batch samples
        batch = next(iter(self.val_loader))
        x = batch["image"][0].to(self.device)  # [0] to take the first sample in the batch
        x = x.unsqueeze(0)  # Add batch dimension as previous line removes it
        recon, mask = self.forward(x)
        visualize_mask_overlay(x, mask, recon, filename=f"epoch_{self.current_epoch:03d}", run_name=lit_model.run_name)

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
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=96)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--roi', type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    parser.add_argument('--experiment', type=int, default=1)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--data', type=str, default="numorph")
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
    mask_ratio = args.mask_ratio

    # Set relevant paths and load data
    if args.data == "brats":
        in_channels = 4
        out_channels = in_channels  # In SSL I am reconstructing the original image, not predicting classes
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_brats")  # Path is current directory + data_brats
        split_file = os.path.join(data_dir, "data_split.json")  # Json path is data directory + json filename
        ssl_train_loader, ssl_val_loader, train_loader, val_loader = ssl_data_loader(dataset_type=args.data,
                                                                                     batch_size=batch_size,
                                                                                     data_dir=data_dir,
                                                                                     split_file=split_file, fold=fold,
                                                                                     roi=roi)
    elif args.data == "numorph":
        in_channels = 1
        out_channels = in_channels  # In SSL I am reconstructing the original image, not predicting classes
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_numorph")
        ssl_train_loader, ssl_val_loader, train_loader, val_loader = ssl_data_loader(dataset_type=args.data,
                                                                                     batch_size=batch_size,
                                                                                     data_dir=data_dir,
                                                                                     roi=roi)
    elif args.data == "selma":
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_selma")

    else:
        raise ValueError("Unknown dataset")

    # Set up lightning module
    lit_model = SSLLitSwinUNETR(in_channels=in_channels, embed_dims=embed_dims,
                                num_heads=num_heads, patch_size=patch_size, window_size=window_size,
                                epochs=epochs, lr=lr, mask_ratio=mask_ratio)

    # For the on_validation_epoch_end operations
    lit_model.val_loader = ssl_val_loader
    lit_model.run_name = f"Run_{args.run}"

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
        filename="best-model-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",  # Metric used to decide best model
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
        check_val_every_n_epoch=15,
    )

    trainer.fit(lit_model, ssl_train_loader, ssl_val_loader, ckpt_path=resume_ckpt)
