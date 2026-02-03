# Import necessary packages
import os
import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from swinunetr_ssl import MAESwinUNETR3D, BYOLSwinUNETR3D
from ssl_data_loading import ssl_data_loader
from utils import visualize_mask_overlay
from monai.losses import SSIMLoss

ssim_loss_fn = SSIMLoss(
    spatial_dims=3,
    data_range=1.0,  # Match [0,1] normalized data
    k1=0.01,  # For stability
    k2=0.03,  # For stability
    reduction='mean'
)
l1_loss_fn = torch.nn.L1Loss()
mse_loss_fn = torch.nn.MSELoss()


def reconstruction_loss(img, recon, mask, ssim_weight=0.1):
    # Apply mask to only compare masked voxels for L1 loss
    mask = mask.to(dtype=img.dtype)
    img_masked = img * mask
    recon_masked = recon * mask
    # L1 loss on masked region only
    loss_l1 = l1_loss_fn(recon_masked, img_masked)
    # SSIM loss on masked regions only for consistency with L1
    loss_ssim = ssim_loss_fn(img_masked, recon_masked)
    # Fall back to L1 only if SSIM fails with NaNs
    if torch.isnan(loss_ssim) or torch.isinf(loss_ssim):
        loss_ssim = torch.tensor(0.0, device=img.device, dtype=img.dtype)
        total_loss = loss_l1
    else:
        total_loss = loss_l1 + ssim_weight * loss_ssim

    return total_loss, loss_l1, loss_ssim


def byol_loss(pred1, pred2, target1, target2):
    # Normalize target and prediction vectors
    pred1_norm = F.normalize(pred1, dim=-1, p=2)
    pred2_norm = F.normalize(pred2, dim=-1, p=2)
    target1_norm = F.normalize(target1, dim=-1, p=2)
    target2_norm = F.normalize(target2, dim=-1, p=2)
    # Symmetric: predict view2 from view1 and vice versa
    loss1 = 2 - 2 * (pred1_norm * target2_norm).sum(dim=-1)
    loss2 = 2 - 2 * (pred2_norm * target1_norm).sum(dim=-1)
    total_loss = loss1 + loss2
    return total_loss.mean()  # Average loss across batches


class SSLLitSwinUNETR(pl.LightningModule):
    def __init__(self, in_channels, patch_size, window_size, embed_dims, num_heads, lr, epochs,
                 mask_ratio, ssl_mode):
        super().__init__()
        self.example_val_batch = None
        self.save_hyperparameters()
        self.lr = lr
        self.epochs = epochs
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.window_size = window_size
        self.mask_ratio = mask_ratio
        self.ssl_mode = ssl_mode

        # Initialize the model based on the SSL mode
        if ssl_mode == "mae":
            self.model = MAESwinUNETR3D(in_channels=in_channels, patch_size=patch_size,
                                        embed_dims=embed_dims, num_heads=num_heads,
                                        window_size=window_size, mask_ratio=mask_ratio)
        elif ssl_mode == "byol":
            self.model = BYOLSwinUNETR3D(in_channels=in_channels, patch_size=patch_size,
                                         embed_dims=embed_dims, num_heads=num_heads,
                                         window_size=window_size)

        else:
            raise ValueError(f"Unknown ssl_mode: {ssl_mode}")

    def forward(self, x, view1=None, view2=None):
        if self.ssl_mode == "mae":
            return self.model(x)
        elif self.ssl_mode == "byol":
            return self.model(view1, view2)

    def training_step(self, batch, batch_idx):
        if self.ssl_mode == "mae":
            x = batch["image"].to(self.device)
            recon, mask = self.model(x)
            loss, loss_l1, loss_ssim = reconstruction_loss(x, recon, mask)

            # Debug: log warning if NaN/Inf detected
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] NaN/Inf loss at epoch {self.current_epoch}, batch {batch_idx}")
                print(f"  recon: [{recon.min():.4f}, {recon.max():.4f}]")
                print(f"  input: [{x.min():.4f}, {x.max():.4f}]")
                print(f"  L1: {loss_l1.item():.6f}, SSIM: {loss_ssim.item():.6f}")

            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_l1", loss_l1, on_step=False, on_epoch=True)
            self.log("train_ssim", loss_ssim, on_step=False, on_epoch=True)
            return loss

        elif self.ssl_mode == "byol":
            v1 = batch["view1"].to(self.device)
            v2 = batch["view2"].to(self.device)
            pred1, pred2, target1, target2 = self.model(v1, v2)
            loss = byol_loss(pred1, pred2, target1, target2)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.example_val_batch = batch

        if self.ssl_mode == "mae":
            x = batch["image"].to(self.device)
            recon, mask = self.model(x)
            val_loss, val_l1, val_ssim = reconstruction_loss(x, recon, mask)
            self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_l1", val_l1, on_step=False, on_epoch=True)
            self.log("val_ssim", val_ssim, on_step=False, on_epoch=True)
            return val_loss

        elif self.ssl_mode == "byol":
            v1 = batch["view1"].to(self.device)
            v2 = batch["view2"].to(self.device)
            pred1, pred2, target1, target2 = self.model(v1, v2)
            val_loss = byol_loss(pred1, pred2, target1, target2)
            self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
            return val_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update target network after each training step
        if self.ssl_mode == "byol":
            self.model.update_moving_average()

    def on_validation_epoch_end(self):  # Only test one sample per val epoch instead of all val batch samples
        # Visualize reconstruction
        if self.ssl_mode == "mae":
            batch = self.example_val_batch
            x = batch["image"][0].to(self.device)  # [0] to take the first sample in the batch
            x = x.unsqueeze(0)  # Add batch dimension as previous line removes it
            recon, mask = self.model(x)
            visualize_mask_overlay(x, mask, recon, filename=f"epoch_{self.current_epoch:03d}", run_name=self.run_name)

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
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--accum_steps', type=int, default=8)
    parser.add_argument('--embed_dim', type=int, default=96)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--roi', type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--experiment', type=int, default=1)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--data', type=str, default="brats")
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument('--ssl_mode', type=str, default="mae", choices=["mae", "byol"])
    args = parser.parse_args()

    # Define hyperparameters
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch
    accum_steps = args.accum_steps
    fold = args.fold
    roi = tuple(args.roi)
    num_heads = [3, 6, 12, 24]
    window_size = (6, 6, 6)
    patch_size = (2, 2, 2)
    embed_dims = [args.embed_dim * (2 ** i) for i in range(5)]
    mask_ratio = args.mask_ratio
    ssl_mode = args.ssl_mode

    print(f"SSL Mode: {ssl_mode}")
    print(f"Batch size: {batch_size}, accumulation steps: {accum_steps}, effective batch: {batch_size*accum_steps}")

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
                                                                                     roi=roi, ssl_mode=ssl_mode)
    elif args.data == "numorph":
        in_channels = 1
        out_channels = in_channels  # In SSL I am reconstructing the original image, not predicting classes
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_numorph")
        ssl_train_loader, ssl_val_loader, train_loader, val_loader = ssl_data_loader(dataset_type=args.data,
                                                                                     batch_size=batch_size,
                                                                                     data_dir=data_dir,
                                                                                     roi=roi, ssl_mode=ssl_mode)
    elif args.data == "selma":
        in_channels = 1
        out_channels = in_channels  # In SSL I am reconstructing the original image, not predicting classes
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_selma")
        ssl_train_loader, ssl_val_loader, train_loader, val_loader = ssl_data_loader(dataset_type=args.data,
                                                                                     batch_size=batch_size,
                                                                                     data_dir=data_dir,
                                                                                     roi=roi, ssl_mode=ssl_mode)

    else:
        raise ValueError("Unknown dataset")

    # Set up lightning module
    lit_model = SSLLitSwinUNETR(in_channels=in_channels, embed_dims=embed_dims,
                                num_heads=num_heads, patch_size=patch_size, window_size=window_size,
                                epochs=epochs, lr=lr, mask_ratio=mask_ratio,
                                ssl_mode=ssl_mode)

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
        devices="auto",
        strategy="auto",
        callbacks=[best_check, last_check],
        logger=logger,
        check_val_every_n_epoch=15,
        precision="16-mixed",
        accumulate_grad_batches=accum_steps  # Effective batch size = batch_size * acc_steps
    )

    trainer.fit(lit_model, ssl_train_loader, ssl_val_loader, ckpt_path=resume_ckpt)
