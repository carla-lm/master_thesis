# Import necessary packages
import os
import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from swinunetr_ssl import MIMSwinUNETR3D, BYOLSwinUNETR3D
from utils import visualize_mask_overlay
from data_loading import ssl_data_loader
from monai.losses import SSIMLoss

ssim_loss_fn = SSIMLoss(spatial_dims=3, data_range=1.0,  # Match [0,1] normalized data
                        k1=0.01,  # For stability in dark regions
                        k2=0.03,  # For stability in uniform regions
                        reduction='mean')
l1_loss_fn = torch.nn.L1Loss()


def reconstruction_loss(img, recon, mask, ssim_weight=0.1):
    # Apply mask to only compare masked voxels for loss
    mask = mask.to(dtype=img.dtype)
    img_masked = img * mask
    recon_masked = recon * mask
    loss_l1 = l1_loss_fn(recon_masked, img_masked)
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
        if ssl_mode == "mim":
            self.model = MIMSwinUNETR3D(in_channels=in_channels, patch_size=patch_size,
                                        embed_dims=embed_dims, num_heads=num_heads,
                                        window_size=window_size, mask_ratio=mask_ratio)
        elif ssl_mode == "byol":
            self.model = BYOLSwinUNETR3D(in_channels=in_channels, patch_size=patch_size,
                                         embed_dims=embed_dims, num_heads=num_heads,
                                         window_size=window_size)

        else:
            raise ValueError(f"Unknown ssl_mode: {ssl_mode}")

    def forward(self, x, view1=None, view2=None):
        if self.ssl_mode == "mim":
            return self.model(x)
        elif self.ssl_mode == "byol":
            return self.model(view1, view2)

    def training_step(self, batch, batch_idx):
        if self.ssl_mode == "mim":
            x = batch["image"].to(self.device)
            recon, mask = self.model(x)
            loss, loss_l1, loss_ssim = reconstruction_loss(x, recon, mask)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_l1", loss_l1, on_step=False, on_epoch=True)
            self.log("train_ssim", loss_ssim, on_step=False, on_epoch=True)
            # Visualize the reconstruction every 10 epochs for the last batch
            if (self.current_epoch + 1) % 10 == 0:
                self._last_vis = (x.detach(), mask.detach(), recon.detach())
            return loss

        elif self.ssl_mode == "byol":
            v1 = batch["view1"].to(self.device)
            v2 = batch["view2"].to(self.device)
            pred1, pred2, target1, target2 = self.model(v1, v2)
            loss = byol_loss(pred1, pred2, target1, target2)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update target network after each training step
        if self.ssl_mode == "byol":
            self.model.update_moving_average()

    def on_train_epoch_end(self):
        # Save reconstruction visualization every 10 epochs for MIM
        if self.ssl_mode == "mim" and hasattr(self, '_last_vis') and (self.current_epoch + 1) % 10 == 0:
            x, mask, recon = self._last_vis
            visualize_mask_overlay(x, mask, recon, filename=f"epoch_{self.current_epoch + 1}.png",
                                   save_dir=self.check_dir)
            del self._last_vis

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
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--accum_steps', type=int, default=8)
    parser.add_argument('--embed_dim', type=int, default=96)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--roi', type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--experiment', type=int, default=1)
    parser.add_argument('--data', type=str, default="brats")
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument('--ssl_mode', type=str, default="mim", choices=["mim", "byol"])
    args = parser.parse_args()

    # Define hyperparameters
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch
    accum_steps = args.accum_steps
    fold = args.fold
    roi = tuple(args.roi)
    num_heads = [3, 6, 12, 24]
    window_size = (8, 8, 8)
    patch_size = (2, 2, 2)
    embed_dims = [args.embed_dim * (2 ** i) for i in range(5)]
    mask_ratio = args.mask_ratio
    ssl_mode = args.ssl_mode

    print(f"SSL mode: {ssl_mode}")
    print(f"Batch size: {batch_size}, accumulation steps: {accum_steps}, effective batch: {batch_size * accum_steps}")

    # Set relevant paths and load data
    if args.data == "brats":
        in_channels = 4
        out_channels = in_channels  # In SSL I am reconstructing the original image, not predicting classes
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_brats")  # Path is current directory + data_brats
        split_file = os.path.join(data_dir, "data_split.json")  # Json path is data directory + json filename
        ssl_train_loader, train_loader, val_loader, test_loader = ssl_data_loader(dataset_type=args.data,
                                                                                  batch_size=batch_size,
                                                                                  data_dir=data_dir,
                                                                                  split_file=split_file, fold=fold,
                                                                                  roi=roi, ssl_mode=ssl_mode)
    elif args.data == "numorph":
        in_channels = 1
        out_channels = in_channels  # In SSL I am reconstructing the original image, not predicting classes
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_numorph")
        ssl_train_loader, train_loader, val_loader, test_loader = ssl_data_loader(dataset_type=args.data,
                                                                                  batch_size=batch_size,
                                                                                  data_dir=data_dir,
                                                                                  roi=roi, ssl_mode=ssl_mode)
    elif args.data == "selma":
        in_channels = 1
        out_channels = in_channels  # In SSL I am reconstructing the original image, not predicting classes
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_selma")
        ssl_train_loader, train_loader, val_loader, test_loader = ssl_data_loader(dataset_type=args.data,
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

    # Create checkpoint folder
    experiment = args.experiment
    run_name = f"Experiment_{experiment}/data_{args.data}"
    if args.resume_dir is not None:  # Create a new folder for continuation so that things don't get overwritten
        old_version = os.path.basename(args.resume_dir)  # version_Y
        new_version = f"{old_version}_cont"  # Specify that it is a continuation
        version_parent = os.path.dirname(args.resume_dir)  # .../Experiment_X/data_Y
        data_name = os.path.basename(version_parent)  # data_Y
        exp_dir = os.path.dirname(version_parent)  # .../Experiment_X
        exp_name = os.path.basename(exp_dir)  # Experiment_X
        save_dir = os.path.dirname(exp_dir)  # checkpoints_ssl
        logger = CSVLogger(save_dir=save_dir,
                           name=f"{exp_name}/{data_name}",
                           version=new_version)
        check_dir = logger.log_dir
    else:
        logger = CSVLogger(save_dir="checkpoints_ssl", name=run_name)
        check_dir = logger.log_dir

    print(f"Saving at {check_dir}")
    os.makedirs(check_dir, exist_ok=True)
    lit_model.check_dir = check_dir  # Used by on_train_epoch_end for saving visualizations

    # Save best model based on training loss (no validation during SSL pretraining)
    best_check = pl.callbacks.ModelCheckpoint(
        dirpath=check_dir,
        filename="best-model-{epoch:02d}-{train_loss:.4f}",
        monitor="train_loss",
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
        precision="16-mixed",
        accumulate_grad_batches=accum_steps  # Effective batch size = batch_size * acc_steps
    )

    trainer.fit(lit_model, ssl_train_loader, ckpt_path=resume_ckpt)
