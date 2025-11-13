# Import necessary packages
import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from swinunetr_ssl import SSLSwinUNETR3D
from ssl_data_loading import ssl_data_loader
from utils import reconstruction_loss


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
        loss = reconstruction_loss(x, recon, mask)
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
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=48)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--roi', type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--experiment', type=int, default=1)
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
    window_size = (8, 8, 8)
    patch_size = (2, 2, 2)
    embed_dims = [args.embed_dim * (2 ** i) for i in range(5)]

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

    trainer.fit(lit_model, ssl_train_loader, ckpt_path=resume_ckpt)
