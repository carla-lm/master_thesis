# Import necessary packages
import os
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from functools import partial
from monai.transforms import (AsDiscrete, Activations)
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from swinunetr import SwinUNETR3D
from data_loading import data_loader


class LitSwinUNETR(pl.LightningModule):
    def __init__(self, model, dataset, lr, epochs, roi, sw_batch_size, infer_overlap):
        super().__init__()
        self.model = model
        self.lr = lr
        self.dataset = dataset
        self.epochs = epochs
        if dataset == "brats":
            self.loss_func = DiceLoss(to_onehot_y=False, sigmoid=True)
        elif dataset == "numorph":
            self.loss_func = DiceCELoss(sigmoid=True, squared_pred=True, lambda_dice=0.5, lambda_ce=0.5)

        self.post_sigmoid = Activations(sigmoid=True)
        self.post_pred = AsDiscrete(argmax=False, threshold=0.5)
        self.dice_metric = DiceMetric(include_background=(dataset == "brats"), reduction=MetricReduction.MEAN_BATCH,
                                      get_not_nans=True)
        self.model_inferer = partial(sliding_window_inference, roi_size=roi, sw_batch_size=sw_batch_size,
                                     predictor=self.model, overlap=infer_overlap)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch["image"], batch["label"]
        logits = self(data)
        loss = self.loss_func(logits, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self):
        self.dice_metric.reset()

    def validation_step(self, batch, batch_idx):
        data, target = batch["image"], batch["label"]
        logits = self.model_inferer(data)
        val_labels_list = decollate_batch(target)
        val_outputs_list = decollate_batch(logits)
        val_output_convert = [self.post_pred(self.post_sigmoid(val_tensor)) for val_tensor in val_outputs_list]
        self.dice_metric(y_pred=val_output_convert, y=val_labels_list)
        return logits

    def on_validation_epoch_end(self):
        dice_score, _ = self.dice_metric.aggregate()
        mean_dice = np.mean(dice_score.cpu().numpy())
        if self.dataset == "brats":
            dice_tc, dice_wt, dice_et = dice_score.cpu().numpy()
            self.log("val_dice_tc", dice_tc)
            self.log("val_dice_wt", dice_wt)
            self.log("val_dice_et", dice_et)
            self.log("val_dice_avg", mean_dice)

        elif self.dataset == "numorph":
            self.log("val_dice_avg", mean_dice)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')  # Trade off precision for performance

    # Set parameters that can be passed by the user
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=48)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--roi', type=int, nargs=3, default=[128, 128, 64])
    parser.add_argument('--val_every', type=int, default=10)
    parser.add_argument('--experiment', type=int, default=1)
    parser.add_argument('--data', type=str, default='numorph')
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument("--monai", dest="monai", action="store_true")
    parser.set_defaults(monai=False)
    args = parser.parse_args()

    # Define hyperparameters
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch
    fold = args.fold
    roi = tuple(args.roi)
    feature_size = args.embed_dim
    val_every = args.val_every
    num_heads = [3, 6, 12, 24]
    window_size = (7, 7, 7)
    patch_size = (2, 2, 2)
    embed_dims = [args.embed_dim * (2 ** i) for i in range(5)]
    sw_batch_size = 4
    infer_overlap = 0.5

    # Set relevant paths and load data
    if args.data == "brats":
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_brats")  # Path is current directory + data_brats
        split_file = os.path.join(data_dir, "data_split.json")  # Json path is data directory + json filename
        train_loader, val_loader = data_loader(dataset_type=args.data, batch_size=batch_size, data_dir=data_dir,
                                               split_file=split_file, fold=fold, roi=roi)
    elif args.data == "numorph":
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_numorph")
        train_loader, val_loader = data_loader(dataset_type=args.data, batch_size=batch_size, data_dir=data_dir,
                                               roi=roi)
    elif args.data == "selma":
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_selma")

    else:
        raise ValueError("No correct dataset has been provided")

    # Initialize model
    if args.monai:
        print("Using MONAI's SwinUNETR model")
        model = SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
        )
    else:
        print("Using custom SwinUNETR model")
        model = SwinUNETR3D(
            in_channels=1,
            patch_size=patch_size,
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            num_classes=1
        )

    # Set up lightning modules
    lit_model = LitSwinUNETR(model, args.data, lr, epochs, roi, sw_batch_size, infer_overlap)

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
        logger = CSVLogger(save_dir="checkpoints", name=run_name)
        check_dir = logger.log_dir

    print(f"Saving at {check_dir}")
    os.makedirs(check_dir, exist_ok=True)

    # Save best model
    best_check = pl.callbacks.ModelCheckpoint(
        dirpath=check_dir,
        filename="best-model-{epoch:02d}-{val_dice_avg:.4f}",
        monitor="val_dice_avg",  # Metric used to decide best model
        save_top_k=1,  # Keep only best checkpoint
        mode="max"
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
        check_val_every_n_epoch=val_every,
        precision="16-mixed",  # Automatic mixed precision for faster training
    )

    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=resume_ckpt)
