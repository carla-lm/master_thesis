import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import pandas as pd

matplotlib.use("TkAgg")

# Load file with training data
log_dir = os.path.join(os.getcwd(), "Final_Results", "Finetune_BYOL", "Experiment_21", "data_selma", "version_0")
metrics_file = os.path.join(log_dir, "metrics.csv")
df = pd.read_csv(metrics_file)
check = "supervised"

# Extract the training and validation loss
train_df = df.dropna(subset=["train_loss"])  # Drop rows that have no training loss (validation epochs)
epochs = train_df["epoch"].values
train_loss = train_df["train_loss"].values

if check == "supervised":
    val_loss = df["val_loss"].dropna().values  # Drop rows that have no validation loss (training epochs)
    val_loss_epochs = df["epoch"][df["val_loss"].notna()].values
    # Extract the validation metrics
    val_df = df.dropna(subset=["val_dice_avg"])  # Drop rows that have no validation metrics
    val_epochs = val_df["epoch"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle("Fine-tuning from BYOL 50% data - Selma", fontsize=16, fontweight="bold")

    # Plot training and validation loss vs epochs
    ax1.plot(epochs, train_loss, label="Training Loss", color="red")
    ax1.plot(val_loss_epochs, val_loss, label="Validation Loss", color="blue")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.legend()
    ax1.grid(True)

    # Plot validation dice metrics
    if "val_dice_avg" in val_df.columns:
        dices_avg = val_df["val_dice_avg"].values
        ax2.plot(val_epochs, dices_avg, label="Mean Dice", color="green", linewidth=2)

    if "val_dice_tc" in val_df.columns:
        ax2.plot(val_epochs, val_df["val_dice_tc"].values, label="Dice TC", color="blue", linestyle="--")

    if "val_dice_wt" in val_df.columns:
        ax2.plot(val_epochs, val_df["val_dice_wt"].values, label="Dice WT", color="brown", linestyle="--")

    if "val_dice_et" in val_df.columns:
        ax2.plot(val_epochs, val_df["val_dice_et"].values, label="Dice ET", color="purple", linestyle="--")

    if "val_jaccard_avg" in val_df.columns:
        ax2.plot(val_epochs, val_df["val_jaccard_avg"].values, label="Mean Jaccard", color="orange", linewidth=2)

    ax2.set_title("Validation Dice and Jaccard Scores")
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Score", fontweight="bold")
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.arange(0, 1.1, 0.1))

    plt.tight_layout()
    plt.show()

elif check == "ssl":  # For SSL only the loss is measured
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    fig.suptitle("SSL BYOL pre-training Selma", fontsize=16, fontweight="bold")
    ax1.plot(epochs, train_loss, label="Training Loss", color="red")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 0.00003)
    plt.tight_layout()
    plt.show()
