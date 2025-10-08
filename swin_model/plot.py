import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
matplotlib.use("TkAgg")

# Load file with training data
log_dir = os.path.join(os.getcwd(), "checkpoints", "Custom_Swin_Lit_Numorph", "version_3")
metrics_file = os.path.join(log_dir, "metrics.csv")
df = pd.read_csv(metrics_file)

# Extract the training and validation loss
train_df = df.dropna(subset=["train_loss"])  # Drop rows that have no training loss (validation epochs)
epochs = train_df["epoch"].values
train_loss = train_df["train_loss"].values
val_loss = df["val_loss"].dropna().values    # Drop rows that have no validation loss (training epochs)
val_loss_epochs = df["epoch"][df["val_loss"].notna()].values

# Plot training and validation loss vs epochs
plt.figure(figsize=(12, 5))
plt.plot(epochs, train_loss, label="Train Loss", color="red")
plt.plot(val_loss_epochs, val_loss, label="Validation Loss", color="blue")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.show()

# Extract the validation metrics
val_df = df.dropna(subset=["val_dice_avg"])  # Drop rows that have no validation metrics
val_epochs = val_df["epoch"].values

# Plot validation dice metrics
plt.figure(figsize=(12, 5))
if "val_dice_avg" in val_df.columns:
    dices_avg = val_df["val_dice_avg"].values
    plt.plot(val_epochs, dices_avg, label="Mean Dice", color="green", linewidth=2)

if "val_dice_tc" in val_df.columns:
    plt.plot(val_epochs, val_df["val_dice_tc"].values, label="Dice TC", color="blue", linestyle="--")

if "val_dice_wt" in val_df.columns:
    plt.plot(val_epochs, val_df["val_dice_wt"].values, label="Dice WT", color="brown", linestyle="--")

if "val_dice_et" in val_df.columns:
    plt.plot(val_epochs, val_df["val_dice_et"].values, label="Dice ET", color="purple", linestyle="--")

if "val_jaccard_avg" in val_df.columns:
    plt.plot(val_epochs, val_df["val_jaccard_avg"].values, label="Mean Jaccard", color="orange", linestyle=":")


plt.title("Validation Dice Scores")
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.show()
