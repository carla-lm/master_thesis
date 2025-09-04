import matplotlib.pyplot as plt
import matplotlib
import os
import json
import pandas as pd
matplotlib.use("TkAgg")

# Load file with training data
log_dir = os.path.join(os.getcwd(), "checkpoints", "Tester_Run_V1")
metrics_file = os.path.join(log_dir, "metrics.csv")
df = pd.read_csv(metrics_file)
df = df.dropna(subset=["val_dice_avg"])  # Drop rows that don't have validation steps

# Extract the data
epochs = df["epoch"].values
train_loss = df["train_loss"].values
dices_avg = df["val_dice_avg"].values
dices_tc = df["val_dice_tc"].values
dices_wt = df["val_dice_wt"].values
dices_et = df["val_dice_et"].values

# Plot training loss vs epochs
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
plt.xlabel("epoch")
plt.plot(epochs, train_loss, color="red")

# Plot val mean dice
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
plt.xlabel("epoch")
plt.plot(epochs, dices_avg, color="green")
plt.show()

# Plot detailed dice metrics
plt.figure("val_dice", (18, 6))
plt.subplot(1, 3, 1)
plt.title("Val Dice TC")
plt.xlabel("epoch")
plt.plot(epochs, dices_tc, color="blue")

plt.subplot(1, 3, 2)
plt.title("Val Dice WT")
plt.xlabel("epoch")
plt.plot(epochs, dices_wt, color="brown")

plt.subplot(1, 3, 3)
plt.title("Val Dice ET")
plt.xlabel("epoch")
plt.plot(epochs, dices_et, color="purple")
plt.show()
