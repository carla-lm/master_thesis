import matplotlib.pyplot as plt
import matplotlib
import os
import json
matplotlib.use("TkAgg")

# Load file with training data
data_dir = os.path.join(os.getcwd(), "checkpoints", "20250821_190305")
history_file = os.path.join(data_dir, "training_history.json")
with open(history_file, "r") as f:
    history = json.load(f)

# Extract the data
trains_epoch = history["trains_epoch"]
loss_epochs = history["loss_epochs"]
dices_avg = history["dices_avg"]
dices_tc = history["dices_tc"]
dices_wt = history["dices_wt"]
dices_et = history["dices_et"]


# Plot the metrics vs epoch
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
plt.xlabel("epoch")
plt.plot(trains_epoch, loss_epochs, color="red")
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
plt.xlabel("epoch")
plt.plot(trains_epoch, dices_avg, color="green")
plt.show()
plt.figure("train", (18, 6))
plt.subplot(1, 3, 1)
plt.title("Val Mean Dice TC")
plt.xlabel("epoch")
plt.plot(trains_epoch, dices_tc, color="blue")
plt.subplot(1, 3, 2)
plt.title("Val Mean Dice WT")
plt.xlabel("epoch")
plt.plot(trains_epoch, dices_wt, color="brown")
plt.subplot(1, 3, 3)
plt.title("Val Mean Dice ET")
plt.xlabel("epoch")
plt.plot(trains_epoch, dices_et, color="purple")
plt.show()
