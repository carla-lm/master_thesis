import os
import csv
import argparse
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from training import LitSwinUNETR
from data_loading import ssl_data_loader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import Activations, AsDiscrete
from functools import partial

# matplotlib.use("TkAgg")  # To see figure live


def visualize_prediction(dataset, image, ground_truth, prediction, idx):
    # Make sure everything is CPU numpy arrays
    image = image.detach().cpu().numpy()
    ground_truth = ground_truth.detach().cpu().numpy()
    prediction = prediction.detach().cpu().numpy()

    mid_slice = image.shape[-1] // 2
    if dataset == "brats":
        img_slice = image[3, :, :, mid_slice]  # Use Flair modality as original image
        lbl_slice = ground_truth[:, :, mid_slice]
        pred_slice = prediction[:, :, mid_slice]
    elif dataset == "numorph":
        img_slice = image[:, :, mid_slice]
        lbl_slice = ground_truth[:, :, mid_slice]
        pred_slice = prediction[:, :, mid_slice]
    elif dataset == "selma":
        img_slice = image[:, :, mid_slice]
        lbl_slice = ground_truth[:, :, mid_slice]
        pred_slice = prediction[:, :, mid_slice]

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title(f"Input (sample {idx})")
    axes[1].imshow(lbl_slice, cmap="viridis")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_slice, cmap="viridis")
    axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')  # Trade off precision for performance

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="selma", required=True, choices=["selma", "brats"])
    parser.add_argument("--test_type", type=str, default="metrics", required=True, choices=["visualize", "metrics"])
    parser.add_argument("--roi", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    fold = args.fold
    roi = tuple(args.roi)

    # Load the trained model from checkpoint
    model = LitSwinUNETR.load_from_checkpoint(args.ckpt_path)
    # model = LitSwinUNETR.load_from_checkpoint(r"D:\Master_Thesis\master_thesis\swin_model\Final_Results\Supervised"
    #                                           r"\Experiment_10\data_selma\version_0\best-model-epoch=09-val_dice_avg"
    #                                           r"=0.5157.ckpt")
    model.eval()
    model.to(device)
    model_inferer = partial(sliding_window_inference, roi_size=roi, sw_batch_size=1,
                            predictor=model, overlap=0.5)  # Set the model inferer

    if args.data == "brats":
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_brats")
        split_file = os.path.join(data_dir, "data_split.json")
        _, _, _, test_loader = ssl_data_loader(dataset_type=args.data, batch_size=1, roi=roi, data_dir=data_dir,
                                               split_file=split_file, fold=fold)

    elif args.data == "numorph":
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_numorph")
        _, _, _, test_loader = ssl_data_loader(dataset_type=args.data, batch_size=1, roi=roi, data_dir=data_dir)

    elif args.data == "selma":
        data_dir = os.path.join(os.getcwd(), "TrainingData", "data_selma")
        _, _, _, test_loader = ssl_data_loader(dataset_type=args.data, batch_size=1, roi=roi, data_dir=data_dir)

    else:
        raise ValueError("Unknown dataset")

    if args.test_type == "visualize":
        # Take just a few random samples
        num_samples_total = len(test_loader.dataset)
        num_samples = min(args.num_samples, num_samples_total)
        np.random.seed(42)  # For reproducibility
        selected_indices = np.random.choice(num_samples_total, size=num_samples, replace=False)
        print(f"Visualizing {num_samples} randomly selected test samples...")

        # Run testing
        with torch.no_grad():
            for idx, sample_idx in enumerate(selected_indices, start=1):
                batch = test_loader.dataset[sample_idx]
                img = torch.as_tensor(batch["image"]).to(device)
                label = torch.as_tensor(batch["label"]).to(device)
                prediction = model_inferer(img.unsqueeze(0))
                if args.data == "brats":
                    prediction = torch.sigmoid(prediction)  # convert logits to [0, 1]
                    prediction = (prediction > 0.5).float()  # threshold to binary mask
                    prediction = prediction.squeeze(0)  # remove batch dimension from inference
                    label = label.float()  # convert bool -> float
                    # Convert multichannel label and prediction to a single channel multilabel mask
                    pred_mask = torch.zeros_like(prediction[0])
                    pred_mask[prediction[1] == 1] = 2
                    pred_mask[prediction[0] == 1] = 1
                    pred_mask[prediction[2] == 1] = 4
                    label_mask = torch.zeros_like(label[0])
                    label_mask[label[1] == 1] = 2
                    label_mask[label[0] == 1] = 1
                    label_mask[label[2] == 1] = 4

                elif args.data in ["numorph", "selma"]:
                    prediction = torch.sigmoid(prediction)  # convert logits to [0, 1]
                    prediction = (prediction > 0.5).float()  # threshold to binary mask
                    prediction = prediction.squeeze(0).squeeze(0)  # remove channel and batch dim
                    label = label.squeeze(0)  # remove channel dim
                    img = img.squeeze(0)  # remove channel dim

                # Visualize each sample in batch
                if args.data == "brats":
                    visualize_prediction(args.data, img, label_mask, pred_mask, idx)
                else:
                    visualize_prediction(args.data, img, label, prediction, idx)

    elif args.test_type == "metrics":
        # Test the model on all samples to get the real-life model performance
        post_sigmoid = Activations(sigmoid=True)
        post_pred = AsDiscrete(argmax=False, threshold=0.5)
        include_bg = (args.data == "brats")
        dice_metric = DiceMetric(include_background=include_bg, reduction="mean")
        jaccard_metric = MeanIoU(include_background=include_bg, reduction="mean")
        # For BraTS per-class tracking
        dice_metric_batch = DiceMetric(include_background=True,
                                       reduction="mean_batch") if args.data == "brats" else None

        all_dice = []
        all_jaccard = []
        all_loss = []
        entity_dice = defaultdict(list)  # Track performance per entity in Selma
        entity_jaccard = defaultdict(list)

        print(f"\nEvaluating on {len(test_loader)} test samples...")
        with torch.no_grad():
            for batch in test_loader:
                img = batch["image"].to(device)
                label = batch["label"].to(device)
                logits = model_inferer(img)
                loss = model.loss_func(logits, label)
                all_loss.append(loss.item())
                pred = post_pred(post_sigmoid(logits))

                # Per-sample Dice and Jaccard
                dice_metric.reset()
                jaccard_metric.reset()
                dice_metric(y_pred=pred, y=label)
                jaccard_metric(y_pred=pred, y=label)
                dice_val = dice_metric.aggregate().item()
                jaccard_val = jaccard_metric.aggregate().item()
                all_dice.append(dice_val)
                all_jaccard.append(jaccard_val)

                # Accumulate per-class tracking for BraTS (TC, WT, ET)
                if args.data == "brats":
                    dice_metric_batch(y_pred=pred, y=label)

                # Per-entity tracking for Selma
                if args.data == "selma":
                    filepath = batch["image"].meta["filename_or_obj"][0]
                    for entity_name in ["Cells", "Nuclei", "Vessels"]:
                        if entity_name in filepath:
                            entity_dice[entity_name].append(dice_val)
                            entity_jaccard[entity_name].append(jaccard_val)
                            break

        # Print results
        print(f"\nTest Results ({len(all_dice)} samples):")
        print(f"  Dice:    {np.mean(all_dice):.4f} +/- {np.std(all_dice):.4f}")
        print(f"  Jaccard: {np.mean(all_jaccard):.4f} +/- {np.std(all_jaccard):.4f}")
        print(f"  Loss:    {np.mean(all_loss):.4f}")

        if args.data == "brats":
            brats_dice = dice_metric_batch.aggregate().cpu().numpy()
            print(f"  Dice TC: {brats_dice[0]:.4f}")
            print(f"  Dice WT: {brats_dice[1]:.4f}")
            print(f"  Dice ET: {brats_dice[2]:.4f}")

        if args.data == "selma":
            for entity in ["Cells", "Nuclei", "Vessels"]:
                if entity in entity_dice:
                    scores = entity_dice[entity]
                    jac_scores = entity_jaccard[entity]
                    print(f"  {entity}: Dice = {np.mean(scores):.4f} +/- {np.std(scores):.4f}, "
                          f"Jaccard = {np.mean(jac_scores):.4f} +/- {np.std(jac_scores):.4f} "
                          f"({len(scores)} samples)")

        # Save results to CSV
        os.makedirs("testing", exist_ok=True)
        csv_path = f"testing/test_results_{args.data}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "mean", "std", "count"])
            writer.writerow(["dice", f"{np.mean(all_dice):.4f}", f"{np.std(all_dice):.4f}", len(all_dice)])
            writer.writerow(["jaccard", f"{np.mean(all_jaccard):.4f}", f"{np.std(all_jaccard):.4f}", len(all_jaccard)])
            writer.writerow(["loss", f"{np.mean(all_loss):.4f}", "-", len(all_loss)])

            if args.data == "brats":
                writer.writerow(["dice_TC", f"{brats_dice[0]:.4f}", "-", "-"])
                writer.writerow(["dice_WT", f"{brats_dice[1]:.4f}", "-", "-"])
                writer.writerow(["dice_ET", f"{brats_dice[2]:.4f}", "-", "-"])

            if args.data == "selma":
                for entity in ["Cells", "Nuclei", "Vessels"]:
                    if entity in entity_dice:
                        scores = entity_dice[entity]
                        jac_scores = entity_jaccard[entity]
                        writer.writerow([f"dice_{entity}", f"{np.mean(scores):.4f}", f"{np.std(scores):.4f}", len(scores)])
                        writer.writerow([f"jaccard_{entity}", f"{np.mean(jac_scores):.4f}", f"{np.std(jac_scores):.4f}", len(jac_scores)])

        print(f"\nResults saved to {csv_path}")
