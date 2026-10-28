
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.dataset import create_dataloaders
from src.models.unet import UNet


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHECKPOINT_PATH = OUTPUT_DIR / "best_unet.pth"

# Số ảnh tốt nhất và kém nhất cần trực quan hóa
NUM_BEST = 3
NUM_WORST = 5


def calculate_metrics(prediction, ground_truth):
    """Calculate binary segmentation metrics."""
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    tp = np.logical_and(prediction, ground_truth).sum()
    fp = np.logical_and(prediction, ~ground_truth).sum()
    fn = np.logical_and(~prediction, ground_truth).sum()

    dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
    iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def save_visualization(image, ground_truth, prediction, filename, metrics):
    """Save input, ground truth, prediction, and boundary overlay."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image)
    axes[0].set_title("Input image")

    axes[1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth")

    axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("U-Net prediction")

    axes[3].imshow(image)
    if ground_truth.any():
        axes[3].contour(
            ground_truth.astype(float),
            levels=[0.5],
            colors=["lime"],
            linewidths=1.5,
        )
    if prediction.any():
        axes[3].contour(
            prediction.astype(float),
            levels=[0.5],
            colors=["red"],
            linewidths=1.5,
        )
    axes[3].set_title("Boundary: green=GT, red=pred")

    fig.suptitle(
        f"{filename} | Dice={metrics['dice']:.4f} | "
        f"IoU={metrics['iou']:.4f}",
        fontsize=13,
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    safe_name = Path(filename).stem
    output_path = OUTPUT_DIR / f"prediction_{safe_name}.png"
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    _, _, test_loader = create_dataloaders()

    model = UNet().to(device)
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print("Running inference on test set...")

    results = []

    with torch.no_grad():
        for images, masks, filenames in test_loader:
            images_device = images.to(device)
            logits = model(images_device)
            predictions = (torch.sigmoid(logits) >= 0.5).cpu().numpy()

            for i, filename in enumerate(filenames):
                image = images[i].permute(1, 2, 0).cpu().numpy()
                ground_truth = masks[i, 0].cpu().numpy() > 0.5
                prediction = predictions[i, 0]

                metrics = calculate_metrics(prediction, ground_truth)

                results.append({
                    "filename": Path(filename).name,
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "image": image,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                })

    results.sort(key=lambda item: item["dice"])

    # Save numerical results
    csv_path = OUTPUT_DIR / "inspection_metrics.csv"
    pd.DataFrame([
        {key: item[key] for key in (
            "filename", "dice", "iou", "precision", "recall"
        )}
        for item in results
    ]).to_csv(csv_path, index=False)

    print("\n=== TEST SET METRICS ===")
    for metric in ("dice", "iou", "precision", "recall"):
        values = [item[metric] for item in results]
        print(f"Mean {metric.capitalize()}: {np.mean(values):.4f}")

    print(f"Number of test images: {len(results)}")
    print(f"Metrics saved to: {csv_path}")

    # Visualize worst-performing images
    worst = results[:NUM_WORST]
    best = results[-NUM_BEST:]

    print("\n=== WORST PREDICTIONS ===")
    for item in worst:
        print(
            f"{item['filename']}: Dice={item['dice']:.4f}, "
            f"IoU={item['iou']:.4f}"
        )
        path = save_visualization(
            item["image"],
            item["ground_truth"],
            item["prediction"],
            item["filename"],
            item,
        )
        print(f"Saved: {path}")

    print("\n=== BEST PREDICTIONS ===")
    for item in reversed(best):
        print(
            f"{item['filename']}: Dice={item['dice']:.4f}, "
            f"IoU={item['iou']:.4f}"
        )
        path = save_visualization(
            item["image"],
            item["ground_truth"],
            item["prediction"],
            item["filename"],
            item,
        )
        print(f"Saved: {path}")

    print("\nInspection completed.")


if __name__ == "__main__":
    main()
