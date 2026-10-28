
from pathlib import Path
import csv

import torch

from src.data.dataset import create_dataloaders
from src.models.deeplabv3 import DeepLabV3


PROJECT_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_DIR / "outputs" / "checkpoints" / "best_deeplabv3.pth"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "deeplabv3_test_results.csv"

SMOOTH = 1e-6


def calculate_metrics(logits, masks):
    # Chuyển logits thành mask nhị phân.
    predictions = (torch.sigmoid(logits) >= 0.5).float()

    # Tính riêng từng ảnh, không gộp tất cả ảnh thành một mask.
    dims = (1, 2, 3)

    tp = (predictions * masks).sum(dim=dims)
    fp = (predictions * (1 - masks)).sum(dim=dims)
    fn = ((1 - predictions) * masks).sum(dim=dims)

    dice = (2 * tp + SMOOTH) / (2 * tp + fp + fn + SMOOTH)
    iou = (tp + SMOOTH) / (tp + fp + fn + SMOOTH)
    precision = (tp + SMOOTH) / (tp + fp + SMOOTH)
    recall = (tp + SMOOTH) / (tp + fn + SMOOTH)

    return dice, iou, precision, recall


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {CHECKPOINT_PATH}\n"
            "Hãy kiểm tra train_deeplab.py đã chạy và lưu checkpoint chưa."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    # Giữ nguyên cách chia train/validation/test của dataset.py.
    _, _, test_loader = create_dataloaders(batch_size=2)

    model = DeepLabV3(pretrained_backbone=False).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print(
        "Best validation Dice:",
        f'{checkpoint["best_val_dice"]:.4f}',
    )

    rows = []

    with torch.no_grad():
        for images, masks, filenames in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            dice, iou, precision, recall = calculate_metrics(
                logits, masks
            )

            for i, filename in enumerate(filenames):
                rows.append({
                    "filename": filename,
                    "dice": dice[i].item(),
                    "iou": iou[i].item(),
                    "precision": precision[i].item(),
                    "recall": recall[i].item(),
                })

    if not rows:
        raise RuntimeError("Tập test không có ảnh nào.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "filename", "dice", "iou", "precision", "recall"
    ]

    with OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== DEEPLABV3 TEST RESULTS =====")
    print("Number of test images:", len(rows))

    for metric in ["dice", "iou", "precision", "recall"]:
        average = sum(row[metric] for row in rows) / len(rows)
        print(f"Mean {metric.capitalize()}: {average:.4f}")

    print("\nPer-image results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
