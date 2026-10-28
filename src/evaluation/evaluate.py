
from pathlib import Path
import csv

import torch
from src.data.dataset import create_dataloaders
from src.models.unet import UNet


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHECKPOINT_PATH = PROJECT_DIR / "outputs" / "checkpoints" / "best_unet.pth"
CSV_PATH = OUTPUT_DIR / "test_results.csv"

THRESHOLD = 0.5
SMOOTH = 1e-6


def get_filename(item, index):
    """Get a readable filename from the dataset's third return value."""
    if isinstance(item, (list, tuple)):
        item = item[index] if index < len(item) else item

    if isinstance(item, (str, Path)):
        return Path(item).name

    return f"test_image_{index + 1:03d}"


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    _, _, test_loader = create_dataloaders()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        f"Best validation Dice: "
        f"{checkpoint['best_val_dice']:.4f}"
    )

    results = []
    image_index = 0

    with torch.inference_mode():
        for images, masks, filenames in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            predictions = (
                torch.sigmoid(logits) >= THRESHOLD
            ).float()

            predictions = predictions.flatten(start_dim=1)
            flat_masks = masks.flatten(start_dim=1)

            intersection = (
                predictions * flat_masks
            ).sum(dim=1)

            denominator = (
                predictions.sum(dim=1)
                + flat_masks.sum(dim=1)
            )

            dice_scores = (
                2.0 * intersection + SMOOTH
            ) / (
                denominator + SMOOTH
            )

            batch_size = images.size(0)

            for i in range(batch_size):
                filename = get_filename(
                    filenames, i
                )

                score = dice_scores[i].item()

                results.append({
                    "filename": filename,
                    "dice": score,
                })

                image_index += 1

    if not results:
        raise RuntimeError("Test dataset is empty.")

    results.sort(key=lambda item: item["dice"])

    mean_dice = sum(
        item["dice"] for item in results
    ) / len(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=["filename", "dice"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("TEST RESULTS — SORTED FROM WORST TO BEST")
    print("=" * 60)
    print(f"Number of test images: {len(results)}")
    print(f"Mean Dice: {mean_dice:.4f}")

    print("\n10 WORST IMAGES:")
    for item in results[:10]:
        print(
            f"Dice: {item['dice']:.4f} | "
            f"File: {item['filename']}"
        )

    print("\n5 BEST IMAGES:")
    for item in results[-5:][::-1]:
        print(
            f"Dice: {item['dice']:.4f} | "
            f"File: {item['filename']}"
        )

    print(f"\nFull results saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
