
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.data.dataset import create_dataloaders
from src.models.unet import UNet
from src.models.deeplabv3 import DeepLabV3


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "model_comparisons"
UNET_CHECKPOINT = PROJECT_DIR / "outputs" / "best_unet.pth"
DEEPLAB_CHECKPOINT = PROJECT_DIR / "outputs" / "best_deeplabv3.pth"

SMOOTH = 1e-6
NUM_IMAGES = 6


def load_checkpoint(path, model, device):
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    # Hỗ trợ cả checkpoint dạng dict và state_dict trực tiếp.
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded: {path.name}")
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print("  Epoch:", checkpoint["epoch"])

    return model


def calculate_dice(prediction, target):
    intersection = (prediction * target).sum()
    denominator = prediction.sum() + target.sum()

    return (
        (2.0 * intersection + SMOOTH)
        / (denominator + SMOOTH)
    ).item()


def get_prediction(model, image, device):
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        prediction = (torch.sigmoid(logits) >= 0.5).float()

    return prediction[0, 0].cpu().numpy()


def overlay_mask(image, mask, color):
    """Tô màu vùng mask, giữ ảnh gốc để dễ quan sát."""
    result = image.copy().astype(np.float32)
    mask = mask.astype(bool)

    color_array = np.array(color, dtype=np.float32)
    result[mask] = (
        0.55 * result[mask] + 0.45 * color_array
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    # Dùng đúng cách chia dữ liệu và preprocessing hiện tại.
    _, _, test_loader = create_dataloaders(batch_size=4)

    unet = load_checkpoint(
        UNET_CHECKPOINT,
        UNet(in_channels=3, out_channels=1, base_channels=16).to(device),
        device,
    )

    deeplab = load_checkpoint(
        DEEPLAB_CHECKPOINT,
        DeepLabV3(pretrained_backbone=False).to(device),
        device,
    )

    results = []

    with torch.no_grad():
        for images, masks, filenames in test_loader:
            for i, filename in enumerate(filenames):
                image_tensor = images[i]
                target = masks[i, 0].numpy()

                unet_pred = get_prediction(unet, image_tensor, device)
                deeplab_pred = get_prediction(
                    deeplab, image_tensor, device
                )

                unet_dice = calculate_dice(unet_pred, target)
                deeplab_dice = calculate_dice(deeplab_pred, target)

                # Khôi phục ảnh RGB về thang 0–255 để hiển thị.
                image = (
                    image_tensor.permute(1, 2, 0).numpy() * 255
                ).clip(0, 255).astype(np.uint8)

                results.append({
                    "filename": filename,
                    "image": image,
                    "target": target,
                    "unet_pred": unet_pred,
                    "deeplab_pred": deeplab_pred,
                    "unet_dice": unet_dice,
                    "deeplab_dice": deeplab_dice,
                    "average_dice": (unet_dice + deeplab_dice) / 2,
                })

    if not results:
        raise RuntimeError("Không có ảnh trong tập test.")

    # Ưu tiên xem những ảnh mà cả hai mô hình gặp khó khăn.
    results.sort(key=lambda item: item["average_dice"])
    selected = results[:min(NUM_IMAGES, len(results))]

    print("\n===== LOWEST-DICE TEST IMAGES =====")
    for item in selected:
        print(
            f'{item["filename"]}: '
            f'U-Net Dice={item["unet_dice"]:.4f}, '
            f'DeepLabV3 Dice={item["deeplab_dice"]:.4f}'
        )

    for index, item in enumerate(selected, start=1):
        image = item["image"]
        target = item["target"]
        unet_pred = item["unet_pred"]
        deeplab_pred = item["deeplab_pred"]

        # Màu overlay: xanh lá = GT, đỏ = prediction.
        unet_overlay = overlay_mask(image, unet_pred, (255, 40, 40))
        deeplab_overlay = overlay_mask(
            image, deeplab_pred, (255, 40, 40)
        )

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

        axes[0].imshow(image)
        axes[0].set_title("Original image")

        axes[1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground truth")

        axes[2].imshow(unet_overlay)
        axes[2].contour(
            target, levels=[0.5], colors=["lime"], linewidths=1
        )
        axes[2].set_title(f'U-Net | Dice {item["unet_dice"]:.4f}')

        axes[3].imshow(deeplab_overlay)
        axes[3].contour(
            target, levels=[0.5], colors=["lime"], linewidths=1
        )
        axes[3].set_title(
            f'DeepLabV3 | Dice {item["deeplab_dice"]:.4f}'
        )

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f'Comparison: {item["filename"]} '
            '(green outline = ground truth)',
            fontsize=12,
        )
        fig.tight_layout()

        output_path = OUTPUT_DIR / (
            f'{index:02d}_{Path(item["filename"]).stem}_comparison.png'
        )
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved {len(selected)} comparison images to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
