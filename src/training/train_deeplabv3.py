
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.models.deeplabv3 import DeepLabV3


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHECKPOINT_PATH = OUTPUT_DIR / "best_deeplabv3.pth"

EPOCHS = 30
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
DICE_SMOOTH = 1e-6


def dice_score_from_logits(logits, masks):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()

    intersection = (predictions * masks).sum(dim=(1, 2, 3))
    denominator = predictions.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

    dice = (2 * intersection + DICE_SMOOTH) / (
        denominator + DICE_SMOOTH
    )

    return dice.mean()


def dice_loss_from_logits(logits, masks):
    probabilities = torch.sigmoid(logits)

    intersection = (probabilities * masks).sum(dim=(1, 2, 3))
    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + masks.sum(dim=(1, 2, 3))
    )

    dice = (2 * intersection + DICE_SMOOTH) / (
        denominator + DICE_SMOOTH
    )

    return 1 - dice.mean()


def run_epoch(model, loader, optimizer, bce_loss, device, training):
    model.train() if training else model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_images = 0

    description = "Train" if training else "Validation"

    with torch.set_grad_enabled(training):
        for images, masks, _ in tqdm(loader, desc=description):
            images = images.to(device)
            masks = masks.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            loss = (
                bce_loss(logits, masks)
                + dice_loss_from_logits(logits, masks)
            )

            if training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            batch_dice = dice_score_from_logits(logits.detach(), masks)

            total_loss += loss.item() * batch_size
            total_dice += batch_dice.item() * batch_size
            total_images += batch_size

    return (
        total_loss / max(total_images, 1),
        total_dice / max(total_images, 1),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA không khả dụng. Hãy kiểm tra PyTorch và GPU trước khi train."
        )

    device = torch.device("cuda")
    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Batch size:", BATCH_SIZE)

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE
    )

    model = DeepLabV3(pretrained_backbone=False).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    bce_loss = nn.BCEWithLogitsLoss()

    best_val_dice = -1.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_dice = run_epoch(
            model, train_loader, optimizer, bce_loss, device, True
        )

        val_loss, val_dice = run_epoch(
            model, val_loader, optimizer, bce_loss, device, False
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Dice: {train_dice:.4f}"
        )
        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_dice": best_val_dice,
                },
                CHECKPOINT_PATH,
            )

            print(f"Saved best checkpoint: {CHECKPOINT_PATH}")

        # Giảm learning rate nếu validation không cải thiện.
        # Scheduler được tạo bên dưới ở phiên bản đơn giản hơn:
        # hiện tại giữ learning rate cố định để dễ so sánh baseline.

    print("\nTraining completed.")
    print("Best epoch:", best_epoch)
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print("Checkpoint:", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
