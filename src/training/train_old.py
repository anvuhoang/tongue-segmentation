from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from src.data.dataset import create_dataloaders
from src.models.unet import UNet


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHECKPOINT_PATH = OUTPUT_DIR / "best_unet.pth"

EPOCHS = 30
LEARNING_RATE = 1e-3
DICE_SMOOTH = 1e-6


def dice_score_from_logits(logits, masks):
    """Calculate mean batch Dice score using a threshold of 0.5."""
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()

    predictions = predictions.flatten(start_dim=1)
    masks = masks.flatten(start_dim=1)

    intersection = (predictions * masks).sum(dim=1)
    denominator = predictions.sum(dim=1) + masks.sum(dim=1)

    scores = (
        2.0 * intersection + DICE_SMOOTH
    ) / (
        denominator + DICE_SMOOTH
    )

    return scores.mean().item()


def dice_loss_from_logits(logits, masks):
    """Differentiable Dice loss for binary segmentation."""
    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.flatten(start_dim=1)
    masks = masks.flatten(start_dim=1)

    intersection = (probabilities * masks).sum(dim=1)
    denominator = probabilities.sum(dim=1) + masks.sum(dim=1)

    dice = (
        2.0 * intersection + DICE_SMOOTH
    ) / (
        denominator + DICE_SMOOTH
    )

    return 1.0 - dice.mean()


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one training or validation epoch."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_dice = 0.0
    total_images = 0

    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)

            bce_loss = criterion(logits, masks)
            dice_loss = dice_loss_from_logits(logits, masks)
            loss = bce_loss + dice_loss

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_dice += dice_score_from_logits(logits.detach(), masks) * batch_size
        total_images += batch_size

    if total_images == 0:
        raise RuntimeError("The data loader is empty.")

    return total_loss / total_images, total_dice / total_images


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # This baseline runs on CPU by default when CUDA is unavailable.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    train_loader, val_loader, _ = create_dataloaders()

    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_dice = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_dice = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )

        val_loss, val_dice = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Dice: {train_dice:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_dice": best_val_dice,
                    "image_size": (288, 384),
                    "seed": 42,
                },
                CHECKPOINT_PATH,
            )

            print(f"  Saved best checkpoint: {CHECKPOINT_PATH}")

    print("\nTraining completed.")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Checkpoint saved at: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()