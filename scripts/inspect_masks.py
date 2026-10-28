
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.dataset import create_dataloaders
from src.models.unet import UNet

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TARGET = "278.bmp"


def report(name, arr):
    ys, xs = np.where(arr > 0)
    print(f"\n{name}")
    print("Shape:", arr.shape)
    print("Foreground ratio:", round(float((arr > 0).mean() * 100), 2), "%")

    if len(xs):
        print("Bounding box:", (xs.min(), ys.min(), xs.max(), ys.max()))
    else:
        print("No foreground pixels")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = create_dataloaders()

    checkpoint = torch.load(
        OUT / "best_unet.pth",
        map_location=device,
        weights_only=True,
    )

    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for images, masks, filenames in test_loader:
        for i, filename in enumerate(filenames):
            if Path(filename).name != TARGET:
                continue

            image = images[i].permute(1, 2, 0).cpu().numpy()
            gt = (masks[i, 0].cpu().numpy() > 0.5).astype(np.uint8)

            with torch.no_grad():
                logits = model(images[i:i + 1].to(device))
                pred = (
                    torch.sigmoid(logits)[0, 0].cpu().numpy() >= 0.5
                ).astype(np.uint8)

            intersection = np.logical_and(gt, pred).sum()
            dice = (2 * intersection + 1e-6) / (
                gt.sum() + pred.sum() + 1e-6
            )

            report("Ground truth", gt)
            report("Prediction", pred)
            print("\nDice:", round(float(dice), 4))

            Image.fromarray(gt * 255).save(OUT / "278_groundtruth.png")
            Image.fromarray(pred * 255).save(OUT / "278_prediction.png")
            Image.fromarray((image * 255).clip(0, 255).astype(np.uint8)).save(
                OUT / "278_input.png"
            )
            print("\nSaved three images in outputs/")
            return

    print(f"{TARGET} was not found in the test set.")


if __name__ == "__main__":
    main()
