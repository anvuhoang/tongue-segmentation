
from pathlib import Path
from collections import Counter
import hashlib

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "dataset"
MASK_DIR = ROOT / "groundtruth" / "mask"

IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    images = {
        p.name: p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }
    masks = {
        p.name: p for p in MASK_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }

    print("=== FILE COUNTS ===")
    print("Images:", len(images))
    print("Masks:", len(masks))

    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))

    print("\n=== PAIRING ===")
    print("Images without masks:", missing_masks)
    print("Masks without images:", missing_images)

    image_hashes = {}
    mask_hashes = {}
    sizes = Counter()
    modes = Counter()
    mask_values = Counter()
    invalid_masks = []

    for name, image_path in sorted(images.items()):
        with Image.open(image_path) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1

        digest = sha256_file(image_path)
        image_hashes.setdefault(digest, []).append(name)

        if name not in masks:
            continue

        with Image.open(masks[name]) as mask:
            arr = np.array(mask)
            mask_size = mask.size

        with Image.open(image_path) as im:
            image_size = im.size

        if image_size != mask_size:
            invalid_masks.append(
                (name, f"size mismatch: image={image_size}, mask={mask_size}")
            )

        values = np.unique(arr)
        mask_values[str(values.tolist())] += 1

        digest = sha256_file(masks[name])
        mask_hashes.setdefault(digest, []).append(name)

        # Kiểm tra mask có phải ảnh nhị phân hoặc grayscale hay không.
        if len(values) > 256:
            invalid_masks.append(
                (name, f"unexpected number of mask values: {len(values)}")
            )

    duplicate_images = [
        names for names in image_hashes.values() if len(names) > 1
    ]
    duplicate_masks = [
        names for names in mask_hashes.values() if len(names) > 1
    ]

    print("\n=== IMAGE PROPERTIES ===")
    print("Image sizes (width, height):", dict(sizes))
    print("Image modes:", dict(modes))

    print("\n=== MASK VALUES ===")
    for values, count in mask_values.items():
        print(f"{values}: {count} files")

    print("\n=== EXACT DUPLICATES ===")
    print("Duplicate image groups:", duplicate_images)
    print("Duplicate mask groups:", duplicate_masks)

    print("\n=== ISSUES ===")
    for issue in invalid_masks:
        print(issue)

    if not missing_masks and not missing_images and not invalid_masks:
        print("\nBasic pairing and size checks passed.")
    else:
        print("\nAudit found issues. Review them before training/evaluation.")


if __name__ == "__main__":
    main()
