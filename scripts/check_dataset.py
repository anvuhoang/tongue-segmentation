from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

IMAGE_DIR = ROOT / "dataset"
GT_IMAGE_DIR = ROOT / "groundtruth" / "images"
MASK_DIR = ROOT / "groundtruth" / "mask"

OUTPUT_DIR = ROOT / "outputs"
VISUALIZATION_PATH = OUTPUT_DIR / "sample_visualization.png"

SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}


# ============================================================
# DATASET UTILITIES
# ============================================================

def get_images(directory: Path) -> dict[str, Path]:
    """Return image files indexed by filename stem."""

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    files = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    result = {}

    for path in files:
        if path.stem in result:
            raise ValueError(
                f"Duplicate filename stem in {directory}: "
                f"{path.stem}"
            )

        result[path.stem] = path

    return result


def inspect_mask(mask_path: Path) -> np.ndarray:
    """Inspect a mask and return its pixel array."""

    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image).copy()
        mode = mask_image.mode
        size = mask_image.size

    print(f"\nMask: {mask_path.name}")
    print(f"  Mode: {mode}")
    print(f"  Size: {size}")
    print(f"  Array shape: {mask.shape}")
    print(f"  Data type: {mask.dtype}")

    if mask.ndim == 2:
        values, counts = np.unique(
            mask,
            return_counts=True,
        )

        print(f"  Unique pixel values: {values.tolist()}")
        print("  Pixel counts:")

        for value, count in zip(values, counts):
            print(f"    {value}: {count:,}")

    elif mask.ndim == 3:
        pixels = mask.reshape(-1, mask.shape[-1])

        colors, counts = np.unique(
            pixels,
            axis=0,
            return_counts=True,
        )

        print(f"  Number of unique colors: {len(colors)}")
        print("  Most frequent colors:")

        order = np.argsort(counts)[::-1][:10]

        for index in order:
            print(
                f"    {colors[index].tolist()}: "
                f"{counts[index]:,} pixels"
            )

    else:
        print("  WARNING: Unexpected mask dimensions.")

    return mask


def compare_images(
    image_path: Path,
    gt_image_path: Path,
) -> bool:
    """Check whether two images have identical pixel data."""

    with Image.open(image_path) as image:
        image_array = np.asarray(image.convert("RGB")).copy()

    with Image.open(gt_image_path) as gt_image:
        gt_array = np.asarray(
            gt_image.convert("RGB")
        ).copy()

    return (
        image_array.shape == gt_array.shape
        and np.array_equal(image_array, gt_array)
    )


# ============================================================
# VISUALIZATION
# ============================================================

def save_visualization(
    image_path: Path,
    gt_image_path: Path | None,
    mask_path: Path,
) -> None:
    """Save input image, reference image, mask, and overlay."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB").copy()

    with Image.open(mask_path) as mask_file:
        mask_image = mask_file.copy()
        mask_array = np.asarray(mask_image).copy()

    if mask_array.ndim != 2:
        raise ValueError(
            "The mask is not a single-channel image. "
            "Inspect its format before creating a binary overlay."
        )

    # The inspected BioHit sample uses a boolean mask.
    # For visualization, True pixels are displayed as white.
    binary_mask = mask_array.astype(bool)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5),
    )

    axes[0].imshow(image)
    axes[0].set_title("Input Image")

    if gt_image_path is not None:
        with Image.open(gt_image_path) as gt_file:
            gt_image = gt_file.convert("RGB").copy()

        axes[1].imshow(gt_image)
        axes[1].set_title("Ground-truth Image")
    else:
        axes[1].text(
            0.5,
            0.5,
            "Not Available",
            ha="center",
            va="center",
        )
        axes[1].set_title("Ground-truth Image")

    axes[2].imshow(
        binary_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Ground-truth Mask")

    axes[3].imshow(image)
    axes[3].imshow(
        binary_mask,
        cmap="Reds",
        alpha=0.4,
        vmin=0,
        vmax=1,
    )
    axes[3].set_title("Image + Mask Overlay")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"BioHit Dataset Sample: {image_path.stem}",
        fontsize=14,
    )

    plt.tight_layout()

    fig.savefig(
        VISUALIZATION_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    print(
        f"\nVisualization saved to:\n"
        f"{VISUALIZATION_PATH}"
    )

    plt.show()
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("BIOHIT DATASET INSPECTION")
    print("=" * 60)

    # Load file lists.
    images = get_images(IMAGE_DIR)
    gt_images = get_images(GT_IMAGE_DIR)
    masks = get_images(MASK_DIR)

    image_names = set(images)
    gt_image_names = set(gt_images)
    mask_names = set(masks)

    # Dataset statistics.
    print(f"\nInput images:       {len(images)}")
    print(f"Ground-truth imgs:  {len(gt_images)}")
    print(f"Ground-truth masks: {len(masks)}")

    # Matching checks.
    images_without_masks = image_names - mask_names
    masks_without_images = mask_names - image_names
    images_without_gt_images = image_names - gt_image_names

    print(
        f"\nImages without masks: "
        f"{len(images_without_masks)}"
    )

    print(
        f"Masks without images: "
        f"{len(masks_without_images)}"
    )

    print(
        "Input images missing from groundtruth/images: "
        f"{len(images_without_gt_images)}"
    )

    if images_without_masks:
        print(
            "  Examples:",
            sorted(images_without_masks)[:10],
        )

    if masks_without_images:
        print(
            "  Examples:",
            sorted(masks_without_images)[:10],
        )

    # Only use matching input images and masks.
    common_names = sorted(image_names & mask_names)

    if not common_names:
        print("\nERROR: No matching image-mask pairs found.")
        return

    print(
        f"\nMatching image-mask pairs: "
        f"{len(common_names)}"
    )

    # Inspect every matched image-mask pair for size compatibility.
    size_mismatches = []

    for name in common_names:
        with Image.open(images[name]) as image_file:
            image_size = image_file.size

        with Image.open(masks[name]) as mask_file:
            mask_size = mask_file.size

        if image_size != mask_size:
            size_mismatches.append(
                (name, image_size, mask_size)
            )

    print(
        f"Image-mask size mismatches: "
        f"{len(size_mismatches)}"
    )

    for name, image_size, mask_size in size_mismatches[:10]:
        print(
            f"  {name}: image={image_size}, "
            f"mask={mask_size}"
        )

    # Inspect one sample.
    sample_name = common_names[0]

    image_path = images[sample_name]
    gt_image_path = gt_images.get(sample_name)
    mask_path = masks[sample_name]

    with Image.open(image_path) as image_file:
        image_size = image_file.size
        image_mode = image_file.mode

    print(f"\nSample: {sample_name}")
    print(f"Input image mode: {image_mode}")
    print(f"Input image size: {image_size}")

    mask_array = inspect_mask(mask_path)

    if gt_image_path is not None:
        same = compare_images(
            image_path,
            gt_image_path,
        )

        print(
            "\nGround-truth image pixel-identical "
            f"to input: {same}"
        )

    # Preliminary mask statistics.
    if mask_array.ndim == 2:
        binary_mask = mask_array.astype(bool)

        total_pixels = binary_mask.size
        foreground_pixels = int(binary_mask.sum())
        background_pixels = total_pixels - foreground_pixels

        print("\nBinary interpretation (preliminary):")
        print(f"  Total pixels:      {total_pixels:,}")
        print(f"  Foreground pixels: {foreground_pixels:,}")
        print(f"  Background pixels: {background_pixels:,}")
        print(
            f"  Foreground ratio: "
            f"{foreground_pixels / total_pixels:.2%}"
        )

        print(
            "\nNote: Confirm visually that the foreground "
            "represents the intended tongue region."
        )

    # Save and display the visualization.
    if size_mismatches:
        print(
            "\nWARNING: Some image-mask pairs have different "
            "sizes. Resolve these before training."
        )

    save_visualization(
        image_path=image_path,
        gt_image_path=gt_image_path,
        mask_path=mask_path,
    )

    print("\nDataset inspection completed.")


if __name__ == "__main__":
    main()
