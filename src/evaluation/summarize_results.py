
from pathlib import Path
import csv

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "outputs"

FILES = {
    "U-Net": OUTPUT_DIR / "inspection_metrics.csv",
    "DeepLabV3": OUTPUT_DIR / "deeplabv3_test_results.csv",
}

METRICS = ["dice", "iou", "precision", "recall"]


def read_results(path):
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"File kết quả rỗng: {path}")

    required_columns = ["filename"] + METRICS
    missing = [col for col in required_columns if col not in rows[0]]

    if missing:
        raise ValueError(
            f"File {path.name} thiếu cột {missing}. "
            f"Các cột hiện có: {list(rows[0].keys())}"
        )

    return rows


def main():
    summary = []

    for model_name, path in FILES.items():
        rows = read_results(path)

        result = {
            "model": model_name,
            "num_test_images": len(rows),
        }

        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            result[f"mean_{metric}"] = sum(values) / len(values)

        summary.append(result)

    output_path = OUTPUT_DIR / "model_comparison_summary.csv"

    columns = [
        "model",
        "num_test_images",
        "mean_dice",
        "mean_iou",
        "mean_precision",
        "mean_recall",
    ]

    with output_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)

    print("\n===== MODEL COMPARISON =====")
    print(
        f'{"Model":<15} {"Images":>7} {"Dice":>10} '
        f'{"IoU":>10} {"Precision":>12} {"Recall":>10}'
    )

    for result in summary:
        print(
            f'{result["model"]:<15} '
            f'{result["num_test_images"]:>7} '
            f'{result["mean_dice"]:>10.4f} '
            f'{result["mean_iou"]:>10.4f} '
            f'{result["mean_precision"]:>12.4f} '
            f'{result["mean_recall"]:>10.4f}'
        )

    print(f"\nĐã lưu kết quả tại: {output_path}")


if __name__ == "__main__":
    main()
