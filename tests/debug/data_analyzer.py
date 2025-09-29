# data_analyzer.py
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from ocr.datasets.base import OCRDataset

# The EXIF orientation tag constant
EXIF_ORIENTATION_TAG = 274


class IdentityTransform:
    """Minimal transform stub for dataset instantiation."""

    def __call__(self, *, image, polygons):  # type: ignore[override]
        return {
            "image": image,
            "polygons": polygons or [],
            "inverse_matrix": np.eye(3, dtype=np.float32),
        }


def analyze_polygon_loss(dataset: OCRDataset, limit: int | None = None) -> None:
    """
    Analyzes the loss of polygons after dataset processing.

    Args:
        dataset: An instantiated OCR dataset object that has .anns and can be indexed.
        limit: Optional cap on number of samples to process.
    """
    print("--- Analyzing Polygon Loss ---")
    raw_total = 0
    transformed_total = 0
    filtered_images_count = 0
    loss_counts = Counter()
    processed_images = 0

    for idx, (filename, raw_polys) in enumerate(dataset.anns.items()):
        if limit is not None and idx >= limit:
            break
        raw_count = 0 if raw_polys is None else len(raw_polys)
        raw_total += raw_count

        sample = dataset[idx]
        transformed_polys = sample["polygons"]
        transformed_count = len(transformed_polys)
        transformed_total += transformed_count
        processed_images += 1

        if transformed_count < raw_count:
            filtered_images_count += 1
            loss_counts[(raw_count, transformed_count)] += 1

    total_loss = raw_total - transformed_total
    loss_percentage = (total_loss / raw_total * 100) if raw_total > 0 else 0

    expected_total = min(len(dataset), limit) if limit is not None else len(dataset)
    print(f"Total images processed: {processed_images} (expected {expected_total})")
    print(f"Images with at least one filtered polygon: {filtered_images_count}")
    print(f"Original total polygons: {raw_total}")
    print(f"Polygons after processing: {transformed_total}")
    print(f"Total polygons lost: {total_loss}")
    print(f"Loss percentage: {loss_percentage:.4f}%")
    if loss_counts:
        print("Most common loss patterns (raw_count -> new_count):")
        for item, count in loss_counts.most_common(5):
            print(f"  {item}: {count} time(s)")
    print("-" * 28 + "\n")


def count_exif_orientations(image_dir: Path) -> None:
    """
    Counts the occurrences of each EXIF orientation tag in a directory of images.

    Args:
        image_dir: Path object pointing to the directory of images.
    """
    print("--- Counting EXIF Orientations ---")
    if not image_dir.is_dir():
        print(f"Error: Directory not found at {image_dir}")
        return

    orientation_counts = Counter()
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))

    for path in image_files:
        try:
            with Image.open(path) as img:
                exif = img.getexif()
                orientation = exif.get(EXIF_ORIENTATION_TAG, 1)  # Default to 1 (normal)
                orientation_counts[orientation] += 1
        except Exception as e:
            print(f"Could not process {path}: {e}")

    print(f"Found {sum(orientation_counts.values())} images.")
    print("Orientation counts:")
    for orientation, count in sorted(orientation_counts.items()):
        print(f"  Orientation {orientation}: {count} image(s)")
    print("-" * 32 + "\n")


def ensure_dataset(image_dir: Path, annotation_path: Path) -> OCRDataset:
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    return OCRDataset(
        image_path=image_dir,
        annotation_path=annotation_path,
        transform=IdentityTransform(),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset diagnostics for EXIF orientations and polygon retention.")
    parser.add_argument(
        "--mode",
        choices={"orientation", "polygons", "both"},
        default="orientation",
        help="Which diagnostics to run.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/datasets/images/train"),
        help="Directory containing input images (default: data/datasets/images/train).",
    )
    parser.add_argument(
        "--annotation-path",
        type=Path,
        default=Path("data/datasets/jsons/train.json"),
        help="Annotation JSON for polygon audit (default: data/datasets/jsons/train.json).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of images to process for polygon audit.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    if args.mode in {"orientation", "both"}:
        count_exif_orientations(args.image_dir)

    if args.mode in {"polygons", "both"}:
        try:
            dataset = ensure_dataset(args.image_dir, args.annotation_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to instantiate dataset for polygon audit: {exc}")
        else:
            analyze_polygon_loss(dataset, limit=args.limit)
