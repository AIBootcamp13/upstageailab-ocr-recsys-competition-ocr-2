# data_analyzer.py
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ocr.datasets.base import OCRDataset
from ocr.utils.orientation import normalize_pil_image, remap_polygons

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


def _ensure_polygon_list(polygons: Any) -> list[np.ndarray]:
    if polygons is None:
        return []
    if isinstance(polygons, np.ndarray):
        return [np.asarray(polygons)]
    if isinstance(polygons, list | tuple):
        return [np.asarray(poly) for poly in polygons if poly is not None]
    return [np.asarray(polygons)]


def analyze_polygon_loss(dataset: OCRDataset, limit: int | None = None) -> None:
    """Analyzes how many polygons survive dataset preprocessing."""

    print("--- Analyzing Polygon Loss ---")
    raw_total = 0
    transformed_total = 0
    filtered_images_count = 0
    loss_counts = Counter()
    processed_images = 0

    for idx, (filename, raw_polys) in enumerate(dataset.anns.items()):
        if limit is not None and idx >= limit:
            break

        raw_polygons = _ensure_polygon_list(raw_polys)
        raw_count = len(raw_polygons)
        raw_total += raw_count

        sample = dataset[idx]
        transformed_polygons = _ensure_polygon_list(sample.get("polygons"))
        transformed_count = len(transformed_polygons)
        transformed_total += transformed_count
        processed_images += 1

        if transformed_count < raw_count:
            filtered_images_count += 1
            loss_counts[(raw_count, transformed_count)] += 1

    total_loss = raw_total - transformed_total
    loss_percentage = (total_loss / raw_total * 100) if raw_total > 0 else 0.0

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
    """Counts the occurrences of each EXIF orientation tag in a directory of images."""

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
        except Exception as exc:  # noqa: BLE001
            print(f"Could not process {path}: {exc}")

    print(f"Found {sum(orientation_counts.values())} images.")
    print("Orientation counts:")
    for orientation, count in sorted(orientation_counts.items()):
        print(f"  Orientation {orientation}: {count} image(s)")
    print("-" * 32 + "\n")


def analyze_orientation_alignment(
    dataset: OCRDataset,
    limit: int | None = None,
    atol: float = 1e-3,
) -> None:
    """Verifies that dataset polygon outputs match EXIF-informed expectations."""

    print("--- Auditing Orientation Alignment ---")
    processed = 0
    checked = 0
    mismatches: list[tuple[str, int, int]] = []
    orientation_counts: Counter[int] = Counter()
    skipped_images: list[str] = []

    for idx, (filename, raw_polys) in enumerate(dataset.anns.items()):
        if limit is not None and idx >= limit:
            break

        image_path = dataset.image_path / filename
        try:
            with Image.open(image_path) as pil_image:
                raw_width, raw_height = pil_image.size
                _, orientation = normalize_pil_image(pil_image)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to analyze {filename}: {exc}")
            skipped_images.append(filename)
            continue

        orientation_counts[orientation] += 1
        processed += 1

        sample = dataset[idx]
        transformed_polygons = _ensure_polygon_list(sample.get("polygons"))
        raw_polygons = _ensure_polygon_list(raw_polys)

        if not raw_polygons and not transformed_polygons:
            continue

        checked += 1

        expected_polygons = remap_polygons(raw_polygons, raw_width, raw_height, orientation) if raw_polygons else []

        if len(expected_polygons) != len(transformed_polygons):
            mismatches.append((filename, len(expected_polygons), len(transformed_polygons)))
            continue

        mismatch_detected = False
        for expected, actual in zip(expected_polygons, transformed_polygons, strict=True):
            expected_arr = np.asarray(expected, dtype=np.float32)
            actual_arr = np.asarray(actual, dtype=np.float32)
            if not np.allclose(expected_arr, actual_arr, atol=atol):
                mismatch_detected = True
                break

        if mismatch_detected:
            mismatches.append((filename, len(expected_polygons), len(transformed_polygons)))

    print(f"Images processed: {processed}")
    if skipped_images:
        print(f"Skipped {len(skipped_images)} image(s) due to load errors.")
    print("Orientation distribution encountered during audit:")
    for orientation, count in sorted(orientation_counts.items()):
        print(f"  Orientation {orientation}: {count} image(s)")
    print(f"Polygon-bearing samples checked: {checked}")
    print(f"Alignment mismatches detected: {len(mismatches)}")
    if mismatches:
        print("  Examples (expected -> transformed counts):")
        for filename, expected_count, transformed_count in mismatches[:5]:
            print(f"    {filename}: {expected_count} -> {transformed_count}")
    print("-" * 35 + "\n")


def ensure_dataset(image_dir: Path, annotation_path: Path) -> OCRDataset:
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    return OCRDataset(
        image_path=image_dir,
        annotation_path=annotation_path,
        transform=IdentityTransform(),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset diagnostics for EXIF orientations, polygon retention, and alignment.",
    )
    parser.add_argument(
        "--mode",
        choices={"orientation", "polygons", "alignment", "both", "all"},
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
    parser.add_argument(
        "--alignment-atol",
        type=float,
        default=1e-3,
        help="Tolerance used when comparing expected vs transformed polygons for alignment audit.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    if args.mode in {"orientation", "both", "all"}:
        count_exif_orientations(args.image_dir)

    dataset: OCRDataset | None = None
    needs_dataset = args.mode in {"polygons", "alignment", "both", "all"}
    if needs_dataset:
        try:
            dataset = ensure_dataset(args.image_dir, args.annotation_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to instantiate dataset for dataset-based diagnostics: {exc}")

    if dataset is not None:
        if args.mode in {"polygons", "both", "all"}:
            analyze_polygon_loss(dataset, limit=args.limit)
        if args.mode in {"alignment", "all"}:
            analyze_orientation_alignment(dataset, limit=args.limit, atol=args.alignment_atol)
    elif needs_dataset:
        print("Dataset-dependent diagnostics were skipped due to earlier errors.")
