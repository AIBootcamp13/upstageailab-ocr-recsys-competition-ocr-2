#!/usr/bin/env python
"""Physically rotate EXIF-tagged images whose polygons already live in the canonical frame.

The script creates a corrected copy of the dataset by:

* Detecting images whose annotations are already expressed in the rotation-corrected
  coordinate system, even though the image still carries a non-trivial EXIF orientation.
* Rotating those images into their canonical orientation and clearing the EXIF flag so
  downstream consumers can treat them as "orientation==1" assets.
* Optionally copying every other image into the target directory to build a complete
  mirrored dataset.

Use this when runtime guards are not sufficient and you need the assets themselves to be
in canonical orientation (for external sharing, reproducibility, or third-party tooling).

Example
-------
```
uv run python scripts/fix_canonical_orientation_images.py \
    data/datasets/images/val \
    data/datasets/jsons/val.json \
    --output-images data/datasets/images_val_canonical \
    --copy-unchanged
```
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from ocr.utils.orientation import (
    EXIF_ORIENTATION_TAG,
    get_exif_orientation,
    normalize_pil_image,
    orientation_requires_rotation,
    polygons_in_canonical_frame,
)


def _collect_polygons(words: dict[str, dict]) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    for payload in words.values():
        points = payload.get("points")
        if points is None:
            continue
        coords = np.asarray(points, dtype=np.float32)
        if coords.size == 0:
            continue
        polygons.append(coords)
    return polygons


def _ensure_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory '{path}' already exists. Use --overwrite to reuse it.")
        if not path.is_dir():
            raise NotADirectoryError(f"Output path '{path}' exists but is not a directory.")
    path.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path, *, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    shutil.copy2(src, dst)


def _save_canonical_image(image: Image.Image, destination: Path, *, dry_run: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return

    exif = image.getexif()
    if exif:
        exif[EXIF_ORIENTATION_TAG] = 1
        image.save(destination, exif=exif.tobytes())
    else:
        image.save(destination)


def _process_dataset(
    images_dir: Path,
    annotations_path: Path,
    output_images: Path,
    *,
    dry_run: bool,
    overwrite: bool,
    copy_unchanged: bool,
    limit: int | None,
) -> dict[str, int]:
    _ensure_output_dir(output_images, overwrite=overwrite)

    with annotations_path.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)

    images_payload: dict[str, dict] = annotations.get("images", {})

    stats = {
        "corrected": 0,
        "copied": 0,
        "skipped_missing": 0,
        "skipped_no_polygons": 0,
        "skipped_not_canonical": 0,
    }

    for index, (filename, payload) in enumerate(images_payload.items(), start=1):
        if limit is not None and index > limit:
            break

        src_path = images_dir / filename
        if not src_path.exists():
            stats["skipped_missing"] += 1
            continue

        words = payload.get("words", {}) or {}
        polygons = _collect_polygons(words)
        if not polygons:
            if copy_unchanged:
                _copy_file(src_path, output_images / filename, dry_run=dry_run)
                stats["copied"] += 1
            else:
                stats["skipped_no_polygons"] += 1
            continue

        with Image.open(src_path) as image:
            orientation = get_exif_orientation(image)
            requires_rotation = orientation_requires_rotation(orientation)
            width, height = image.size

            is_canonical = False
            if requires_rotation:
                is_canonical = polygons_in_canonical_frame(polygons, width, height, orientation)

            if requires_rotation and is_canonical:
                normalized_image, _ = normalize_pil_image(image)
                # normalize_pil_image may return the original reference; clone to avoid side-effects
                if normalized_image is image:
                    normalized_image = image.copy()
                _save_canonical_image(normalized_image, output_images / filename, dry_run=dry_run)
                stats["corrected"] += 1
            else:
                if copy_unchanged:
                    _copy_file(src_path, output_images / filename, dry_run=dry_run)
                    stats["copied"] += 1
                else:
                    stats["skipped_not_canonical"] += 1

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Rotate EXIF-tagged images whose annotations are already canonical, writing corrected copies.")
    )
    parser.add_argument("images", type=Path, help="Directory containing the original images.")
    parser.add_argument(
        "annotations",
        type=Path,
        help="Annotation JSON file (expects the competition 'images' mapping layout).",
    )
    parser.add_argument(
        "--output-images",
        type=Path,
        required=True,
        help="Destination directory for corrected (and optionally copied) images.",
    )
    parser.add_argument(
        "--output-annotations",
        type=Path,
        help="Optional path to write a copy of the annotations JSON alongside corrected images.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report without writing any files.",
    )
    parser.add_argument(
        "--copy-unchanged",
        action="store_true",
        help="Also copy images that do not require correction into the output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow reusing an existing output directory (files may be overwritten).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N entries (useful for smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stats = _process_dataset(
        images_dir=args.images,
        annotations_path=args.annotations,
        output_images=args.output_images,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        copy_unchanged=args.copy_unchanged,
        limit=args.limit,
    )

    if args.output_annotations is not None and not args.dry_run:
        args.output_annotations.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.annotations, args.output_annotations)

    print("Correction summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    if args.dry_run:
        print("No files were written (--dry-run).")


if __name__ == "__main__":
    main()
