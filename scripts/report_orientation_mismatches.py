#!/usr/bin/env python
"""Report annotation samples whose polygons already live in the canonical orientation frame.

This helps identify data points that would otherwise be rotated twice when the EXIF
orientation is applied during preprocessing.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

from ocr.utils.orientation import (
    get_exif_orientation,
    orientation_requires_rotation,
    polygons_in_canonical_frame,
)


def _collect_polygons(words: dict[str, dict]) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    for payload in words.values():
        points = payload.get("points")
        if not isinstance(points, Iterable):
            continue
        coords = np.asarray(points, dtype=np.float32)
        if coords.size == 0:
            continue
        polygons.append(coords)
    return polygons


def _scan_dataset(images_dir: Path, annotations_path: Path, limit: int | None) -> list[tuple[str, int]]:
    with annotations_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    mismatches: list[tuple[str, int]] = []
    for filename, payload in data.get("images", {}).items():
        image_path = images_dir / filename
        if not image_path.exists():
            continue

        words = payload.get("words", {})
        if not words:
            continue

        with Image.open(image_path) as img:
            orientation = get_exif_orientation(img)
            width, height = img.size

        if not orientation_requires_rotation(orientation):
            continue

        polygons = _collect_polygons(words)
        if not polygons:
            continue

        if polygons_in_canonical_frame(polygons, width, height, orientation):
            mismatches.append((filename, orientation))
            if limit is not None and len(mismatches) >= limit:
                break

    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect canonical-orientation annotations despite EXIF rotation.")
    parser.add_argument("images", type=Path, help="Directory containing validation/test images")
    parser.add_argument("annotations", type=Path, help="JSON annotations file (layout like val.json)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of mismatching samples reported (default: unlimited)",
    )
    args = parser.parse_args()

    mismatches = _scan_dataset(args.images, args.annotations, args.limit)
    if not mismatches:
        print("No canonical-frame mismatches detected.")
        return

    print(f"Detected {len(mismatches)} annotation(s) already in canonical frame despite EXIF rotation.\n")
    print("filename,orientation")
    for filename, orientation in mismatches:
        print(f"{filename},{orientation}")


if __name__ == "__main__":
    main()
