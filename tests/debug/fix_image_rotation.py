#!/usr/bin/env python
"""Fix canonical rotation issues in test images.

This script corrects image rotation based on EXIF orientation data
to ensure all test images are in the proper upright orientation for OCR.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ocr.utils.orientation import normalize_pil_image


def fix_image_rotation(input_path: Path, output_path: Path) -> bool:
    """Fix rotation for a single image based on EXIF orientation."""
    try:
        with Image.open(input_path) as img:
            # Normalize the image (apply EXIF rotation)
            normalized_img, original_orientation = normalize_pil_image(img)

            # Save the normalized image
            normalized_img.save(output_path, quality=95, optimize=True)

            return original_orientation != 1  # Return True if rotation was applied

    except Exception as e:
        print(f"Error processing {input_path.name}: {e}")
        # Copy the original file if processing fails
        shutil.copy2(input_path, output_path)
        return False


def fix_test_images_rotation(test_dir: Path, output_dir: Path, overwrite: bool = False) -> None:
    """Fix rotation issues for all test images."""
    if overwrite:
        output_dir = test_dir
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(test_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} test images")

    rotated_count = 0
    total_processed = 0

    for image_path in tqdm(image_files, desc="Fixing image rotations"):
        if overwrite:
            output_path = image_path
        else:
            output_path = output_dir / image_path.name

        if output_path.exists() and not overwrite:
            continue  # Skip if output already exists

        was_rotated = fix_image_rotation(image_path, output_path)
        if was_rotated:
            rotated_count += 1
        total_processed += 1

    print("\n=== Rotation Fix Summary ===")
    print(f"Total images processed: {total_processed}")
    print(f"Images that were rotated: {rotated_count}")
    print(f"Images that were already correct: {total_processed - rotated_count}")

    if not overwrite:
        print(f"\nCorrected images saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix canonical rotation issues in test images")
    parser.add_argument("test_dir", type=Path, help="Directory containing test images")
    parser.add_argument("--output_dir", type=Path, help="Output directory for corrected images (default: test_dir_corrected)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite original images instead of creating new directory")

    args = parser.parse_args()

    if not args.test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")

    if args.overwrite:
        output_dir = args.test_dir
        print(f"WARNING: Will overwrite original images in {args.test_dir}")
        confirm = input("Are you sure? (yes/no): ")
        if confirm.lower() != "yes":
            print("Operation cancelled.")
            return
    else:
        output_dir = args.output_dir or args.test_dir.parent / f"{args.test_dir.name}_corrected"

    print(f"Fixing rotation issues in: {args.test_dir}")
    fix_test_images_rotation(args.test_dir, output_dir, args.overwrite)


if __name__ == "__main__":
    main()
