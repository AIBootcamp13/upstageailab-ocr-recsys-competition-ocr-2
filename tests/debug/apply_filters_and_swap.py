#!/usr/bin/env python3
"""
Apply receipt filters to train and validation images, then swap directories.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def apply_receipt_filters(gray):
    """Optimized filter chain for receipt text detection."""
    # 1. CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Bilateral filter to reduce noise while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # 3. Morphological operations to clean up text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    # 4. Sharpening to enhance text edges
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    return enhanced


def process_directory(input_dir, output_dir):
    """Process all images in a directory with receipt filters."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Warning: Input directory {input_path} does not exist")
        return 0

    output_path.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    print(f"Processing {len(image_files)} images from {input_path}...")

    processed_count = 0
    for image_file in tqdm(image_files, desc=f"Processing {input_path.name}"):
        try:
            # Read image
            image = cv2.imread(str(image_file))
            if image is None:
                continue

            # Convert to grayscale for processing
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Apply receipt filters
            filtered = apply_receipt_filters(gray)

            # Save with original filename
            output_file = output_path / image_file.name
            cv2.imwrite(str(output_file), filtered)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    return processed_count


def swap_directories(base_dir):
    """Swap images/ and filtered/ directories, adding swapped suffix."""
    base_path = Path(base_dir)
    images_dir = base_path / "images"
    filtered_dir = base_path / "filtered"

    if not images_dir.exists():
        print(f"Error: {images_dir} does not exist")
        return False

    if not filtered_dir.exists():
        print(f"Error: {filtered_dir} does not exist")
        return False

    # Create swapped directory name
    swapped_dir = base_path / "images_swapped"

    # Rename images/ to images_swapped/
    print(f"Renaming {images_dir} to {swapped_dir}")
    images_dir.rename(swapped_dir)

    # Rename filtered/ to images/
    print(f"Renaming {filtered_dir} to {images_dir}")
    filtered_dir.rename(images_dir)

    return True


def main():
    parser = argparse.ArgumentParser(description="Apply receipt filters to train/val images and swap directories")
    parser.add_argument(
        "--base_dir",
        default="/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets",
        help="Base directory containing images/ and images_val_canonical/",
    )

    args = parser.parse_args()

    base_path = Path(args.base_dir)

    # Define directories
    train_dir = base_path / "images" / "train"
    val_dir = base_path / "images_val_canonical"
    filtered_dir = base_path / "filtered"

    print("Starting receipt filter processing...")

    # Process training images
    train_count = process_directory(train_dir, filtered_dir / "train")

    # Process validation images
    val_count = process_directory(val_dir, filtered_dir / "val")

    print(f"\nProcessed {train_count} training images and {val_count} validation images")

    # Swap directories
    print("\nSwapping directories...")
    if swap_directories(base_path):
        print("Directory swap completed successfully!")
        print("- Original images/ renamed to images_swapped/")
        print("- Filtered images/ is now the active images/ directory")
    else:
        print("Directory swap failed!")

    return 0


if __name__ == "__main__":
    exit(main())
