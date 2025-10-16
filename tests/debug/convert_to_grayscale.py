#!/usr/bin/env python3
"""
Script to convert test images to grayscale optimized for receipt text.
Maintains original filenames and saves to test_gray/ directory.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def enhance_text_contrast(image):
    """
    Enhance image contrast for better text readability on receipts.
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply slight Gaussian blur to reduce noise while preserving text edges
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Apply sharpening filter to enhance text edges
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, kernel)

    return sharpened


def convert_to_grayscale(input_dir, output_dir):
    """
    Convert all images in input_dir to grayscale and save to output_dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    print(f"Found {len(image_files)} images to process")

    # Process each image
    for image_file in tqdm(image_files, desc="Converting images"):
        try:
            # Read image
            image = cv2.imread(str(image_file))

            if image is None:
                print(f"Warning: Could not read {image_file}")
                continue

            # Convert to grayscale with text enhancement
            gray_image = enhance_text_contrast(image)

            # Save with original filename and appropriate compression
            output_file = output_path / image_file.name

            # Use JPEG compression with quality 85 for reasonable file sizes
            if output_file.suffix.lower() in [".jpg", ".jpeg"]:
                cv2.imwrite(str(output_file), gray_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            else:
                # For PNG or other formats, use default settings
                cv2.imwrite(str(output_file), gray_image)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    print(f"Conversion completed! Processed {len(image_files)} images.")
    print(f"Output saved to: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Convert test images to grayscale for receipt text optimization")
    parser.add_argument("--input_dir", type=str, default="data/datasets/images/test", help="Input directory containing test images")
    parser.add_argument("--output_dir", type=str, default="test_gray", help="Output directory for grayscale images")

    args = parser.parse_args()

    # Convert relative paths to absolute if needed
    if not Path(args.input_dir).is_absolute():
        args.input_dir = Path.cwd() / args.input_dir

    if not Path(args.output_dir).is_absolute():
        args.output_dir = Path.cwd() / args.output_dir

    # Check if input directory exists
    if not Path(args.input_dir).exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1

    convert_to_grayscale(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
