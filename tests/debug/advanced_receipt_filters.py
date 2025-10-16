#!/usr/bin/env python3
"""
Advanced receipt text enhancement filters.
Builds on the existing convert_to_grayscale.py with additional preprocessing techniques.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def enhance_text_contrast_advanced(image, method="receipt_optimized"):
    """
    Advanced text enhancement with multiple filter options optimized for receipts.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if method == "receipt_optimized":
        # Multi-stage enhancement for receipts
        enhanced = apply_receipt_filters(gray)
    elif method == "high_contrast":
        enhanced = apply_high_contrast_filters(gray)
    elif method == "shadow_removal":
        enhanced = apply_shadow_removal_filters(gray)
    elif method == "noise_reduction":
        enhanced = apply_noise_reduction_filters(gray)
    else:
        # Default CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

    return enhanced


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


def apply_high_contrast_filters(gray):
    """High contrast filters for faded or low-contrast text."""
    # Adaptive thresholding for high contrast
    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Morphological opening to remove noise
    kernel = np.ones((2, 2), np.uint8)
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)

    return enhanced


def apply_shadow_removal_filters(gray):
    """Filters to remove shadows and improve text visibility."""
    # Normalize lighting using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.medianBlur(background, 5)

    # Divide gray by background to normalize
    normalized = cv2.divide(gray, background, scale=255)

    # CLAHE on normalized image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    return enhanced


def apply_noise_reduction_filters(gray):
    """Aggressive noise reduction while preserving text."""
    # Non-local means denoising
    enhanced = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(enhanced)

    # Median blur to further reduce noise
    enhanced = cv2.medianBlur(enhanced, 3)

    return enhanced


def convert_with_advanced_filters(input_dir, output_dir, method="receipt_optimized"):
    """Convert images using advanced filter methods."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    print(f"Processing {len(image_files)} images with {method} filters")

    for image_file in tqdm(image_files, desc=f"Applying {method} filters"):
        try:
            image = cv2.imread(str(image_file))
            if image is None:
                continue

            enhanced = enhance_text_contrast_advanced(image, method)

            output_file = output_path / f"{image_file.stem}_{method}{image_file.suffix}"
            cv2.imwrite(str(output_file), enhanced)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    print(f"Advanced filtering completed! Output in: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Advanced receipt text enhancement")
    parser.add_argument("--input_dir", default="test_corrected", help="Input directory with corrected images")
    parser.add_argument("--output_dir", default="test_enhanced", help="Output directory for enhanced images")
    parser.add_argument(
        "--method",
        default="receipt_optimized",
        choices=["receipt_optimized", "high_contrast", "shadow_removal", "noise_reduction"],
        help="Enhancement method to apply",
    )

    args = parser.parse_args()

    if not Path(args.input_dir).is_absolute():
        args.input_dir = Path.cwd() / args.input_dir
    if not Path(args.output_dir).is_absolute():
        args.output_dir = Path.cwd() / args.output_dir

    if not Path(args.input_dir).exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1

    convert_with_advanced_filters(args.input_dir, args.output_dir, args.method)
    return 0


if __name__ == "__main__":
    exit(main())
