#!/usr/bin/env python
"""Check test images for canonical rotation issues.

This script analyzes test images to detect which ones are rotated sideways
(90° or 270° rotations) that should be corrected for proper OCR processing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import cv2
from PIL import Image
from tqdm import tqdm

from ocr.utils.orientation import get_exif_orientation


class RotationAnalysis(NamedTuple):
    """Analysis result for a single image."""

    filename: str
    exif_orientation: int
    detected_rotation: int  # 0, 90, 180, 270 degrees
    confidence: float
    aspect_ratio: float
    width: int
    height: int


def analyze_image_rotation(image_path: Path) -> RotationAnalysis:
    """Analyze an image to detect if it's rotated sideways."""
    # Load image with OpenCV for analysis
    img_cv = cv2.imread(str(image_path))
    if img_cv is None:
        raise ValueError(f"Could not load image: {image_path}")

    height, width = img_cv.shape[:2]
    aspect_ratio = width / height

    # Load with PIL for EXIF data
    with Image.open(image_path) as img_pil:
        exif_orientation = get_exif_orientation(img_pil)

    # Convert to grayscale for analysis
    if len(img_cv.shape) == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cv

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours to analyze text-like structures
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: analyze based on aspect ratio only
        if aspect_ratio < 0.8:  # Tall image (likely rotated 90° CW)
            return RotationAnalysis(
                filename=image_path.name,
                exif_orientation=exif_orientation,
                detected_rotation=90,
                confidence=0.7,
                aspect_ratio=aspect_ratio,
                width=width,
                height=height,
            )
        elif aspect_ratio > 1.2:  # Wide image (likely upright)
            return RotationAnalysis(
                filename=image_path.name,
                exif_orientation=exif_orientation,
                detected_rotation=0,
                confidence=0.8,
                aspect_ratio=aspect_ratio,
                width=width,
                height=height,
            )
        else:  # Square-ish, assume upright
            return RotationAnalysis(
                filename=image_path.name,
                exif_orientation=exif_orientation,
                detected_rotation=0,
                confidence=0.5,
                aspect_ratio=aspect_ratio,
                width=width,
                height=height,
            )

    # Analyze contours to detect text orientation
    horizontal_lines = 0
    vertical_lines = 0
    diagonal_lines = 0

    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Skip very small contours
        if w < 10 or h < 10:
            continue

        # Classify based on aspect ratio of bounding box
        contour_aspect = w / h if h > 0 else 0

        if contour_aspect > 3:  # Very wide - likely horizontal text
            horizontal_lines += 1
        elif contour_aspect < 0.33:  # Very tall - likely vertical text
            vertical_lines += 1
        else:
            diagonal_lines += 1

    # Determine likely rotation based on contour analysis
    total_contours = horizontal_lines + vertical_lines + diagonal_lines

    if total_contours == 0:
        detected_rotation = 0
        confidence = 0.5
    elif horizontal_lines > vertical_lines * 2:
        # Many horizontal lines suggest upright orientation
        detected_rotation = 0
        confidence = min(0.9, horizontal_lines / total_contours)
    elif vertical_lines > horizontal_lines * 2:
        # Many vertical lines suggest 90° rotation
        detected_rotation = 90
        confidence = min(0.9, vertical_lines / total_contours)
    else:
        # Mixed or unclear - check aspect ratio as tiebreaker
        if aspect_ratio < 0.8:
            detected_rotation = 90
            confidence = 0.6
        else:
            detected_rotation = 0
            confidence = 0.6

    # Adjust confidence based on aspect ratio
    if aspect_ratio < 0.7:
        if detected_rotation == 90:
            confidence = min(1.0, confidence + 0.2)
        else:
            confidence = max(0.1, confidence - 0.3)
    elif aspect_ratio > 1.3:
        if detected_rotation == 0:
            confidence = min(1.0, confidence + 0.2)
        else:
            confidence = max(0.1, confidence - 0.3)

    return RotationAnalysis(
        filename=image_path.name,
        exif_orientation=exif_orientation,
        detected_rotation=detected_rotation,
        confidence=confidence,
        aspect_ratio=aspect_ratio,
        width=width,
        height=height,
    )


def analyze_test_images(test_dir: Path, output_json: Path | None = None) -> list[RotationAnalysis]:
    """Analyze all test images for rotation issues."""
    image_files = list(test_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} test images to analyze")

    results = []
    for image_path in tqdm(image_files, desc="Analyzing images"):
        try:
            analysis = analyze_image_rotation(image_path)
            results.append(analysis)
        except Exception as e:
            print(f"Error analyzing {image_path.name}: {e}")
            continue

    # Sort by confidence (most confident issues first)
    results.sort(key=lambda x: x.confidence, reverse=True)

    # Save results to JSON if requested
    if output_json:
        data = {
            "analysis_summary": {
                "total_images": len(results),
                "images_needing_rotation": len([r for r in results if r.detected_rotation != 0 and r.confidence > 0.7]),
                "high_confidence_rotations": len([r for r in results if r.confidence > 0.8]),
            },
            "results": [
                {
                    "filename": r.filename,
                    "exif_orientation": r.exif_orientation,
                    "detected_rotation": r.detected_rotation,
                    "confidence": round(r.confidence, 3),
                    "aspect_ratio": round(r.aspect_ratio, 3),
                    "dimensions": f"{r.width}x{r.height}",
                }
                for r in results
            ],
        }

        with output_json.open("w") as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to {output_json}")

    return results


def print_summary(results: list[RotationAnalysis]) -> None:
    """Print a summary of the rotation analysis."""
    total = len(results)
    needs_rotation = [r for r in results if r.detected_rotation != 0]
    high_confidence = [r for r in results if r.confidence > 0.8]
    very_high_confidence = [r for r in results if r.confidence > 0.9]

    print("\n=== Rotation Analysis Summary ===")
    print(f"Total images analyzed: {total}")
    print(f"Images that may need rotation: {len(needs_rotation)}")
    print(f"High confidence detections (>80%): {len(high_confidence)}")
    print(f"Very high confidence detections (>90%): {len(very_high_confidence)}")

    if needs_rotation:
        print("\n=== Images Needing Rotation (sorted by confidence) ===")
        print("Filename | Detected Rotation | Confidence | Aspect Ratio | Dimensions")
        print("-" * 80)
        for result in needs_rotation[:20]:  # Show top 20
            print(
                f"{result.filename[:24]:<25} | {result.detected_rotation:>3}° | {result.confidence:>4.1%} | {result.aspect_ratio:>5.2f} | {result.width}x{result.height}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check test images for canonical rotation issues")
    parser.add_argument("test_dir", type=Path, help="Directory containing test images")
    parser.add_argument("--output", type=Path, help="Output JSON file for detailed results")
    parser.add_argument("--min-confidence", type=float, default=0.7, help="Minimum confidence threshold for reporting (default: 0.7)")

    args = parser.parse_args()

    if not args.test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")

    print(f"Analyzing test images in: {args.test_dir}")
    results = analyze_test_images(args.test_dir, args.output)

    # Filter results by confidence
    filtered_results = [r for r in results if r.confidence >= args.min_confidence]

    print_summary(filtered_results)

    # Show detailed results for images needing rotation
    rotation_needed = [r for r in filtered_results if r.detected_rotation != 0]
    if rotation_needed:
        print("\n=== Detailed Analysis for Images Needing Rotation ===")
        for result in rotation_needed[:10]:  # Show top 10 most confident
            print(f"\n{result.filename}:")
            print(f"  EXIF Orientation: {result.exif_orientation}")
            print(f"  Detected Rotation: {result.detected_rotation}°")
            print(f"  Confidence: {result.confidence:.1%}")
            print(f"  Aspect Ratio: {result.aspect_ratio:.2f} (width/height)")
            print(f"  Dimensions: {result.width}x{result.height}")
            if result.aspect_ratio < 0.8:
                print("  → Likely rotated 90° clockwise (portrait image detected as landscape)")
            elif result.aspect_ratio > 1.2:
                print("  → Likely upright (landscape image)")
    else:
        print("\nNo images detected as needing rotation above the confidence threshold.")


if __name__ == "__main__":
    main()
