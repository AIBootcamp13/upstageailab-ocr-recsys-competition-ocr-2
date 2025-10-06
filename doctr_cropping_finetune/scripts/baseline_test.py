#!/usr/bin/env python3
"""
Baseline testing script for docTR cropping fine-tuning.

This script tests the current default document detection parameters on a set of images
and generates visual overlays showing the detected document boundaries.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ocr.datasets.preprocessing.config import DocumentPreprocessorConfig
from ocr.datasets.preprocessing.detector import DocumentDetector


def setup_logging(log_file: Path) -> logging.Logger:
    """Setup logging to both file and console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def draw_document_overlay(image_array: np.ndarray, corners: np.ndarray | None) -> Image.Image:
    """Draw the green document overlay on an image."""
    base_image = Image.fromarray(image_array)
    if corners is None:
        return base_image

    draw = ImageDraw.Draw(base_image, "RGBA")

    # Convert corners to points
    if isinstance(corners, np.ndarray):
        corners_array = corners
    elif isinstance(corners, list):
        corners_array = np.asarray(corners)
    else:
        return base_image

    if corners_array.size < 8:
        return base_image

    points = [(float(x), float(y)) for x, y in corners_array]

    # Draw translucent green polygon
    draw.polygon(points, outline=(0, 255, 0, 255), fill=(0, 255, 0, 40))

    # Draw center dot
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    draw.ellipse(
        [
            (center_x - 3, center_y - 3),
            (center_x + 3, center_y + 3),
        ],
        fill=(0, 128, 0, 255),
    )

    return base_image


def test_document_detection(image_path: Path, detector: DocumentDetector, output_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    """Test document detection on a single image."""
    logger.info(f"Testing image: {image_path.name}")

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return {"image": str(image_path), "error": "Failed to load"}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect document
    corners, method = detector.detect(image_rgb)

    # Generate overlay
    overlay_image = draw_document_overlay(image_rgb, corners)

    # Save overlay
    output_path = output_dir / f"{image_path.stem}_overlay.png"
    overlay_image.save(output_path)

    # Calculate metrics if corners found
    metrics = {}
    if corners is not None:
        # Calculate area ratio
        height, width = image_rgb.shape[:2]
        total_area = height * width

        # Calculate polygon area
        corners_reshaped = corners.reshape(-1, 2)
        polygon_area = cv2.contourArea(corners_reshaped.astype(np.float32))
        area_ratio = polygon_area / total_area if total_area > 0 else 0

        metrics = {
            "area_ratio": float(area_ratio),
            "corners": corners.tolist(),
            "polygon_area": float(polygon_area),
            "image_area": int(total_area),
        }

    result = {
        "image": str(image_path),
        "method": method,
        "corners_detected": corners is not None,
        "output_overlay": str(output_path),
        "metrics": metrics,
    }

    logger.info(f"Result: method={method}, detected={corners is not None}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Test docTR document detection baseline")
    parser.add_argument("--images", type=str, required=True, help="Path to directory containing test images")
    parser.add_argument("--output", type=str, default="./parameter_tests/baseline", help="Output directory for results")
    parser.add_argument("--config", type=str, help="Path to config file (optional, uses defaults)")

    args = parser.parse_args()

    images_dir = Path(args.images)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / "baseline_test.log"
    logger = setup_logging(log_file)

    logger.info("Starting baseline document detection test")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Find image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        logger.error(f"No image files found in {images_dir}")
        return

    logger.info(f"Found {len(image_files)} images to test")

    # Create detector with default config
    config = DocumentPreprocessorConfig()
    detector = DocumentDetector(
        logger=logger,
        min_area_ratio=config.document_detection_min_area_ratio,
        use_adaptive=config.document_detection_use_adaptive,
        use_fallback=config.document_detection_use_fallback_box,
        use_camscanner=config.document_detection_use_camscanner,
    )

    logger.info("Detector configuration:")
    logger.info(f"  min_area_ratio: {config.document_detection_min_area_ratio}")
    logger.info(f"  use_adaptive: {config.document_detection_use_adaptive}")
    logger.info(f"  use_fallback: {config.document_detection_use_fallback_box}")
    logger.info(f"  use_camscanner: {config.document_detection_use_camscanner}")

    # Test each image
    results = []
    for image_file in image_files:
        try:
            result = test_document_detection(image_file, detector, output_dir, logger)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {image_file}: {e}")
            results.append({"image": str(image_file), "error": str(e)})

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    successful = sum(1 for r in results if r.get("corners_detected", False))
    total = len(results)

    logger.info(f"Test completed: {successful}/{total} images had corners detected")
    logger.info(f"Results saved to: {results_file}")

    # Print summary to console
    print("\nSummary:")
    print(f"Images processed: {total}")
    print(f"Successful detections: {successful}")
    print(f"Success rate: {successful / total * 100:.1f}%")


if __name__ == "__main__":
    main()
