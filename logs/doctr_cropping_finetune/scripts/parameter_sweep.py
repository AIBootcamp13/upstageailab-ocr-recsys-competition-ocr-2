#!/usr/bin/env python3
"""
Parameter sweep script for docTR cropping fine-tuning.

Tests different combinations of document detection parameters to find optimal settings.
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

from ocr.datasets.preprocessing.detector import DocumentDetector


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


def test_parameters_on_image(image_path: Path, params: dict[str, Any], output_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    """Test specific parameters on a single image."""
    logger.debug(f"Testing {image_path.name} with params: {params}")

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return {"image": str(image_path), "error": "Failed to load"}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create detector with custom params
    detector = DocumentDetector(
        logger=logger,
        min_area_ratio=params["min_area_ratio"],
        use_adaptive=params["use_adaptive"],
        use_fallback=params["use_fallback"],
        use_camscanner=params["use_camscanner"],
    )

    # Detect document
    corners, method = detector.detect(image_rgb)

    # Generate overlay
    overlay_image = draw_document_overlay(image_rgb, corners)

    # Create parameter-specific output name
    param_str = f"area{params['min_area_ratio']}_adapt{int(params['use_adaptive'])}_fb{int(params['use_fallback'])}_cam{int(params['use_camscanner'])}"
    output_path = output_dir / f"{image_path.stem}_{param_str}_overlay.png"
    overlay_image.save(output_path)

    # Calculate metrics
    metrics = {}
    if corners is not None:
        height, width = image_rgb.shape[:2]
        total_area = height * width

        corners_reshaped = corners.reshape(-1, 2)
        polygon_area = cv2.contourArea(corners_reshaped.astype(np.float32))
        area_ratio = polygon_area / total_area if total_area > 0 else 0

        metrics = {
            "area_ratio": float(area_ratio),
            "corners": corners.tolist(),
            "polygon_area": float(polygon_area),
            "image_area": int(total_area),
        }

    return {
        "image": str(image_path),
        "parameters": params,
        "method": method,
        "corners_detected": corners is not None,
        "output_overlay": str(output_path),
        "metrics": metrics,
    }


def generate_parameter_combinations() -> list[dict[str, Any]]:
    """Generate all parameter combinations to test."""
    combinations = []

    # Area ratios to test
    area_ratios = [0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30]

    # Method combinations
    method_configs = [
        {"use_adaptive": False, "use_fallback": False, "use_camscanner": False},  # Canny only
        {"use_adaptive": True, "use_fallback": False, "use_camscanner": False},  # Canny + adaptive
        {"use_adaptive": True, "use_fallback": True, "use_camscanner": False},  # All except camscanner
        {"use_adaptive": False, "use_fallback": False, "use_camscanner": True},  # CamScanner only
        {"use_adaptive": True, "use_fallback": True, "use_camscanner": True},  # All methods
    ]

    for area_ratio in area_ratios:
        for method_config in method_configs:
            combinations.append({"min_area_ratio": area_ratio, **method_config})

    return combinations


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for docTR document detection")
    parser.add_argument("--images", type=str, required=True, help="Path to directory containing test images")
    parser.add_argument("--output", type=str, default="./parameter_tests/sweep", help="Output directory for results")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to test (for quick testing)")

    args = parser.parse_args()

    images_dir = Path(args.images)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / "parameter_sweep.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    logger.info("Starting parameter sweep for document detection")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Find image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        logger.error(f"No image files found in {images_dir}")
        return

    if args.max_images:
        image_files = image_files[: args.max_images]
        logger.info(f"Limited to {len(image_files)} images for testing")

    logger.info(f"Testing {len(image_files)} images")

    # Generate parameter combinations
    param_combinations = generate_parameter_combinations()
    logger.info(f"Testing {len(param_combinations)} parameter combinations")

    # Test each combination on each image
    all_results = []

    for i, params in enumerate(param_combinations):
        logger.info(f"Testing parameter set {i + 1}/{len(param_combinations)}: {params}")

        param_results = []
        for image_file in image_files:
            try:
                result = test_parameters_on_image(image_file, params, output_dir, logger)
                param_results.append(result)
            except Exception as e:
                logger.error(f"Error testing {image_file} with params {params}: {e}")
                param_results.append({"image": str(image_file), "parameters": params, "error": str(e)})

        all_results.extend(param_results)

        # Progress update
        successful = sum(1 for r in param_results if r.get("corners_detected", False))
        total = len(param_results)
        logger.info(f"Parameter set {i + 1} results: {successful}/{total} detections")

    # Save results
    results_file = output_dir / "sweep_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate summary
    summary = generate_summary(all_results)
    summary_file = output_dir / "sweep_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Sweep completed. Results saved to: {results_file}")
    logger.info(f"Summary saved to: {summary_file}")

    # Print top performing configurations
    print("\nTop 5 parameter configurations:")
    sorted_configs = sorted(summary.items(), key=lambda x: x[1]["success_rate"], reverse=True)
    for i, (config_str, stats) in enumerate(sorted_configs[:5]):
        print(f"{i + 1}. {config_str}: {stats['success_rate']:.1f}% success ({stats['successful']}/{stats['total']})")


def generate_summary(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Generate summary statistics for each parameter configuration."""
    config_stats = {}

    for result in results:
        if "error" in result:
            continue

        params = result["parameters"]
        config_key = f"area{params['min_area_ratio']}_adapt{int(params['use_adaptive'])}_fb{int(params['use_fallback'])}_cam{int(params['use_camscanner'])}"

        if config_key not in config_stats:
            config_stats[config_key] = {"parameters": params, "total": 0, "successful": 0, "methods": {}}

        config_stats[config_key]["total"] += 1
        if result["corners_detected"]:
            config_stats[config_key]["successful"] += 1

        method = result.get("method", "unknown")
        if method not in config_stats[config_key]["methods"]:
            config_stats[config_key]["methods"][method] = 0
        config_stats[config_key]["methods"][method] += 1

    # Calculate success rates
    for config_key, stats in config_stats.items():
        stats["success_rate"] = stats["successful"] / stats["total"] * 100

    return config_stats


if __name__ == "__main__":
    main()
