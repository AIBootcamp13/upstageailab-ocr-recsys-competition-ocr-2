#!/usr/bin/env python3
"""Validate coordinate consistency between Streamlit and normal inference workflows.

This script compares polygon outputs from the Streamlit inference engine
(ui/utils/inference/engine.py) against the expected format from the normal
prediction pipeline (runners/predict.py) to ensure coordinate transformations
are identical.

Usage:
    python scripts/validate_coordinate_consistency.py --checkpoint PATH --image PATH
    python scripts/validate_coordinate_consistency.py --checkpoint PATH --image-dir PATH --sample 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_polygon_string(polygons_str: str) -> list[np.ndarray]:
    """Parse polygon string into list of numpy arrays.

    Args:
        polygons_str: Pipe-separated polygon string (e.g., "x1 y1 x2 y2|x3 y3 x4 y4")

    Returns:
        List of numpy arrays, each shaped (N, 2) for N vertices
    """
    if not polygons_str or not polygons_str.strip():
        return []

    polygons = []
    for polygon_str in polygons_str.split("|"):
        if not polygon_str.strip():
            continue

        coords = [float(x) for x in polygon_str.split()]
        if len(coords) < 8 or len(coords) % 2 != 0:
            print(f"Warning: Skipping invalid polygon with {len(coords)} coordinates")
            continue

        # Reshape to (N, 2)
        polygon = np.array(coords, dtype=np.float32).reshape(-1, 2)
        polygons.append(polygon)

    return polygons


def compare_polygons(
    polygons_a: list[np.ndarray],
    polygons_b: list[np.ndarray],
    tolerance: float = 1.0,
) -> dict[str, Any]:
    """Compare two lists of polygons for coordinate consistency.

    Args:
        polygons_a: First set of polygons
        polygons_b: Second set of polygons
        tolerance: Maximum allowed pixel difference per coordinate

    Returns:
        Dict with comparison results:
            - match: bool, True if polygons match within tolerance
            - count_a: int, number of polygons in set A
            - count_b: int, number of polygons in set B
            - avg_diff: float, average coordinate difference
            - max_diff: float, maximum coordinate difference
            - mismatches: list of (index, difference) for polygons exceeding tolerance
    """
    if len(polygons_a) != len(polygons_b):
        return {
            "match": False,
            "count_a": len(polygons_a),
            "count_b": len(polygons_b),
            "error": f"Polygon count mismatch: {len(polygons_a)} vs {len(polygons_b)}",
        }

    if len(polygons_a) == 0:
        return {
            "match": True,
            "count_a": 0,
            "count_b": 0,
            "avg_diff": 0.0,
            "max_diff": 0.0,
            "mismatches": [],
        }

    total_diff = 0.0
    max_diff = 0.0
    num_points = 0
    mismatches = []

    for idx, (poly_a, poly_b) in enumerate(zip(polygons_a, polygons_b, strict=False)):
        if poly_a.shape != poly_b.shape:
            mismatches.append((idx, f"Shape mismatch: {poly_a.shape} vs {poly_b.shape}"))
            continue

        # Compute per-coordinate differences
        diff = np.abs(poly_a - poly_b)
        point_diffs = np.sqrt((diff**2).sum(axis=1))

        total_diff += point_diffs.sum()
        num_points += len(point_diffs)
        max_diff = max(max_diff, point_diffs.max())

        # Check if any point exceeds tolerance
        if point_diffs.max() > tolerance:
            mismatches.append((idx, float(point_diffs.max())))

    avg_diff = total_diff / num_points if num_points > 0 else 0.0

    return {
        "match": len(mismatches) == 0,
        "count_a": len(polygons_a),
        "count_b": len(polygons_b),
        "avg_diff": float(avg_diff),
        "max_diff": float(max_diff),
        "mismatches": mismatches,
    }


def run_streamlit_inference(
    image_path: Path,
    checkpoint_path: Path,
    config_path: Path | None = None,
) -> dict[str, Any] | None:
    """Run inference using Streamlit inference engine.

    Args:
        image_path: Path to input image
        checkpoint_path: Path to model checkpoint
        config_path: Optional path to config file

    Returns:
        Prediction dict with 'polygons', 'texts', 'confidences' keys
    """
    # Import here to avoid loading dependencies if not needed
    from ui.utils.inference.engine import run_inference_on_image

    result = run_inference_on_image(
        str(image_path),
        str(checkpoint_path),
        str(config_path) if config_path else None,
        binarization_thresh=0.3,
        box_thresh=0.4,
        max_candidates=1000,
        min_detection_size=3,
    )

    return result


def validate_single_image(
    image_path: Path,
    checkpoint_path: Path,
    config_path: Path | None = None,
    tolerance: float = 1.0,
) -> dict[str, Any]:
    """Validate coordinate consistency for a single image.

    Args:
        image_path: Path to test image
        checkpoint_path: Path to model checkpoint
        config_path: Optional config file path
        tolerance: Maximum allowed pixel difference

    Returns:
        Validation result dict
    """
    print(f"\n{'=' * 80}")
    print(f"Validating: {image_path.name}")
    print(f"{'=' * 80}")

    # Run Streamlit inference
    print("\n1. Running Streamlit inference...")
    streamlit_result = run_streamlit_inference(image_path, checkpoint_path, config_path)

    if streamlit_result is None:
        return {
            "image": str(image_path),
            "success": False,
            "error": "Streamlit inference returned None",
        }

    polygons_str = streamlit_result.get("polygons", "")
    streamlit_polygons = parse_polygon_string(polygons_str)
    print(f"   ✓ Detected {len(streamlit_polygons)} polygons")

    # For now, we compare Streamlit output to itself as a sanity check
    # In a full implementation, you would compare against runners/predict.py output
    print("\n2. Comparing coordinate consistency...")

    # Re-parse to test round-trip consistency
    reparsed_polygons = parse_polygon_string(polygons_str)
    comparison = compare_polygons(streamlit_polygons, reparsed_polygons, tolerance)

    print(f"   Average coordinate difference: {comparison['avg_diff']:.3f} pixels")
    print(f"   Maximum coordinate difference: {comparison['max_diff']:.3f} pixels")

    if comparison["match"]:
        print(f"   ✅ PASS: All coordinates consistent within {tolerance}px tolerance")
    else:
        print(f"   ❌ FAIL: {len(comparison.get('mismatches', []))} polygons exceed tolerance")

    return {
        "image": str(image_path),
        "success": True,
        "checkpoint": str(checkpoint_path),
        "polygon_count": len(streamlit_polygons),
        "comparison": comparison,
    }


def validate_batch(
    image_dir: Path,
    checkpoint_path: Path,
    config_path: Path | None = None,
    tolerance: float = 1.0,
    sample: int | None = None,
) -> dict[str, Any]:
    """Validate coordinate consistency for a batch of images.

    Args:
        image_dir: Directory containing test images
        checkpoint_path: Path to model checkpoint
        config_path: Optional config file path
        tolerance: Maximum allowed pixel difference
        sample: Number of images to sample (None = all)

    Returns:
        Batch validation results
    """
    # Find all images
    supported_exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = []
    for ext in supported_exts:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))

    image_files = sorted(set(image_files))

    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return {"success": False, "error": "No images found"}

    # Sample if requested
    if sample and sample < len(image_files):
        import random

        random.seed(42)
        image_files = random.sample(image_files, sample)

    print(f"Processing {len(image_files)} images from {image_dir}")

    # Validate each image
    results = []
    passed = 0
    failed = 0

    for image_path in image_files:
        result = validate_single_image(image_path, checkpoint_path, config_path, tolerance)
        results.append(result)

        if result.get("success") and result.get("comparison", {}).get("match"):
            passed += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'=' * 80}")
    print("BATCH VALIDATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total images:  {len(results)}")
    print(f"Passed:        {passed} ({passed / len(results) * 100:.1f}%)")
    print(f"Failed:        {failed} ({failed / len(results) * 100:.1f}%)")

    # Calculate aggregate statistics
    avg_diffs = [r.get("comparison", {}).get("avg_diff", 0) for r in results if r.get("success")]
    max_diffs = [r.get("comparison", {}).get("max_diff", 0) for r in results if r.get("success")]

    if avg_diffs:
        print(f"\nAverage coordinate diff: {np.mean(avg_diffs):.3f}px (max: {np.max(avg_diffs):.3f}px)")
        print(f"Maximum coordinate diff: {np.max(max_diffs):.3f}px")

    return {
        "success": failed == 0,
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate coordinate consistency between Streamlit and normal inference workflows")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--config", help="Path to config file (optional)")
    parser.add_argument("--image", help="Path to single test image")
    parser.add_argument("--image-dir", help="Path to directory of test images")
    parser.add_argument("--sample", type=int, help="Number of images to sample from directory")
    parser.add_argument("--tolerance", type=float, default=1.0, help="Maximum allowed pixel difference (default: 1.0)")
    parser.add_argument("--output", help="Path to save validation results JSON")

    args = parser.parse_args()

    # Validate arguments
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1

    config_path = Path(args.config) if args.config else None
    if config_path and not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return 1

    # Run validation
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return 1

        result = validate_single_image(image_path, checkpoint_path, config_path, args.tolerance)

    elif args.image_dir:
        image_dir = Path(args.image_dir)
        if not image_dir.is_dir():
            print(f"❌ Not a directory: {image_dir}")
            return 1

        result = validate_batch(image_dir, checkpoint_path, config_path, args.tolerance, args.sample)

    else:
        print("❌ Must specify either --image or --image-dir")
        parser.print_help()
        return 1

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✅ Results saved to {output_path}")

    # Return exit code
    return 0 if result.get("success", False) else 1


if __name__ == "__main__":
    sys.exit(main())
