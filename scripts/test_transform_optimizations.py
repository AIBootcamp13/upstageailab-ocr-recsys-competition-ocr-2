#!/usr/bin/env python3
"""
Test different transform optimizations for data loading performance.
"""

import logging
import time
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image

from ocr.datasets.transforms import DBTransforms

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def time_transform(transform, image, polygons, num_runs=10):
    """Time a transform operation."""
    times = []
    for _ in range(num_runs):
        start = time.time()
        transform(image=image, polygons=polygons)
        times.append(time.time() - start)
    return np.mean(times), np.std(times)


def create_test_transforms():
    """Create different transform configurations to test."""
    transforms = {}

    # Current validation transforms
    transforms["current_val"] = DBTransforms(
        [
            A.LongestMaxSize(max_size=640, p=1.0),
            A.PadIfNeeded(min_width=640, min_height=640, border_mode=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
        A.KeypointParams(format="xy", remove_invisible=True),
    )

    # Simpler resize (no padding)
    transforms["simple_resize"] = DBTransforms(
        [
            A.Resize(height=640, width=640, p=1),  # Direct resize instead of LongestMaxSize + Pad
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
        A.KeypointParams(format="xy", remove_invisible=True),
    )

    # No normalization (for comparison)
    transforms["no_normalize"] = DBTransforms(
        [
            A.LongestMaxSize(max_size=640, p=1.0),
            A.PadIfNeeded(min_width=640, min_height=640, border_mode=0, p=1.0),
        ],
        A.KeypointParams(format="xy", remove_invisible=True),
    )

    # Only resize, no padding
    transforms["resize_only"] = DBTransforms(
        [
            A.LongestMaxSize(max_size=640, p=1.0),
        ],
        A.KeypointParams(format="xy", remove_invisible=True),
    )

    return transforms


def load_sample_image():
    """Load a sample image for testing."""
    image_path = Path("data/datasets/images_val_canonical") / "img_1.jpg"
    if not image_path.exists():
        # Try to find any image
        images = list(Path("data/datasets/images_val_canonical").glob("*.jpg"))
        if images:
            image_path = images[0]
        else:
            raise FileNotFoundError("No images found in validation dataset")

    pil_image = Image.open(image_path)
    image = np.array(pil_image)
    pil_image.close()

    # Create dummy polygons
    polygons = [np.array([[[100, 100], [200, 100], [200, 150], [100, 150]]], dtype=np.float32)]

    return image, polygons


def main():
    print("Testing different transform configurations...")

    # Load test data
    try:
        image, polygons = load_sample_image()
        print(f"Test image shape: {image.shape}")
    except Exception as e:
        print(f"Failed to load test image: {e}")
        return

    # Create transforms
    transforms = create_test_transforms()

    # Time each transform
    results = {}
    for name, transform in transforms.items():
        print(f"\nTesting {name}...")
        try:
            mean_time, std_time = time_transform(transform, image, polygons, num_runs=20)
            results[name] = {"mean": mean_time, "std": std_time}
            print(".4f")
        except Exception as e:
            print(f"  Failed: {e}")
            results[name] = {"mean": float("inf"), "std": 0}

    # Print comparison
    print("\n" + "=" * 60)
    print("TRANSFORM PERFORMANCE COMPARISON")
    print("=" * 60)

    baseline = results.get("current_val", {}).get("mean", float("inf"))
    for name, timing in results.items():
        mean_time = timing["mean"]
        if mean_time == float("inf"):
            print("20")
        else:
            baseline / mean_time if baseline != float("inf") else 1.0
            print("20")


if __name__ == "__main__":
    main()
