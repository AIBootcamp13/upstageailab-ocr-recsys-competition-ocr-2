#!/usr/bin/env python3
"""
Profile transform pipeline to identify bottlenecks.

This script measures the time spent in each transform step to guide optimization efforts.
"""

import time
from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

from ocr.utils.orientation import normalize_pil_image


def profile_individual_transforms(image_array, num_iterations=100):
    """Profile each transform individually to identify bottlenecks."""

    transforms_to_profile = [
        ("LongestMaxSize", A.LongestMaxSize(max_size=640, interpolation=1, p=1.0)),
        ("PadIfNeeded", A.PadIfNeeded(min_width=640, min_height=640, border_mode=0, p=1.0)),
        ("Normalize", A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
        ("ToTensorV2", ToTensorV2()),
    ]

    results = {}

    for name, transform in transforms_to_profile:
        times = []
        for _ in range(num_iterations):
            img = image_array.copy()
            start = time.perf_counter()
            _ = transform(image=img)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms

        results[name] = {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
        }

    return results


def profile_full_pipeline(image_array, keypoints=None, num_iterations=100):
    """Profile the full transform pipeline."""

    full_pipeline = A.Compose(
        [
            A.LongestMaxSize(max_size=640, interpolation=1, p=1.0),
            A.PadIfNeeded(min_width=640, min_height=640, border_mode=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=True) if keypoints else None,
    )

    times = []
    for _ in range(num_iterations):
        img = image_array.copy()
        start = time.perf_counter()
        if keypoints:
            _ = full_pipeline(image=img, keypoints=keypoints)
        else:
            _ = full_pipeline(image=img)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def profile_dataset_loading(dataset, num_samples=50):
    """Profile actual dataset loading to get real-world measurements."""

    timing_breakdown = {
        "total": [],
        "image_load": [],
        "transform": [],
        "polygon_processing": [],
    }

    for idx in tqdm(range(min(num_samples, len(dataset))), desc="Profiling dataset"):
        start_total = time.perf_counter()

        # This calls __getitem__ which includes all steps
        _ = dataset[idx]

        total_time = (time.perf_counter() - start_total) * 1000
        timing_breakdown["total"].append(total_time)

    return {
        "mean_ms": np.mean(timing_breakdown["total"]),
        "std_ms": np.std(timing_breakdown["total"]),
        "min_ms": np.min(timing_breakdown["total"]),
        "max_ms": np.max(timing_breakdown["total"]),
    }


def main():
    """Main profiling entry point."""

    print("=" * 80)
    print("Transform Pipeline Profiling")
    print("=" * 80)
    print()

    # Load sample images from validation dataset
    dataset_path = Path("data/datasets/images_val_canonical")
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return

    # Load a few sample images with different sizes
    sample_images = []
    for img_file in list(dataset_path.glob("*.jpg"))[:5]:
        pil_img = Image.open(img_file)
        normalized_img, _ = normalize_pil_image(pil_img)
        if normalized_img.mode != "RGB":
            rgb_img = normalized_img.convert("RGB")
        else:
            rgb_img = normalized_img
        img_array = np.array(rgb_img)
        sample_images.append((img_file.name, img_array))
        pil_img.close()
        if normalized_img is not pil_img:
            normalized_img.close()

    if not sample_images:
        print("Error: No sample images found")
        return

    print(f"Loaded {len(sample_images)} sample images")
    print()

    # Profile individual transforms on first sample image
    print("-" * 80)
    print("INDIVIDUAL TRANSFORM PROFILING (100 iterations per transform)")
    print("-" * 80)
    sample_name, sample_array = sample_images[0]
    print(f"Sample image: {sample_name} (shape: {sample_array.shape})")
    print()

    individual_results = profile_individual_transforms(sample_array, num_iterations=100)

    print(f"{'Transform':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-" * 80)

    total_mean = 0
    for name, stats in individual_results.items():
        print(f"{name:<20} {stats['mean_ms']:<12.3f} {stats['std_ms']:<12.3f} {stats['min_ms']:<12.3f} {stats['max_ms']:<12.3f}")
        total_mean += stats["mean_ms"]

    print("-" * 80)
    print(f"{'TOTAL (sum)':<20} {total_mean:<12.3f}")
    print()

    # Profile full pipeline
    print("-" * 80)
    print("FULL PIPELINE PROFILING (100 iterations)")
    print("-" * 80)

    # Create sample keypoints (simulating polygons)
    sample_keypoints = [(100, 100), (200, 100), (200, 200), (100, 200)]

    full_results_no_kp = profile_full_pipeline(sample_array, keypoints=None, num_iterations=100)
    full_results_with_kp = profile_full_pipeline(sample_array, keypoints=sample_keypoints, num_iterations=100)

    print(f"Without keypoints: {full_results_no_kp['mean_ms']:.3f} ± {full_results_no_kp['std_ms']:.3f} ms")
    print(f"With keypoints:    {full_results_with_kp['mean_ms']:.3f} ± {full_results_with_kp['std_ms']:.3f} ms")
    print(f"Keypoint overhead: {full_results_with_kp['mean_ms'] - full_results_no_kp['mean_ms']:.3f} ms")
    print()

    # Profile across different image sizes
    print("-" * 80)
    print("PROFILING ACROSS DIFFERENT IMAGE SIZES")
    print("-" * 80)

    for img_name, img_array in sample_images:
        result = profile_full_pipeline(img_array, keypoints=None, num_iterations=50)
        print(f"{img_name:<40} {img_array.shape} -> {result['mean_ms']:.3f} ± {result['std_ms']:.3f} ms")

    print()

    # Calculate percentage breakdown
    print("-" * 80)
    print("PERFORMANCE BREAKDOWN (% of total time)")
    print("-" * 80)

    for name, stats in individual_results.items():
        percentage = (stats["mean_ms"] / total_mean) * 100
        print(f"{name:<20} {percentage:>6.2f}%")

    print()

    # Recommendations
    print("=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Find the slowest transform
    slowest_transform = max(individual_results.items(), key=lambda x: x[1]["mean_ms"])
    print(f"🔴 Slowest transform: {slowest_transform[0]} ({slowest_transform[1]['mean_ms']:.3f} ms)")
    print()

    # Provide specific recommendations
    if slowest_transform[0] == "LongestMaxSize":
        print("Recommendation for LongestMaxSize:")
        print("  - Already using INTER_LINEAR (fastest OpenCV interpolation)")
        print("  - Consider using cv2.resize directly with cv2.INTER_NEAREST for even faster (but lower quality)")
        print("  - Or cache resized images if validation set is small")
    elif slowest_transform[0] == "Normalize":
        print("Recommendation for Normalize:")
        print("  - Consider pre-normalizing images and storing as float32")
        print("  - Or use custom normalization kernel for better performance")
    elif slowest_transform[0] == "ToTensorV2":
        print("Recommendation for ToTensorV2:")
        print("  - ToTensorV2 is already optimized")
        print("  - Consider using torch.from_numpy directly if possible")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
