#!/usr/bin/env python3
"""Benchmark data loading and transform optimisations via Hydra datasets."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

import albumentations as A
import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image

from ocr.datasets import ValidatedOCRDataset
from ocr.datasets.transforms import DBTransforms
from ocr.utils.image_loading import get_image_loader_info, load_image_optimized
from ocr.utils.path_utils import get_path_resolver

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_val_transform(interpolation=1):  # cv2.INTER_LINEAR
    """Create validation transform pipeline with specified interpolation."""
    transforms = [
        A.LongestMaxSize(max_size=640, interpolation=interpolation, p=1.0),
        A.PadIfNeeded(min_width=640, min_height=640, border_mode=0, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    keypoint_params = A.KeypointParams(format="xy", remove_invisible=True)
    return DBTransforms(transforms, keypoint_params)


def benchmark_image_loading(dataset: ValidatedOCRDataset, num_samples: int = 50):
    """Benchmark different image loading methods."""
    timings = defaultdict(list)

    logger.info(f"Benchmarking image loading for {num_samples} samples...")

    for i in range(min(num_samples, len(dataset))):
        image_filename = list(dataset.anns.keys())[i]
        image_path = dataset.image_path / image_filename

        # Method 1: Original PIL loading
        start = time.time()
        pil_image = Image.open(image_path)
        pil_time = time.time() - start
        timings["pil_loading"].append(pil_time)
        pil_image.close()

        # Method 2: Optimized loading (TurboJPEG + PIL fallback)
        start = time.time()
        opt_image = load_image_optimized(image_path)
        opt_time = time.time() - start
        timings["optimized_loading"].append(opt_time)
        opt_image.close()

        speedup = pil_time / opt_time if opt_time > 0 else float("inf")
        logger.info(f"Sample {i + 1}: PIL={pil_time:.4f}s, Optimized={opt_time:.4f}s, Speedup={speedup:.2f}x")

    return timings


def benchmark_transforms(dataset: ValidatedOCRDataset, num_samples: int = 50):
    """Benchmark transform performance with different interpolations."""
    timings = defaultdict(list)

    logger.info(f"Benchmarking transforms for {num_samples} samples...")

    # Create transforms with different interpolations
    linear_transform = create_val_transform(interpolation=1)  # cv2.INTER_LINEAR
    cubic_transform = create_val_transform(interpolation=3)  # cv2.INTER_CUBIC

    for i in range(min(num_samples, len(dataset))):
        image_filename = list(dataset.anns.keys())[i]
        image_path = dataset.image_path / image_filename

        # Load image
        pil_image = load_image_optimized(image_path)
        image_array = np.array(pil_image)
        polygons = dataset.anns[image_filename] or None

        # Benchmark LINEAR interpolation
        start = time.time()
        linear_transform(image_array, polygons=polygons)
        linear_time = time.time() - start
        timings["linear_transform"].append(linear_time)

        # Benchmark CUBIC interpolation
        start = time.time()
        cubic_transform(image_array, polygons=polygons)
        cubic_time = time.time() - start
        timings["cubic_transform"].append(cubic_time)

        speedup = cubic_time / linear_time if linear_time > 0 else float("inf")
        logger.info(f"Sample {i + 1}: Linear={linear_time:.4f}s, Cubic={cubic_time:.4f}s, Speedup={speedup:.2f}x")

        pil_image.close()

    return timings


def benchmark_full_pipeline(dataset: ValidatedOCRDataset, num_samples: int = 50):
    """Benchmark the complete data loading pipeline."""
    timings = defaultdict(list)

    logger.info(f"Benchmarking full pipeline for {num_samples} samples...")

    for i in range(min(num_samples, len(dataset))):
        start_time = time.time()
        _ = dataset[i]  # Load the item
        total_time = time.time() - start_time
        timings["full_pipeline"].append(total_time)

        logger.info(f"Sample {i + 1}: Full pipeline = {total_time:.4f}s")

    return timings


def print_benchmark_summary(timings, title):
    """Print benchmark summary statistics."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")

    for method, times in timings.items():
        if times:
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)
            print(f"{method:20}: avg={avg_time:.4f}s ± {std_time:.4f}s (min={min_time:.4f}s, max={max_time:.4f}s)")


def calculate_speedup(baseline_times, optimized_times):
    """Calculate speedup statistics."""
    if not baseline_times or not optimized_times:
        return None

    baseline_avg = np.mean(baseline_times)
    optimized_avg = np.mean(optimized_times)

    if optimized_avg == 0:
        return float("inf")

    speedup = baseline_avg / optimized_avg
    return speedup


def _resolve_dataset(cfg: DictConfig, dataset_key: str) -> ValidatedOCRDataset:
    datasets_cfg = cfg.get("datasets")
    if datasets_cfg is None:
        raise KeyError("Hydra config is missing the 'datasets' section required for benchmarking")

    if dataset_key not in datasets_cfg:
        available = ", ".join(sorted(datasets_cfg.keys()))
        raise KeyError(f"Dataset '{dataset_key}' not found. Available datasets: {available}")

    dataset: ValidatedOCRDataset = instantiate(datasets_cfg[dataset_key])
    return dataset


def _resolve_benchmark_options(cfg: DictConfig) -> tuple[str, int, int]:
    options = cfg.get("benchmark_optimizations") or {}
    dataset_key = options.get("dataset_key", "val_dataset")
    num_samples = int(options.get("num_samples", 50))
    transform_samples = int(options.get("transform_samples", min(20, num_samples)))
    return dataset_key, num_samples, transform_samples


@hydra.main(config_path=str(get_path_resolver().config.config_dir), config_name="performance_test", version_base="1.2")
def main(cfg: DictConfig) -> dict[str, dict[str, list[float]]]:
    print("Data Loading Optimization Benchmark")
    print("=" * 40)

    loader_info = get_image_loader_info()
    print("Image Loading Backends:")
    print(f"  TurboJPEG: {'Available' if loader_info['turbojpeg_available'] else 'Not Available'}")
    print(f"  PIL: {'Available' if loader_info['pil_available'] else 'Not Available'}")
    print()

    dataset_key, num_samples, transform_samples = _resolve_benchmark_options(cfg)
    dataset = _resolve_dataset(cfg, dataset_key)

    print(f"Dataset key: {dataset_key}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Using {num_samples} samples for benchmarking")
    print()

    loading_timings = benchmark_image_loading(dataset, num_samples=num_samples)
    transform_timings = benchmark_transforms(dataset, num_samples=transform_samples)
    pipeline_timings = benchmark_full_pipeline(dataset, num_samples=num_samples)

    print_benchmark_summary(loading_timings, "IMAGE LOADING BENCHMARK")
    print_benchmark_summary(transform_timings, "TRANSFORM INTERPOLATION BENCHMARK")
    print_benchmark_summary(pipeline_timings, "FULL PIPELINE BENCHMARK")

    print(f"\n{'=' * 60}")
    print("SPEEDUP ANALYSIS")
    print(f"{'=' * 60}")

    loading_speedup = calculate_speedup(loading_timings.get("pil_loading", []), loading_timings.get("optimized_loading", []))
    if loading_speedup:
        print(f"Image Loading Speedup: {loading_speedup:.2f}x")

    transform_speedup = calculate_speedup(transform_timings.get("cubic_transform", []), transform_timings.get("linear_transform", []))
    if transform_speedup:
        print(f"Transform Speedup (Cubic → Linear): {transform_speedup:.2f}x")

    print("\nBenchmark completed successfully!")

    return {
        "loading": loading_timings,
        "transforms": transform_timings,
        "pipeline": pipeline_timings,
    }


if __name__ == "__main__":
    main()
