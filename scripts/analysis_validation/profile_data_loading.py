#!/usr/bin/env python3
"""
Profiling script to investigate bottlenecks in data loading pipeline.
Measures time spent in image loading, transforms, and other stages.
"""

import logging
import time
from collections import defaultdict
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from ocr.datasets import ValidatedOCRDataset
from ocr.datasets.schemas import DatasetConfig, ImageMetadata, PolygonData, TransformInput
from ocr.datasets.transforms import DBTransforms
from ocr.utils.polygon_utils import filter_degenerate_polygons

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_val_transform():
    """Create validation transform pipeline."""
    transforms = [
        A.LongestMaxSize(max_size=640, p=1.0),
        A.PadIfNeeded(min_width=640, min_height=640, border_mode=0, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    keypoint_params = A.KeypointParams(format="xy", remove_invisible=True)
    return DBTransforms(transforms, keypoint_params)


def profile_dataset_loading(dataset, num_samples=50):
    """Profile the dataset loading pipeline."""
    timings = defaultdict(list)

    logger.info(f"Profiling {num_samples} samples from dataset...")

    for i in range(min(num_samples, len(dataset))):
        start_time = time.time()

        # Time full loading
        dataset[i]  # Load the item
        timings["total_loading"].append(time.time() - start_time)

        logger.info(f"Sample {i + 1}/{num_samples}: Total time = {timings['total_loading'][-1]:.4f}s")

    return timings


def profile_dataloader_loading(dataloader, num_batches=10):
    """Profile DataLoader batch loading."""
    timings = defaultdict(list)

    logger.info(f"Profiling {num_batches} batches from DataLoader...")

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        batch_time = time.time()
        # The batch is already loaded here
        timings["batch_loading"].append(time.time() - batch_time)

        logger.info(f"Batch {i + 1}/{num_batches}: Time = {timings['batch_loading'][-1]:.4f}s")

    return timings


def profile_individual_stages(dataset, num_samples=10):
    """Profile individual stages by simulating the __getitem__ operations."""
    timings = defaultdict(list)

    logger.info(f"Profiling individual stages for {num_samples} samples...")

    for i in range(min(num_samples, len(dataset))):
        image_filename = list(dataset.anns.keys())[i]
        image_path = dataset.image_path / image_filename

        # Stage 1: PIL Image loading
        start = time.time()
        pil_image = Image.open(image_path)
        timings["pil_open"].append(time.time() - start)

        # Stage 2: Image normalization (EXIF handling)
        start = time.time()
        raw_width, raw_height = pil_image.size
        # Simulate normalize_pil_image (simplified)
        if pil_image.mode != "RGB":
            image = pil_image.convert("RGB")
        else:
            image = pil_image.copy()
        timings["image_normalize"].append(time.time() - start)

        # Stage 3: Transform application
        start = time.time()
        polygons = dataset.anns[image_filename] or None

        polygon_models = None
        if polygons:
            polygon_models = []
            for polygon in polygons:
                try:
                    polygon_models.append(PolygonData(points=np.asarray(polygon, dtype=np.float32)))
                except Exception:
                    continue

        image_array = np.array(image)
        metadata = ImageMetadata(
            filename=image_filename,
            path=image_path,
            original_shape=(image_array.shape[0], image_array.shape[1]),
            dtype=str(image_array.dtype),
            raw_size=(raw_width, raw_height),
            orientation=1,
            polygon_frame="raw",
        )

        transform_input = TransformInput(image=image_array, polygons=polygon_models, metadata=metadata)
        transformed = dataset.transform(transform_input)
        timings["transforms"].append(time.time() - start)

        # Stage 4: Polygon filtering
        start = time.time()
        transformed_polygons = transformed.get("polygons", []) or []
        if transformed_polygons:
            _ = filter_degenerate_polygons([np.asarray(poly) for poly in transformed_polygons])
        timings["polygon_filter"].append(time.time() - start)

        # Stage 5: Map loading
        start = time.time()
        maps_dir = dataset.image_path.parent / f"{dataset.image_path.name}_maps"
        map_filename = maps_dir / f"{Path(image_filename).stem}.npz"
        if map_filename.exists():
            maps_data = np.load(map_filename)
            # Load the maps (but don't assign to variables)
            _ = maps_data["prob_map"]
            _ = maps_data["thresh_map"]
        timings["map_loading"].append(time.time() - start)

        pil_image.close()

        logger.info(
            f"Sample {i + 1}: PIL={timings['pil_open'][-1]:.4f}s, "
            f"Normalize={timings['image_normalize'][-1]:.4f}s, "
            f"Transform={timings['transforms'][-1]:.4f}s, "
            f"Filter={timings['polygon_filter'][-1]:.4f}s, "
            f"Map={timings['map_loading'][-1]:.4f}s"
        )

    return timings


def print_summary(timings):
    """Print timing summary statistics."""
    print("\n" + "=" * 60)
    print("PROFILING SUMMARY")
    print("=" * 60)

    for stage, times in timings.items():
        if times:
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            total_time = np.sum(times)
            print(f"{stage:15}: avg={avg_time:.4f}s, min={min_time:.4f}s, max={max_time:.4f}s, total={total_time:.4f}s")


def main():
    # Configuration
    image_path = "data/datasets/images_val_canonical"
    annotation_path = "data/datasets/jsons/val.json"
    num_samples = 50
    batch_size = 16
    num_workers = 8

    # Create dataset
    transform = create_val_transform()
    config = DatasetConfig(
        image_path=Path(image_path),
        annotation_path=Path(annotation_path),
        preload_maps=False,
        load_maps=False,
        preload_images=False,
    )
    dataset = ValidatedOCRDataset(config=config, transform=transform)

    print(f"Dataset size: {len(dataset)}")
    print(f"Using {num_samples} samples for profiling")

    # Profile individual stages
    stage_timings = profile_individual_stages(dataset, num_samples=min(10, num_samples))

    # Profile full dataset loading
    full_timings = profile_dataset_loading(dataset, num_samples=num_samples)

    # Profile DataLoader (with error handling)
    try:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True
        )
        dataloader_timings = profile_dataloader_loading(dataloader, num_batches=num_samples // batch_size)
    except Exception as e:
        logger.warning(f"DataLoader profiling failed: {e}")
        dataloader_timings = {}

    # Print summaries
    print_summary(stage_timings)
    print_summary(full_timings)
    if dataloader_timings:
        print_summary(dataloader_timings)


if __name__ == "__main__":
    main()
