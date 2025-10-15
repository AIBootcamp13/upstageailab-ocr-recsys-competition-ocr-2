#!/usr/bin/env python3
"""
Debug script to trace the canonical_size bug in the OCR dataset pipeline.
This helps understand where the integer vs tuple issue originates.
"""

import sys

sys.path.append("/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2")

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def test_image_types():
    """Test different image object types and their .size attribute"""
    print("=== Testing Image Object Types ===")

    # Create a sample image
    sample_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Test PIL Image
    pil_image = Image.fromarray(sample_array)
    print(f"PIL Image.size: {pil_image.size} (type: {type(pil_image.size)})")

    # Test numpy array
    numpy_array = np.array(pil_image)
    print(f"Numpy array.shape: {numpy_array.shape} (type: {type(numpy_array.shape)})")
    print(f"Numpy array.size: {numpy_array.size} (type: {type(numpy_array.size)})")

    # Test normalized numpy array (float32)
    normalized_array = numpy_array.astype(np.float32) / 255.0
    print(f"Normalized array.size: {normalized_array.size} (type: {type(normalized_array.size)})")


def inspect_dataset_pipeline():
    """Inspect the actual dataset pipeline to see what types are being used"""
    print("\n=== Inspecting Dataset Pipeline ===")

    from ocr.datasets.base import Dataset as OCRDataset

    # Mock image loading config
    image_loading_config = {"use_turbojpeg": False, "turbojpeg_fallback": True}

    # Create dataset instance with minimal transform
    data_path = Path("/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets")

    try:
        dataset = OCRDataset(
            image_path=data_path / "images_val_canonical",
            annotation_path=data_path / "jsons/val.json",
            transform=lambda x: x,  # Identity transform for debugging
            preload_maps=False,
            preload_images=False,
            image_loading_config=image_loading_config,
        )

        print(f"Dataset created with {len(dataset)} samples")

        # Test first item
        try:
            item = dataset[0]
            image = item["image"]
            shape = item["shape"]

            print("Sample Item Analysis:")
            print(f"  Image type: {type(image)}")
            print(f"  Image shape (if numpy): {getattr(image, 'shape', 'N/A')}")
            print(f"  Image size: {getattr(image, 'size', 'N/A')}")
            print(f"  Stored shape: {shape} (type: {type(shape)})")
            print(f"  Raw size: {item.get('raw_size', 'N/A')}")

        except Exception as e:
            print(f"Error loading item: {e}")

    except Exception as e:
        print(f"Error creating dataset: {e}")


def test_collate_function():
    """Test the collate function with different canonical_size types"""
    print("\n=== Testing Collate Function ===")

    from ocr.datasets.db_collate_fn import DBCollateFN

    # Create mock batch data
    mock_batch = [
        {
            "image": torch.randn(3, 224, 224),
            "image_filename": "test1.jpg",
            "image_path": "/path/to/test1.jpg",
            "shape": (224, 224),  # Correct tuple
            "raw_size": (224, 224),
            "orientation": 1,
        },
        {
            "image": torch.randn(3, 224, 224),
            "image_filename": "test2.jpg",
            "image_path": "/path/to/test2.jpg",
            "shape": 50176,  # Incorrect integer (224*224*1)
            "raw_size": (224, 224),
            "orientation": 1,
        },
    ]

    collate_fn = DBCollateFN(shrink_ratio=0.4)

    try:
        collated = collate_fn(mock_batch)
        print("Collate successful!")
        print(f"Canonical sizes: {collated['canonical_size']}")

        # Test what happens in lightning module
        for idx in range(len(mock_batch)):
            canonical_size = collated.get("canonical_size", [None])[idx]
            print(f"Item {idx} canonical_size: {canonical_size} (type: {type(canonical_size)})")

            try:
                canonical_tuple = tuple(canonical_size) if canonical_size is not None else None
                print(f"  Converted to tuple: {canonical_tuple}")
            except TypeError as e:
                print(f"  ERROR converting to tuple: {e}")

    except Exception as e:
        print(f"Collate failed: {e}")


if __name__ == "__main__":
    test_image_types()
    inspect_dataset_pipeline()
    test_collate_function()
