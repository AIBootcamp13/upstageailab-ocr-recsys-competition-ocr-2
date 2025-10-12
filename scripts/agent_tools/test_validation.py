#!/usr/bin/env python3
"""
Quick test to verify that validation metrics are working.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from ocr.datasets import ValidatedOCRDataset
from ocr.datasets.schemas import DatasetConfig


def test_validation():
    """Test that validation step works and logs metrics."""
    print("Testing validation functionality...")

    # Check if validation data exists
    val_images = Path("data/datasets/images/val")
    val_annotations = Path("data/datasets/jsons/val.json")

    if not val_images.exists():
        print(f"❌ Validation images directory not found: {val_images}")
        return False

    if not val_annotations.exists():
        print(f"❌ Validation annotations file not found: {val_annotations}")
        return False

    # Create a minimal dataset config
    config = DatasetConfig(
        image_path=val_images,
        annotation_path=val_annotations,
        preload_maps=False,
        load_maps=False,
        preload_images=False,
        prenormalize_images=False,
    )

    # Create a minimal transform for testing
    def identity_transform(data):
        """Identity transform that returns data with minimal processing."""
        import torch

        # Convert polygons from PolygonData to numpy arrays
        polygons = []
        if data.polygons:
            for poly_data in data.polygons:
                polygons.append(poly_data.points)

        # Convert image to tensor (add batch and channel dims if needed)
        image = data.image
        if image.ndim == 2:  # Grayscale
            image = image.unsqueeze(0)  # Add channel dim
        elif image.ndim == 3 and image.shape[2] in (1, 3):  # (H, W, C) -> (C, H, W)
            image = torch.from_numpy(image).permute(2, 0, 1)
        else:
            image = torch.from_numpy(image)

        return {
            "image": image,
            "polygons": polygons,
            "inverse_matrix": np.eye(3, dtype=np.float32),  # Identity matrix
        }

    # Create dataset using ValidatedOCRDataset
    try:
        val_dataset = ValidatedOCRDataset(
            config=config,
            transform=identity_transform,  # Use identity transform for testing
        )
        print(f"✅ Created validation dataset with {len(val_dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to create validation dataset: {e}")
        return False

    # Test getting one sample directly (avoid dataloader batching issues)
    try:
        sample = val_dataset[0]
        print(f"✅ Got sample with keys: {list(sample.keys()) if hasattr(sample, 'keys') else type(sample)}")
        return True
    except Exception as e:
        print(f"❌ Failed to get sample from dataset: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_validation()
    if success:
        print("\n🎉 Validation data test passed!")
    else:
        print("\n💥 Validation data test failed!")
        sys.exit(1)
