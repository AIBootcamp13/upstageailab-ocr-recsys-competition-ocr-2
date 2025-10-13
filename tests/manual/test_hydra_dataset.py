#!/usr/bin/env python3
"""
Simple script to test dataset instantiation.
"""

from pathlib import Path
from unittest.mock import Mock

from ocr.datasets.base import ValidatedOCRDataset
from ocr.datasets.schemas import CacheConfig, DatasetConfig, ImageLoadingConfig
from ocr.utils.path_utils import setup_project_paths

setup_project_paths()


def test_dataset_instantiation():
    """Test that ValidatedOCRDataset can be instantiated."""
    print("Testing ValidatedOCRDataset instantiation...")

    # Create a minimal config
    cache_config = CacheConfig(cache_transformed_tensors=False, cache_images=True, cache_maps=True)
    image_loading_config = ImageLoadingConfig(use_turbojpeg=False, turbojpeg_fallback=False)

    # Use a dummy path that doesn't exist to test instantiation without actual data
    config = DatasetConfig(
        image_path=Path("/tmp/dummy_images"),
        annotation_path=None,  # No annotations for testing
        cache_config=cache_config,
        image_loading_config=image_loading_config,
    )

    try:
        # Create a mock transform
        mock_transform = Mock(return_value={"image": None, "polygons": [], "metadata": {}})

        # This should instantiate without errors
        dataset = ValidatedOCRDataset(config=config, transform=mock_transform)
        print("✅ ValidatedOCRDataset instantiated successfully!")
        print(f"   Dataset type: {type(dataset).__name__}")
        print(f"   Config: {config}")

        # Since we have no actual data, length should be 0
        print(f"   Length: {len(dataset)}")

    except Exception as e:
        print(f"❌ Failed to instantiate dataset: {e}")
        raise


if __name__ == "__main__":
    test_dataset_instantiation()
