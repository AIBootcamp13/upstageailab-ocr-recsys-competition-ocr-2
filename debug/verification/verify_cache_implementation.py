#!/usr/bin/env python3
"""
Quick test script to verify tensor cache statistics are working.

Usage:
    uv run python verify_cache_implementation.py
"""

import logging
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def test_cache_statistics():
    """Test that the cache statistics tracking is implemented correctly."""
    from ocr.datasets.base import Dataset as OCRDataset

    # Check that the class has the new methods and attributes
    assert hasattr(OCRDataset, "log_cache_statistics"), "Missing log_cache_statistics method"

    # Create a mock dataset to verify initialization
    dataset_path = Path("data/datasets/images_val_canonical")
    annotation_path = Path("data/datasets/jsons/val.json")

    if not dataset_path.exists() or not annotation_path.exists():
        print("⚠️  Test data not found, skipping dataset instantiation test")
        print("✅ Code structure verification passed")
        return

    # Verify instance variables are initialized
    from ocr.transforms.db_transforms import DBTransforms

    transforms = DBTransforms(
        image_short_side=640,
        shrink_ratio=0.4,
        thresh_min=0.3,
        thresh_max=0.7,
    )

    dataset = OCRDataset(
        image_path=str(dataset_path),
        annotation_path=str(annotation_path),
        transform=transforms.train_transform,
        cache_transformed_tensors=True,
    )

    # Verify cache tracking attributes exist
    assert hasattr(dataset, "_cache_hit_count"), "Missing _cache_hit_count attribute"
    assert hasattr(dataset, "_cache_miss_count"), "Missing _cache_miss_count attribute"
    assert hasattr(dataset, "tensor_cache"), "Missing tensor_cache attribute"

    assert dataset._cache_hit_count == 0, "Cache hit count should start at 0"
    assert dataset._cache_miss_count == 0, "Cache miss count should start at 0"

    print("✅ All cache statistics attributes initialized correctly")

    # Test that accessing an item increments miss count
    if len(dataset) > 0:
        dataset[0]
        assert dataset._cache_miss_count == 1, "Cache miss count should be 1 after first access"
        assert 0 in dataset.tensor_cache, "Item should be cached after first access"

        # Access same item again - should be a cache hit
        dataset[0]
        assert dataset._cache_hit_count == 1, "Cache hit count should be 1 after second access"

        print("✅ Cache hit/miss tracking working correctly")

        # Test statistics logging
        dataset.log_cache_statistics()
        assert dataset._cache_hit_count == 0, "Counters should reset after logging"
        assert dataset._cache_miss_count == 0, "Counters should reset after logging"

        print("✅ Statistics logging and reset working correctly")

    print("\n🎉 All verification tests passed!")


if __name__ == "__main__":
    test_cache_statistics()
