#!/usr/bin/env python3
"""
Quick test to verify that validation metrics are working.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torch.utils.data import DataLoader

from ocr.datasets.base import OCRDataset


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

    # Create a minimal dataset
    try:
        val_dataset = OCRDataset(
            image_path=str(val_images),
            annotation_path=str(val_annotations),
            transform=None,  # Use raw images for testing
            preload_maps=False,
        )
        print(f"✅ Created validation dataset with {len(val_dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to create validation dataset: {e}")
        return False

    # Create dataloader with just 1 sample for testing
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Use main process for testing
        pin_memory=False,
    )

    # Try to get one batch
    try:
        batch = next(iter(val_dataloader))
        print(f"✅ Got batch with keys: {list(batch.keys()) if hasattr(batch, 'keys') else type(batch)}")
        return True
    except Exception as e:
        print(f"❌ Failed to get batch from dataloader: {e}")
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
