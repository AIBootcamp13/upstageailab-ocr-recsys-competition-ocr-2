#!/usr/bin/env python3
"""
Quick test to verify that load_maps parameter works correctly.

Usage:
    python test_load_maps_disabled.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ocr.datasets.base import OCRDataset


def test_load_maps_disabled():
    """Test that load_maps=False prevents .npz files from being loaded."""

    print("Creating test dataset structure...")

    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create directories
        images_dir = tmpdir / "images"
        images_dir.mkdir()
        maps_dir = tmpdir / "images_maps"
        maps_dir.mkdir()

        # Create a dummy image
        img = Image.new("RGB", (100, 100), color="white")
        img.save(images_dir / "test.jpg")

        # Create a dummy .npz map file
        prob_map = np.random.rand(100, 100).astype(np.float32)
        thresh_map = np.random.rand(100, 100).astype(np.float32)
        np.savez(maps_dir / "test.npz", prob_map=prob_map, thresh_map=thresh_map)

        # Create annotation file
        annotations = {"images": {"test.jpg": {"words": {"word1": {"points": [[10, 10], [20, 10], [20, 20], [10, 20]]}}}}}

        anno_file = tmpdir / "annotations.json"
        with open(anno_file, "w") as f:
            json.dump(annotations, f)

        # Create a simple transform
        def simple_transform(image, polygons):
            import torch

            if isinstance(image, Image.Image):
                image = np.array(image)
            # Convert to tensor
            if image.dtype == np.uint8:
                image = torch.from_numpy(image).float() / 255.0
            else:
                image = torch.from_numpy(image).float()
            # Rearrange to CHW format
            if len(image.shape) == 3:
                image = image.permute(2, 0, 1)

            return {"image": image, "polygons": polygons, "inverse_matrix": np.eye(3)}

        print("\n--- Test 1: load_maps=False (maps should NOT be loaded) ---")
        dataset_no_maps = OCRDataset(
            image_path=str(images_dir), annotation_path=str(anno_file), transform=simple_transform, load_maps=False, preload_maps=False
        )

        item = dataset_no_maps[0]

        if "prob_map" in item:
            print("❌ FAILED: prob_map found in item when load_maps=False")
            return False
        if "thresh_map" in item:
            print("❌ FAILED: thresh_map found in item when load_maps=False")
            return False

        print("✅ PASSED: Maps NOT loaded when load_maps=False")

        print("\n--- Test 2: load_maps=True (maps SHOULD be loaded) ---")
        dataset_with_maps = OCRDataset(
            image_path=str(images_dir), annotation_path=str(anno_file), transform=simple_transform, load_maps=True, preload_maps=False
        )

        item = dataset_with_maps[0]

        if "prob_map" not in item:
            print("❌ FAILED: prob_map NOT found in item when load_maps=True")
            return False
        if "thresh_map" not in item:
            print("❌ FAILED: thresh_map NOT found in item when load_maps=True")
            return False

        print("✅ PASSED: Maps loaded when load_maps=True")

        # Verify the maps have the correct shape
        assert item["prob_map"].shape == (100, 100), f"Unexpected prob_map shape: {item['prob_map'].shape}"
        assert item["thresh_map"].shape == (100, 100), f"Unexpected thresh_map shape: {item['thresh_map'].shape}"
        print("✅ PASSED: Maps have correct shape")

        print("\n🎉 All tests passed!")
        return True


if __name__ == "__main__":
    success = test_load_maps_disabled()
    sys.exit(0 if success else 1)
