"""
Unit tests for map loading and caching functionality in OCRDataset

Tests the robust loading of .npz map files, handling of corrupted/missing files,
and fallback behavior when maps cannot be loaded.

This addresses Phase 2.2 of the data pipeline testing plan to ensure
graceful handling of map loading failures and maintain caching performance.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from ocr.datasets.base import OCRDataset


class TestMapLoadingCaching:
    """Test cases for map loading and caching in OCRDataset."""

    @pytest.fixture
    def temp_dataset_structure(self):
        """Create a temporary dataset structure for testing."""
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

            # Create annotation file
            annotations = {"images": {"test.jpg": {"words": {"word1": {"points": [[10, 10], [20, 10], [20, 20], [10, 20]]}}}}}

            anno_file = tmpdir / "annotations.json"
            with open(anno_file, "w") as f:
                json.dump(annotations, f)

            yield tmpdir, images_dir, maps_dir, anno_file

    @pytest.fixture
    def simple_transform(self):
        """Create a simple transform for testing."""

        def transform(image, polygons):
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

        return transform

    def test_missing_maps_directory_graceful_handling(self, temp_dataset_structure, simple_transform):
        """Test that missing maps directory is handled gracefully during preloading."""
        tmpdir, images_dir, maps_dir, anno_file = temp_dataset_structure

        # Remove the maps directory
        maps_dir.rmdir()

        # Create dataset with preload_maps=True
        dataset = OCRDataset(
            image_path=str(images_dir), annotation_path=str(anno_file), transform=simple_transform, preload_maps=True, load_maps=False
        )

        # Should not crash, should log warning and continue
        assert len(dataset.maps_cache) == 0

    def test_corrupted_npz_file_handling(self, temp_dataset_structure, simple_transform):
        """Test that corrupted .npz files are handled gracefully."""
        tmpdir, images_dir, maps_dir, anno_file = temp_dataset_structure

        # Create a corrupted .npz file (just write some text)
        corrupted_file = maps_dir / "test.npz"
        with open(corrupted_file, "w") as f:
            f.write("This is not a valid npz file")

        # Create dataset with preload_maps=True
        dataset = OCRDataset(
            image_path=str(images_dir), annotation_path=str(anno_file), transform=simple_transform, preload_maps=True, load_maps=False
        )

        # Should handle corruption gracefully - corrupted file should not be in cache
        assert "test.jpg" not in dataset.maps_cache

    def test_partial_corruption_resilient_preloading(self, temp_dataset_structure, simple_transform):
        """Test that partial corruption doesn't prevent loading of valid maps."""
        tmpdir, images_dir, maps_dir, anno_file = temp_dataset_structure

        # Create a valid map for test.jpg
        prob_map = np.random.rand(1, 100, 100).astype(np.float32)
        thresh_map = np.random.rand(1, 100, 100).astype(np.float32)
        np.savez(maps_dir / "test.npz", prob_map=prob_map, thresh_map=thresh_map)

        # Create another image and corrupted map
        img2 = Image.new("RGB", (100, 100), color="black")
        img2.save(images_dir / "test2.jpg")

        # Corrupted map for test2.jpg
        corrupted_file = maps_dir / "test2.npz"
        with open(corrupted_file, "w") as f:
            f.write("corrupted")

        # Update annotations
        annotations = {
            "images": {
                "test.jpg": {"words": {"word1": {"points": [[10, 10], [20, 10], [20, 20], [10, 20]]}}},
                "test2.jpg": {"words": {"word2": {"points": [[30, 30], [40, 30], [40, 40], [30, 40]]}}},
            }
        }
        with open(anno_file, "w") as f:
            json.dump(annotations, f)

        # Create dataset with preload_maps=True
        dataset = OCRDataset(
            image_path=str(images_dir), annotation_path=str(anno_file), transform=simple_transform, preload_maps=True, load_maps=False
        )

        # Should load valid map but skip corrupted one
        assert "test.jpg" in dataset.maps_cache
        assert "test2.jpg" not in dataset.maps_cache
        assert len(dataset.maps_cache) == 1

    def test_missing_map_fallback_during_loading(self, temp_dataset_structure, simple_transform):
        """Test that missing maps during __getitem__ are handled gracefully."""
        tmpdir, images_dir, maps_dir, anno_file = temp_dataset_structure

        # Don't create any map files

        # Create dataset with load_maps=True
        dataset = OCRDataset(
            image_path=str(images_dir), annotation_path=str(anno_file), transform=simple_transform, preload_maps=False, load_maps=True
        )

        # Get item - should not crash even though map file is missing
        item = dataset[0]

        # Should not have prob_map or thresh_map keys
        assert "prob_map" not in item
        assert "thresh_map" not in item

    def test_corrupted_map_fallback_during_loading(self, temp_dataset_structure, simple_transform):
        """Test that corrupted maps during __getitem__ fallback gracefully."""
        tmpdir, images_dir, maps_dir, anno_file = temp_dataset_structure

        # Create corrupted map file
        corrupted_file = maps_dir / "test.npz"
        with open(corrupted_file, "w") as f:
            f.write("corrupted data")

        # Create dataset with load_maps=True
        dataset = OCRDataset(
            image_path=str(images_dir), annotation_path=str(anno_file), transform=simple_transform, preload_maps=False, load_maps=True
        )

        # Get item - should handle corruption gracefully
        item = dataset[0]

        # Should not have prob_map or thresh_map keys due to corruption
        assert "prob_map" not in item
        assert "thresh_map" not in item

    def test_valid_map_loading_from_disk(self, temp_dataset_structure, simple_transform):
        """Test successful loading of valid maps from disk."""
        tmpdir, images_dir, maps_dir, anno_file = temp_dataset_structure

        # Create valid map
        prob_map = np.random.rand(1, 100, 100).astype(np.float32)
        thresh_map = np.random.rand(1, 100, 100).astype(np.float32)
        np.savez(maps_dir / "test.npz", prob_map=prob_map, thresh_map=thresh_map)

        # Create dataset with load_maps=True
        dataset = OCRDataset(
            image_path=str(images_dir), annotation_path=str(anno_file), transform=simple_transform, preload_maps=False, load_maps=True
        )

        # Get item
        item = dataset[0]

        # Should have maps loaded
        assert "prob_map" in item
        assert "thresh_map" in item
        np.testing.assert_array_equal(item["prob_map"], prob_map)
        np.testing.assert_array_equal(item["thresh_map"], thresh_map)

    def test_preloaded_map_priority_over_disk(self, temp_dataset_structure, simple_transform):
        """Test that preloaded maps take priority over disk loading."""
        tmpdir, images_dir, maps_dir, anno_file = temp_dataset_structure

        # Create different maps on disk vs what we'll preload
        disk_prob_map = np.ones((1, 100, 100), dtype=np.float32)
        disk_thresh_map = np.ones((1, 100, 100), dtype=np.float32) * 2
        np.savez(maps_dir / "test.npz", prob_map=disk_prob_map, thresh_map=disk_thresh_map)

        # Create dataset and manually set cache to different values
        dataset = OCRDataset(
            image_path=str(images_dir), annotation_path=str(anno_file), transform=simple_transform, preload_maps=False, load_maps=True
        )

        # Manually set cache to different values
        cache_prob_map = np.zeros((1, 100, 100), dtype=np.float32)
        cache_thresh_map = np.zeros((1, 100, 100), dtype=np.float32)
        dataset.maps_cache["test.jpg"] = {"prob_map": cache_prob_map, "thresh_map": cache_thresh_map}

        # Get item - should use cached version, not disk version
        item = dataset[0]

        assert "prob_map" in item
        assert "thresh_map" in item
        np.testing.assert_array_equal(item["prob_map"], cache_prob_map)
        np.testing.assert_array_equal(item["thresh_map"], cache_thresh_map)

    @patch("ocr.datasets.base.np.load")
    def test_numpy_load_exception_handling(self, mock_np_load, temp_dataset_structure, simple_transform):
        """Test that numpy load exceptions are caught and handled."""
        tmpdir, images_dir, maps_dir, anno_file = temp_dataset_structure

        # Create valid map file
        prob_map = np.random.rand(1, 100, 100).astype(np.float32)
        thresh_map = np.random.rand(1, 100, 100).astype(np.float32)
        np.savez(maps_dir / "test.npz", prob_map=prob_map, thresh_map=thresh_map)

        # Mock np.load to raise an exception
        mock_np_load.side_effect = Exception("Mocked numpy load error")

        # Create dataset with load_maps=True
        dataset = OCRDataset(
            image_path=str(images_dir), annotation_path=str(anno_file), transform=simple_transform, preload_maps=False, load_maps=True
        )

        # Get item - should handle the exception gracefully
        item = dataset[0]

        # Should not have maps due to exception
        assert "prob_map" not in item
        assert "thresh_map" not in item

    def test_map_shape_validation_rejects_invalid_maps(self, temp_dataset_structure, simple_transform):
        """Test that maps with invalid shapes are rejected during loading."""
        tmpdir, images_dir, maps_dir, anno_file = temp_dataset_structure

        # Create invalid map (wrong shape - should be (1, H, W) but we'll make it (H, W))
        prob_map = np.random.rand(100, 100).astype(np.float32)  # Missing channel dimension
        thresh_map = np.random.rand(100, 100).astype(np.float32)
        np.savez(maps_dir / "test.npz", prob_map=prob_map, thresh_map=thresh_map)

        # Create dataset with load_maps=True
        dataset = OCRDataset(
            image_path=str(images_dir), annotation_path=str(anno_file), transform=simple_transform, preload_maps=False, load_maps=True
        )

        # Get item - should reject invalid map
        item = dataset[0]

        # Should not have prob_map or thresh_map keys due to invalid shape
        assert "prob_map" not in item
        assert "thresh_map" not in item

    def test_map_shape_validation_rejects_mismatched_dimensions(self, temp_dataset_structure, simple_transform):
        """Test that maps with dimensions not matching image are rejected."""
        tmpdir, images_dir, maps_dir, anno_file = temp_dataset_structure

        # Create map with wrong dimensions (image is 100x100, map is 50x50)
        prob_map = np.random.rand(1, 50, 50).astype(np.float32)
        thresh_map = np.random.rand(1, 50, 50).astype(np.float32)
        np.savez(maps_dir / "test.npz", prob_map=prob_map, thresh_map=thresh_map)

        # Create dataset with load_maps=True
        dataset = OCRDataset(
            image_path=str(images_dir), annotation_path=str(anno_file), transform=simple_transform, preload_maps=False, load_maps=True
        )

        # Get item - should reject map with wrong dimensions
        item = dataset[0]

        # Should not have prob_map or thresh_map keys due to dimension mismatch
        assert "prob_map" not in item
        assert "thresh_map" not in item

    def test_map_shape_validation_accepts_valid_maps(self, temp_dataset_structure, simple_transform):
        """Test that maps with correct shapes are accepted."""
        tmpdir, images_dir, maps_dir, anno_file = temp_dataset_structure

        # Create valid map (1, 100, 100) to match 100x100 image
        prob_map = np.random.rand(1, 100, 100).astype(np.float32)
        thresh_map = np.random.rand(1, 100, 100).astype(np.float32)
        np.savez(maps_dir / "test.npz", prob_map=prob_map, thresh_map=thresh_map)

        # Create dataset with load_maps=True
        dataset = OCRDataset(
            image_path=str(images_dir), annotation_path=str(anno_file), transform=simple_transform, preload_maps=False, load_maps=True
        )

        # Get item - should accept valid map
        item = dataset[0]

        # Should have maps
        assert "prob_map" in item
        assert "thresh_map" in item
        np.testing.assert_array_equal(item["prob_map"], prob_map)
        np.testing.assert_array_equal(item["thresh_map"], thresh_map)

    def test_cached_vs_generated_map_consistency(self, temp_dataset_structure, simple_transform):
        """Test that cached maps are numerically identical to freshly generated ones."""
        tmpdir, images_dir, maps_dir, anno_file = temp_dataset_structure

        # Create a sample with polygons
        polygons = [np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)]

        # Update annotations to include polygons
        annotations = {"images": {"test.jpg": {"words": {"word1": {"points": polygons[0].tolist()}}}}}
        with open(anno_file, "w") as f:
            json.dump(annotations, f)

        # Create dataset and get the sample
        dataset = OCRDataset(
            image_path=str(images_dir), annotation_path=str(anno_file), transform=simple_transform, preload_maps=False, load_maps=False
        )

        sample = dataset[0]

        # Generate maps using collate function
        from ocr.datasets.db_collate_fn import DBCollateFN

        collate_fn = DBCollateFN()
        # Remove batch dimension for make_prob_thresh_map
        image_tensor = sample["image"]  # Shape: (3, 100, 100)
        generated_maps = collate_fn.make_prob_thresh_map(image_tensor, [polygons[0]], "test.jpg")

        # Save generated maps to file
        prob_map = np.expand_dims(generated_maps["prob_map"], axis=0)  # Add channel dim
        thresh_map = np.expand_dims(generated_maps["thresh_map"], axis=0)
        np.savez(maps_dir / "test.npz", prob_map=prob_map, thresh_map=thresh_map)

        # Create new dataset that loads maps
        dataset_with_maps = OCRDataset(
            image_path=str(images_dir), annotation_path=str(anno_file), transform=simple_transform, preload_maps=False, load_maps=True
        )

        # Load the cached maps
        item = dataset_with_maps[0]

        # Compare - should be numerically identical
        assert "prob_map" in item
        assert "thresh_map" in item

        # Check shapes match
        assert item["prob_map"].shape == prob_map.shape
        assert item["thresh_map"].shape == thresh_map.shape

        # Check numerical accuracy (< 1% difference as per requirements)
        prob_diff = np.abs(item["prob_map"] - prob_map)
        thresh_diff = np.abs(item["thresh_map"] - thresh_map)

        assert np.max(prob_diff) < 0.01, f"Prob map difference too large: {np.max(prob_diff)}"
        assert np.max(thresh_diff) < 0.01, f"Thresh map difference too large: {np.max(thresh_diff)}"

        # Also check mean difference is very small
        assert np.mean(prob_diff) < 0.001, f"Prob map mean difference too large: {np.mean(prob_diff)}"
        assert np.mean(thresh_diff) < 0.001, f"Thresh map mean difference too large: {np.mean(thresh_diff)}"
