"""
Unit tests for polygon shape handling in DBCollateFN

Tests the polygon normalization logic that converts between different
polygon shape formats: (N, 2) ↔ (1, N, 2) for batch processing.

This addresses part of BUG-2025-004 to ensure robust polygon shape handling
in the collation process.
"""

import numpy as np
import pytest
import torch

from ocr.datasets.db_collate_fn import DBCollateFN


class TestDBCollatePolygonShapes:
    """Test cases for polygon shape normalization in DBCollateFN."""

    @pytest.fixture
    def collate_fn(self):
        """Create a DBCollateFN instance for testing."""
        return DBCollateFN()

    @pytest.fixture
    def sample_image_tensor(self):
        """Create a sample image tensor for testing."""
        return torch.randn(3, 224, 224)

    def test_polygon_normalization_2d_to_2d(self, collate_fn, sample_image_tensor):
        """Test that (N, 2) polygons remain unchanged."""
        # Use a triangle that definitely has positive area
        polygons_2d = [
            np.array([[10, 20], [30, 40], [20, 60]], dtype=np.float32),  # Triangle with area 300
        ]

        batch = [
            {
                "image": sample_image_tensor,
                "image_filename": "test1.jpg",
                "image_path": "/path/to/test1.jpg",
                "inverse_matrix": np.eye(3),
                "polygons": polygons_2d,
                "prob_map": np.zeros((224, 224), dtype=np.float32),
                "thresh_map": np.zeros((224, 224), dtype=np.float32),
            }
        ]

        result = collate_fn(batch)

        # Check that polygons are preserved
        assert len(result["polygons"]) == 1
        assert len(result["polygons"][0]) == 1  # Valid triangle should remain

        # Check shapes are still (N, 2)
        for poly in result["polygons"][0]:
            assert poly.ndim == 2
            assert poly.shape[1] == 2  # (x, y) coordinates

    def test_polygon_normalization_3d_to_2d(self, collate_fn, sample_image_tensor):
        """Test that (1, N, 2) polygons work correctly in map generation."""
        # Use a triangle with positive area
        polygons_3d = [
            np.array([[[10, 20], [30, 40], [20, 60]]], dtype=np.float32),  # Triangle with batch dim
        ]

        batch = [
            {
                "image": sample_image_tensor,
                "image_filename": "test1.jpg",
                "image_path": "/path/to/test1.jpg",
                "inverse_matrix": np.eye(3),
                "polygons": polygons_3d,
                "prob_map": np.zeros((224, 224), dtype=np.float32),
                "thresh_map": np.zeros((224, 224), dtype=np.float32),
            }
        ]

        result = collate_fn(batch)

        # Check that polygons are preserved as-is (collate doesn't modify them)
        assert len(result["polygons"]) == 1
        assert len(result["polygons"][0]) == 1  # Valid triangle should remain

        # Polygons retain their original shape in the result
        for poly in result["polygons"][0]:
            assert poly.ndim == 2  # Validation normalizes to (N, 2)
            assert poly.shape[1] == 2  # (x, y) coordinates
            # Should not have batch dimension after validation
            assert poly.shape[0] == 3  # Triangle has 3 points

        # But map generation should work correctly
        assert result["prob_maps"].shape == (1, 1, 224, 224)
        assert result["thresh_maps"].shape == (1, 1, 224, 224)

    def test_mixed_batch_polygon_shapes(self, collate_fn, sample_image_tensor):
        """Test batch with mixed polygon shapes: some (N,2), some (1,N,2)."""
        # First sample with (N, 2) polygons - use valid triangle
        polygons_2d = [
            np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
        ]

        # Second sample with (1, N, 2) polygons - use valid triangle
        polygons_3d = [
            np.array([[[60, 70], [80, 90], [100, 80]]], dtype=np.float32),
        ]

        batch = [
            {
                "image": sample_image_tensor,
                "image_filename": "test1.jpg",
                "image_path": "/path/to/test1.jpg",
                "inverse_matrix": np.eye(3),
                "polygons": polygons_2d,
                "prob_map": np.zeros((224, 224), dtype=np.float32),
                "thresh_map": np.zeros((224, 224), dtype=np.float32),
            },
            {
                "image": sample_image_tensor,
                "image_filename": "test2.jpg",
                "image_path": "/path/to/test2.jpg",
                "inverse_matrix": np.eye(3),
                "polygons": polygons_3d,
                "prob_map": np.zeros((224, 224), dtype=np.float32),
                "thresh_map": np.zeros((224, 224), dtype=np.float32),
            },
        ]

        result = collate_fn(batch)

        # Check that both samples are processed
        assert len(result["polygons"]) == 2

        # Check that all polygons are normalized to (N, 2) after validation
        for sample_polygons in result["polygons"]:
            for poly in sample_polygons:
                assert poly.ndim == 2
                assert poly.shape[1] == 2  # (x, y) coordinates

        # Maps should be batched correctly
        assert result["prob_maps"].shape == (2, 1, 224, 224)
        assert result["thresh_maps"].shape == (2, 1, 224, 224)

    def test_empty_polygons_handling(self, collate_fn, sample_image_tensor):
        """Test handling of empty polygon lists."""
        batch = [
            {
                "image": sample_image_tensor,
                "image_filename": "test1.jpg",
                "image_path": "/path/to/test1.jpg",
                "inverse_matrix": np.eye(3),
                "polygons": [],  # Empty polygons
                "prob_map": np.zeros((224, 224), dtype=np.float32),
                "thresh_map": np.zeros((224, 224), dtype=np.float32),
            }
        ]

        result = collate_fn(batch)

        # Should handle empty polygons gracefully
        assert len(result["polygons"]) == 1
        assert result["polygons"][0] == []

    def test_single_point_polygons(self, collate_fn, sample_image_tensor):
        """Test handling of degenerate polygons (single point)."""
        polygons_single_point = [
            np.array([[50, 50]], dtype=np.float32),  # Single point - should be filtered
        ]

        batch = [
            {
                "image": sample_image_tensor,
                "image_filename": "test1.jpg",
                "image_path": "/path/to/test1.jpg",
                "inverse_matrix": np.eye(3),
                "polygons": polygons_single_point,
                "prob_map": np.zeros((224, 224), dtype=np.float32),
                "thresh_map": np.zeros((224, 224), dtype=np.float32),
            }
        ]

        result = collate_fn(batch)

        # Should filter out single point polygons
        assert len(result["polygons"]) == 1
        assert len(result["polygons"][0]) == 0  # Single point should be filtered out

    def test_polygon_shape_preservation_in_prob_map_generation(self, collate_fn, sample_image_tensor):
        """Test that polygon shapes are correctly handled during prob/thresh map generation."""
        # Test with (1, N, 2) polygons that need normalization
        polygons_3d = [
            np.array([[[10, 10], [110, 10], [110, 110], [10, 110]]], dtype=np.float32),  # Rectangle
        ]

        batch = [
            {
                "image": sample_image_tensor,
                "image_filename": "test1.jpg",
                "image_path": "/path/to/test1.jpg",
                "inverse_matrix": np.eye(3),
                "polygons": polygons_3d,
                # Don't provide pre-computed maps to force generation
            }
        ]

        result = collate_fn(batch)

        # Should generate prob and thresh maps
        assert "prob_maps" in result
        assert "thresh_maps" in result
        assert result["prob_maps"].shape[0] == 1  # Batch size
        assert result["thresh_maps"].shape[0] == 1  # Batch size

        # Maps should have correct spatial dimensions (batched)
        _, h, w = sample_image_tensor.shape
        assert result["prob_maps"].shape == (1, 1, h, w)
        assert result["thresh_maps"].shape == (1, 1, h, w)

    def test_polygon_normalization_with_precomputed_maps(self, collate_fn, sample_image_tensor):
        """Test that polygon normalization works when pre-computed maps are provided."""
        # Use a valid triangle
        polygons_3d = [
            np.array([[[10, 10], [110, 10], [60, 110]]], dtype=np.float32),  # Triangle
        ]

        batch = [
            {
                "image": sample_image_tensor,
                "image_filename": "test1.jpg",
                "image_path": "/path/to/test1.jpg",
                "inverse_matrix": np.eye(3),
                "polygons": polygons_3d,
                "prob_map": np.zeros((224, 224), dtype=np.float32),
                "thresh_map": np.zeros((224, 224), dtype=np.float32),
            }
        ]

        result = collate_fn(batch)

        # Should use pre-computed maps
        assert "prob_maps" in result
        assert "thresh_maps" in result
        assert result["prob_maps"].shape == (1, 1, 224, 224)
        assert result["thresh_maps"].shape == (1, 1, 224, 224)

        # Polygons should be normalized to (N, 2) after validation
        assert len(result["polygons"]) == 1
        assert len(result["polygons"][0]) == 1
        assert result["polygons"][0][0].ndim == 2  # Normalized to (N, 2)
        assert result["polygons"][0][0].shape == (3, 2)  # Triangle has 3 points
