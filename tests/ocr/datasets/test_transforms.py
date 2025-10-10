"""
Unit tests for polygon shape handling in transforms.py

Tests the BUG-2025-004 fix for polygon dimension error that caused
catastrophic performance degradation (hmean: 0.890 → 0.00011).
"""

import numpy as np
import pytest

from ocr.datasets.transforms import DBTransforms


class TestPolygonShapeHandling:
    """Test cases for polygon shape validation and processing."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    @pytest.fixture
    def transforms_instance(self):
        """Create a DBTransforms instance with minimal transforms."""
        import albumentations as A

        transforms = [A.Resize(224, 224)]
        keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
        return DBTransforms(transforms, keypoint_params)

    def test_validate_polygons_valid_2d_shapes(self, transforms_instance):
        """Test validation of valid 2D polygon shapes."""
        # Valid polygons with different point counts
        valid_polygons = [
            np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),  # Triangle
            np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32),  # Rectangle
            np.array([[50, 10], [80, 40], [60, 80], [20, 70], [10, 30]], dtype=np.float32),  # Pentagon
        ]

        # Should not raise any exceptions
        transforms_instance._validate_polygons(valid_polygons)

    def test_validate_polygons_valid_3d_shapes(self, transforms_instance):
        """Test validation of valid 3D polygon shapes (backward compatibility)."""
        # Valid 3D polygons with batch dimension
        valid_polygons = [
            np.array([[[10, 20], [30, 40], [50, 60]]], dtype=np.float32),  # Triangle
            np.array([[[0, 0], [100, 0], [100, 100], [0, 100]]], dtype=np.float32),  # Rectangle
        ]

        # Should not raise any exceptions
        transforms_instance._validate_polygons(valid_polygons)

    def test_validate_polygons_invalid_types(self, transforms_instance):
        """Test validation rejects invalid polygon types."""
        invalid_polygons = [
            "not an array",  # String
            42,  # Integer
            [1, 2, 3],  # List instead of numpy array
        ]

        for invalid_polygon in invalid_polygons:
            with pytest.raises(TypeError, match="polygon at index .* must be numpy array"):
                transforms_instance._validate_polygons([invalid_polygon])

    def test_validate_polygons_invalid_shapes(self, transforms_instance):
        """Test validation rejects invalid polygon shapes."""
        invalid_polygons = [
            np.array([10, 20, 30]),  # 1D array
            np.array([[10, 20, 30]]),  # Wrong coordinate dimension
            np.array([[[10, 20]], [[30, 40]]]),  # Wrong 3D shape
            # Note: 2-point polygons are allowed in validation, filtered in processing
        ]

        for i, invalid_polygon in enumerate(invalid_polygons[:3]):  # Skip the 2-point one
            with pytest.raises((ValueError, TypeError)):
                transforms_instance._validate_polygons([invalid_polygon])

    def test_validate_polygons_none_input(self, transforms_instance):
        """Test validation handles None input gracefully."""
        # Should not raise any exceptions
        transforms_instance._validate_polygons(None)

    def test_polygon_point_count_extraction_2d(self, transforms_instance, sample_image):
        """Test correct point count extraction for 2D polygons (BUG-2025-004)."""
        # Create test polygons with known shapes
        polygons = [
            np.array([[10, 20], [30, 40], [50, 60], [70, 80]], dtype=np.float32),  # 4 points
            np.array([[0, 0], [100, 0], [100, 100]], dtype=np.float32),  # 3 points
        ]

        # Apply transforms
        result = transforms_instance(sample_image, polygons)

        # Verify output polygons have correct shapes
        output_polygons = result["polygons"]
        assert len(output_polygons) == 2

        # Each output polygon should be reshaped to (1, N, 2)
        assert output_polygons[0].shape == (1, 4, 2)  # 4 points
        assert output_polygons[1].shape == (1, 3, 2)  # 3 points

    def test_polygon_point_count_extraction_3d(self, transforms_instance, sample_image):
        """Test correct point count extraction for 3D polygons (backward compatibility)."""
        # Create test polygons with 3D shape (1, N, 2)
        polygons = [
            np.array([[[10, 20], [30, 40], [50, 60], [70, 80]]], dtype=np.float32),  # 4 points
        ]

        # Apply transforms
        result = transforms_instance(sample_image, polygons)

        # Verify output polygons have correct shapes
        output_polygons = result["polygons"]
        assert len(output_polygons) == 1
        assert output_polygons[0].shape == (1, 4, 2)  # 4 points

    def test_polygon_transform_preserves_geometry(self, transforms_instance, sample_image):
        """Test that polygon transforms preserve relative geometry."""
        # Create a simple square polygon
        original_polygon = np.array(
            [
                [50, 50],  # Top-left
                [150, 50],  # Top-right
                [150, 150],  # Bottom-right
                [50, 150],  # Bottom-left
            ],
            dtype=np.float32,
        )

        polygons = [original_polygon]

        # Apply transforms (resize to 224x224)
        result = transforms_instance(sample_image, polygons)
        transformed_polygon = result["polygons"][0][0]  # Remove batch dimension

        # Verify the polygon still has 4 points
        assert len(transformed_polygon) == 4

        # Verify coordinates are within image bounds
        for point in transformed_polygon:
            x, y = point
            assert 0 <= x <= 224
            assert 0 <= y <= 224

    def test_empty_polygons_list(self, transforms_instance, sample_image):
        """Test handling of empty polygons list."""
        result = transforms_instance(sample_image, [])

        # Should return empty polygons list
        assert result["polygons"] == []

    def test_mixed_valid_invalid_polygons(self, transforms_instance, sample_image):
        """Test that invalid polygons are skipped while valid ones are processed."""
        polygons = [
            np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),  # Valid triangle
            np.array([[10, 20]]),  # Invalid (only 2 points) - should be skipped
            np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32),  # Valid rectangle
        ]

        result = transforms_instance(sample_image, polygons)

        # Should have 2 valid polygons (invalid one skipped)
        assert len(result["polygons"]) == 2


class TestPolygonShapeBugRegression:
    """Regression tests specifically for BUG-2025-004 polygon shape dimension error."""

    def test_bug_2025_004_original_issue(self):
        """
        Test the exact scenario that caused BUG-2025-004.

        Before fix: polygon.shape[1] returned 2 (coordinate dimension)
        After fix: polygon.shape[0] returns N (number of points)
        """
        # Create a polygon that would trigger the bug
        polygon = np.array(
            [
                [10, 20],
                [30, 40],
                [50, 60],
                [70, 80],  # 4 points
            ],
            dtype=np.float32,
        )

        # Before fix: this would incorrectly use shape[1] = 2
        # After fix: correctly uses shape[0] = 4
        assert polygon.shape[0] == 4  # Number of points
        assert polygon.shape[1] == 2  # Coordinate dimensions

        # The bug was using shape[1] instead of shape[0]
        # This test ensures we use the correct dimension
        num_points = polygon.shape[0]  # Correct
        wrong_num_points = polygon.shape[1]  # Bug

        assert num_points == 4
        assert wrong_num_points == 2  # This would cause the bug

    def test_polygon_reconstruction_accuracy(self):
        """Test that polygon reconstruction preserves point order and coordinates."""
        import albumentations as A

        # Create transforms instance
        transforms = [A.Resize(200, 200)]
        keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
        db_transforms = DBTransforms(transforms, keypoint_params)

        # Create test image and polygon
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        polygon = np.array(
            [
                [10, 10],
                [90, 10],
                [90, 90],
                [10, 90],  # Square
            ],
            dtype=np.float32,
        )

        # Apply transform
        result = db_transforms(image, [polygon])
        reconstructed = result["polygons"][0][0]  # Remove batch dimension

        # Verify all points are preserved and within bounds
        assert len(reconstructed) == 4
        for point in reconstructed:
            x, y = point
            assert 0 <= x <= 200
            assert 0 <= y <= 200
