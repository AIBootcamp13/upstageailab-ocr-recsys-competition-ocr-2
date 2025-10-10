"""
Unit tests for transform pipeline data contracts

Tests the geometric transformation contracts to ensure keypoint/polygon
preservation and inverse transform matrix accuracy across the DBTransforms pipeline.

This addresses Phase 2.1 of the Data Pipeline Testing Implementation Roadmap.
"""

import numpy as np
import pytest
import torch

from ocr.datasets.transforms import DBTransforms


class TestTransformPipelineContracts:
    """Test cases for transform pipeline data contracts and geometric accuracy."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        return np.random.randint(0, 255, (320, 240, 3), dtype=np.uint8)

    @pytest.fixture
    def transforms_instance(self):
        """Create a DBTransforms instance with resize transform."""
        import albumentations as A

        transforms = [A.Resize(224, 224)]
        keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
        return DBTransforms(transforms, keypoint_params)

    @pytest.fixture
    def identity_transforms_instance(self):
        """Create a DBTransforms instance with identity transform (no changes)."""
        import albumentations as A

        transforms = []  # No transforms = identity
        keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
        return DBTransforms(transforms, keypoint_params)

    def test_keypoint_preservation_identity_transform(self, identity_transforms_instance, sample_image):
        """Test that keypoints are preserved through identity transform."""
        # Create test polygons
        polygons = [
            np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),  # Triangle
            np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32),  # Rectangle
        ]

        result = identity_transforms_instance(sample_image, polygons)

        # Check output structure
        assert "image" in result
        assert "polygons" in result
        assert "inverse_matrix" in result

        # Check that polygons are preserved
        assert len(result["polygons"]) == 2

        # Check polygon shapes (should be (1, N, 2))
        for poly in result["polygons"]:
            assert poly.ndim == 3
            assert poly.shape[0] == 1  # Batch dimension
            assert poly.shape[2] == 2  # (x, y) coordinates

        # Check that keypoints are approximately preserved (identity transform)
        np.testing.assert_array_almost_equal(result["polygons"][0][0], polygons[0], decimal=1)
        np.testing.assert_array_almost_equal(result["polygons"][1][0], polygons[1], decimal=1)

    def test_inverse_transform_matrix_accuracy(self, transforms_instance, sample_image):
        """Test that inverse transform matrix provides reasonable coordinate mapping."""
        # Create a polygon in the original image space
        original_polygon = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float32)

        result = transforms_instance(sample_image, [original_polygon])

        # Get the transformed polygon and inverse matrix
        transformed_polygon = result["polygons"][0][0]  # Remove batch dim
        inverse_matrix = result["inverse_matrix"]

        # Test that inverse matrix is a valid 3x3 transformation matrix
        assert inverse_matrix.shape == (3, 3)
        assert inverse_matrix[2, 2] == 1.0  # Homogeneous coordinate preservation

        # Test that applying inverse matrix to transformed points gives reasonable coordinates
        # Add homogeneous coordinates to transformed points
        homogeneous_points = np.column_stack([transformed_polygon, np.ones(len(transformed_polygon))])
        # Apply inverse matrix
        original_space_points = homogeneous_points @ inverse_matrix.T
        # Remove homogeneous coordinate
        original_space_points = original_space_points[:, :2]

        # The result should be within reasonable bounds (not negative, not extremely large)
        assert np.all(original_space_points >= -50)  # Allow some margin for transform effects
        assert np.all(original_space_points <= 400)  # Allow some margin for transform effects

        # Test that the matrix has reasonable scale factors (should scale up from 224 to ~240-320)
        scale_x = inverse_matrix[0, 0]
        scale_y = inverse_matrix[1, 1]
        assert 1.0 < scale_x < 2.0  # Should scale up from 224 to larger size
        assert 1.0 < scale_y < 2.0

    def test_input_output_shape_consistency(self, transforms_instance, sample_image):
        """Test that input and output shapes follow expected contracts."""
        polygons = [
            np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
            np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32),
        ]

        result = transforms_instance(sample_image, polygons)

        # Check image shape (should be tensor with shape C, H, W)
        assert result["image"].dim() == 3  # PyTorch tensor
        assert result["image"].shape[0] == 3  # RGB channels
        assert result["image"].shape[1:] == (224, 224)  # Resized dimensions

        # Check polygons shape (should be list of (1, N, 2) arrays)
        assert isinstance(result["polygons"], list)
        assert len(result["polygons"]) == len(polygons)

        for poly in result["polygons"]:
            assert isinstance(poly, np.ndarray)
            assert poly.ndim == 3
            assert poly.shape[0] == 1  # Batch dimension
            assert poly.shape[2] == 2  # (x, y) coordinates
            assert poly.shape[1] >= 3  # At least 3 points for valid polygon

        # Check inverse matrix shape
        assert isinstance(result["inverse_matrix"], np.ndarray)
        assert result["inverse_matrix"].shape == (3, 3)

    def test_geometric_transform_mathematical_correctness(self, transforms_instance):
        """Test that geometric transformations are mathematically correct."""
        # Create a simple square polygon
        original_square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

        # Create image that contains this square
        image = np.full((200, 200, 3), 128, dtype=np.uint8)

        result = transforms_instance(image, [original_square])

        transformed_square = result["polygons"][0][0]
        inverse_matrix = result["inverse_matrix"]

        # Verify that applying forward transform then inverse gives original
        # This is a more rigorous test of the transform pipeline

        # Simulate forward transform on original points
        # (In practice, this would be done by albumentations, but we test the inverse)

        # Apply inverse to transformed points
        homogeneous_transformed = np.column_stack([transformed_square, np.ones(len(transformed_square))])
        restored = homogeneous_transformed @ inverse_matrix.T
        restored = restored[:, :2]

        # Should match original square (within tolerance)
        np.testing.assert_array_almost_equal(restored, original_square, decimal=1)

    def test_empty_polygons_handling(self, transforms_instance, sample_image):
        """Test handling of empty polygon lists."""
        result = transforms_instance(sample_image, [])

        assert "polygons" in result
        assert result["polygons"] == []

    def test_none_polygons_handling(self, transforms_instance, sample_image):
        """Test handling of None polygons."""
        result = transforms_instance(sample_image, None)

        assert "polygons" in result
        assert result["polygons"] == []

    def test_polygon_with_minimum_points(self, transforms_instance, sample_image):
        """Test polygon with minimum valid points (triangle)."""
        triangle = np.array([[10, 10], [20, 30], [30, 10]], dtype=np.float32)

        result = transforms_instance(sample_image, [triangle])

        assert len(result["polygons"]) == 1
        assert result["polygons"][0].shape == (1, 3, 2)  # 3 points, (x,y) each

    def test_polygon_clamping_to_image_bounds(self, transforms_instance):
        """Test that polygons are clamped to image boundaries."""
        # Create polygon that extends beyond image bounds
        out_of_bounds_polygon = np.array(
            [
                [-10, -10],  # Outside bounds
                [350, -10],  # Way outside width
                [350, 250],  # Outside bounds
                [-10, 250],  # Outside bounds
            ],
            dtype=np.float32,
        )

        # Create image that should contain the clamped polygon
        image = np.full((240, 320, 3), 128, dtype=np.uint8)

        result = transforms_instance(image, [out_of_bounds_polygon])

        # Polygon should be clamped to valid coordinates
        transformed_polygon = result["polygons"][0][0]

        # All coordinates should be within bounds [0, 223] after resize
        # Allow small floating point tolerance
        assert np.all(transformed_polygon >= 0)
        assert np.all(transformed_polygon <= 224.0)  # Allow slight overflow due to floating point precision

    def test_transform_contract_validation_error_messages(self, transforms_instance, sample_image):
        """Test that contract violations produce clear error messages."""
        # Test invalid polygon type
        with pytest.raises(TypeError, match="polygons must be a list"):
            transforms_instance(sample_image, "not a list")

        # Test invalid polygon element type
        with pytest.raises(TypeError, match="polygon at index 0 must be numpy array"):
            transforms_instance(sample_image, ["not an array"])

        # Test invalid polygon shape (wrong dimensions)
        with pytest.raises(ValueError, match="polygon at index 0 must be 2D or 3D array"):
            transforms_instance(sample_image, [np.array([1, 2, 3, 4])])  # 1D array

        # Test invalid polygon shape (wrong coordinate dimensions)
        with pytest.raises(ValueError, match="polygon at index 0 must have shape \\(N, 2\\)"):
            transforms_instance(sample_image, [np.array([[1, 2, 3], [4, 5, 6]])])  # (N, 3) instead of (N, 2)

    def test_shape_contract_validation_catches_violations(self, transforms_instance):
        """Test that shape contract validation catches input and output violations."""
        # Test invalid image dimensions
        with pytest.raises(ValueError, match="Image must be 2D or 3D array"):
            transforms_instance(np.array([1, 2, 3]), None)  # 1D array

        # Test invalid image channels
        invalid_image = np.random.randint(0, 255, (32, 32, 5), dtype=np.uint8)  # 5 channels
        with pytest.raises(ValueError, match="Image must have 1 or 3 channels"):
            transforms_instance(invalid_image, None)

        # Test that valid inputs pass validation
        valid_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = transforms_instance(valid_image, None)

        # Verify output contracts are enforced
        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].ndim == 3
        assert isinstance(result["polygons"], list)
        assert isinstance(result["inverse_matrix"], np.ndarray)
        assert result["inverse_matrix"].shape == (3, 3)

    def test_transform_pipeline_end_to_end_consistency(self, transforms_instance):
        """Test end-to-end consistency of the transform pipeline."""
        # Create test data
        image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        polygons = [
            np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float32),
            np.array([[200, 100], [250, 120], [240, 180]], dtype=np.float32),
        ]

        # Apply transforms
        result = transforms_instance(image, polygons)

        # Verify all expected outputs are present
        required_keys = ["image", "polygons", "inverse_matrix"]
        for key in required_keys:
            assert key in result

        # Verify output types
        assert hasattr(result["image"], "shape")  # Tensor-like
        assert isinstance(result["polygons"], list)
        assert isinstance(result["inverse_matrix"], np.ndarray)

        # Verify polygon count is preserved (or reduced due to clamping/filtering)
        assert len(result["polygons"]) <= len(polygons)

        # Verify inverse matrix is valid (determinant != 0)
        det = np.linalg.det(result["inverse_matrix"])
        assert abs(det) > 1e-6, "Inverse matrix should be invertible"

    def test_transform_pipeline_end_to_end_with_realistic_data(self, transforms_instance, sample_image):
        """Test that transform pipeline maintains consistency end-to-end with realistic data."""
        # Create realistic polygon data
        polygons = [
            np.array([[10, 20], [30, 20], [30, 40], [10, 40]], dtype=np.float32),  # Rectangle
            np.array([[50, 60], [70, 50], [90, 70], [60, 80]], dtype=np.float32),  # Irregular quad
        ]

        result = transforms_instance(sample_image, polygons)

        # Verify all expected outputs are present
        assert "image" in result
        assert "polygons" in result
        assert "inverse_matrix" in result

        # Verify output types and shapes
        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].shape[0] == 3  # Channels first
        assert result["image"].shape[1] == 224  # Resized height
        assert result["image"].shape[2] == 224  # Resized width

        # Verify polygons are transformed
        assert len(result["polygons"]) == len(polygons)
        for polygon in result["polygons"]:
            assert polygon.shape[0] == 1  # Batch dimension
            assert polygon.shape[2] == 2  # (x, y) coordinates
            assert polygon.shape[1] >= 3  # At least 3 points for valid polygon

        # Verify inverse matrix is valid transformation matrix
        inv_matrix = result["inverse_matrix"]
        assert inv_matrix.shape == (3, 3)
        assert np.allclose(inv_matrix[2, :], [0, 0, 1])  # Homogeneous coordinates

    def test_end_to_end_with_real_dataset_sample(self):
        """Test transform pipeline with a realistic dataset sample."""
        import albumentations as A

        # Create transforms similar to what would be used in training
        transforms = [A.Rotate(limit=10, p=0.5), A.GaussianBlur(blur_limit=3, p=0.1), A.Resize(224, 224)]
        keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
        transforms_instance = DBTransforms(transforms, keypoint_params)

        # Create a more realistic image and polygons
        image = np.random.randint(0, 255, (320, 480, 3), dtype=np.uint8)

        # Simulate text detection polygons (typical OCR scenario)
        polygons = [
            # Horizontal text line
            np.array([[50, 100], [200, 100], [200, 130], [50, 130]], dtype=np.float32),
            # Vertical text line
            np.array([[250, 50], [280, 50], [280, 150], [250, 150]], dtype=np.float32),
            # Irregular text region
            np.array([[100, 200], [180, 190], [190, 220], [80, 230]], dtype=np.float32),
        ]

        # Apply transforms
        result = transforms_instance(image, polygons)

        # Verify the pipeline worked correctly
        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].shape == (3, 224, 224)

        # All polygons should be preserved (possibly transformed)
        assert len(result["polygons"]) == len(polygons)

        # Each polygon should maintain valid shape
        for polygon in result["polygons"]:
            assert polygon.ndim == 3
            assert polygon.shape[0] == 1  # Batch dimension
            assert polygon.shape[2] == 2  # Coordinate dimension
            assert polygon.shape[1] >= 3  # At least triangle

        # Verify inverse matrix is valid transformation matrix
        inv_matrix = result["inverse_matrix"]
        assert inv_matrix.shape == (3, 3)
        assert np.allclose(inv_matrix[2, :], [0, 0, 1])  # Homogeneous coordinates

    def test_transform_pipeline_with_edge_cases(self):
        """Test transform pipeline with edge cases and boundary conditions."""
        import albumentations as A

        transforms = [A.Resize(224, 224)]
        keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
        transforms_instance = DBTransforms(transforms, keypoint_params)

        # Test with very small image
        small_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        polygons = [np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float32)]

        result = transforms_instance(small_image, polygons)
        assert result["image"].shape == (3, 224, 224)
        assert len(result["polygons"]) == 1

        # Test with large image
        large_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        large_polygons = [np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)]

        result = transforms_instance(large_image, large_polygons)
        assert result["image"].shape == (3, 224, 224)
        assert len(result["polygons"]) == 1

        # Test with grayscale image (should be handled)
        gray_image = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
        result = transforms_instance(gray_image, None)
        assert result["image"].shape[0] == 1  # Single channel
        assert result["image"].shape[1:] == (224, 224)

    def test_performance_regression_basic_transform(self, transforms_instance, sample_image):
        """Test that basic transform operations meet performance expectations."""
        import time

        polygons = [
            np.array([[10, 20], [30, 20], [30, 40], [10, 40]], dtype=np.float32),
            np.array([[50, 60], [70, 50], [90, 70], [60, 80]], dtype=np.float32),
        ]

        # Measure performance over multiple runs
        times = []
        num_runs = 10

        for _ in range(num_runs):
            start_time = time.time()
            result = transforms_instance(sample_image, polygons)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        max_time = max(times)

        # Performance expectations (these are reasonable bounds for the test environment)
        # Adjust these based on your system's capabilities
        assert avg_time < 0.1  # Average transform should take less than 100ms
        assert max_time < 0.2  # No single transform should take more than 200ms

        # Verify result is still correct
        assert isinstance(result["image"], torch.Tensor)
        assert len(result["polygons"]) == 2

    def test_performance_regression_memory_usage(self, transforms_instance, sample_image):
        """Test that transform operations don't have memory leaks or excessive usage."""
        import os

        import psutil

        polygons = [np.array([[10, 20], [30, 20], [30, 40], [10, 40]], dtype=np.float32)]

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run multiple transforms
        for i in range(50):
            result = transforms_instance(sample_image, polygons)
            # Verify result integrity
            assert result["image"] is not None
            assert len(result["polygons"]) == 1

        # Check memory usage after transforms
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB for 50 transforms)
        assert memory_increase < 50.0, f"Memory increased by {memory_increase:.1f}MB, which seems excessive"

    def test_performance_regression_batch_processing(self):
        """Test performance of batch processing vs individual processing."""
        import time

        import albumentations as A

        transforms = [A.Resize(224, 224)]
        keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
        transforms_instance = DBTransforms(transforms, keypoint_params)

        # Create batch of images and polygons
        batch_size = 8
        images = [np.random.randint(0, 255, (320, 240, 3), dtype=np.uint8) for _ in range(batch_size)]
        polygons_batch = [[np.array([[10, 20], [30, 20], [30, 40], [10, 40]], dtype=np.float32)] for _ in range(batch_size)]

        # Measure individual processing time
        individual_times = []
        for img, polys in zip(images, polygons_batch, strict=True):
            start_time = time.time()
            result = transforms_instance(img, polys)
            end_time = time.time()
            individual_times.append(end_time - start_time)

        total_individual_time = sum(individual_times)

        # Measure batch processing time (simulated by processing sequentially)
        # In a real scenario, you'd have a batched transform, but here we test sequential
        batch_start_time = time.time()
        batch_results = []
        for img, polys in zip(images, polygons_batch, strict=True):
            result = transforms_instance(img, polys)
            batch_results.append(result)
        batch_end_time = time.time()
        total_batch_time = batch_end_time - batch_start_time

        # Batch processing should not be significantly slower than individual
        # (allowing for some overhead)
        assert total_batch_time < total_individual_time * 1.5

        # All results should be valid
        assert len(batch_results) == batch_size
        for result in batch_results:
            assert isinstance(result["image"], torch.Tensor)
            assert result["image"].shape == (3, 224, 224)

    def test_performance_regression_large_polygons(self):
        """Test performance with large numbers of polygons (stress test)."""
        import time

        import albumentations as A

        transforms = [A.Resize(224, 224)]
        keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
        transforms_instance = DBTransforms(transforms, keypoint_params)

        image = np.random.randint(0, 255, (320, 240, 3), dtype=np.uint8)

        # Create many polygons (stress test)
        num_polygons = 50
        polygons = []
        for i in range(num_polygons):
            # Create polygons at different positions
            x_offset = (i % 10) * 20
            y_offset = (i // 10) * 20
            polygon = np.array(
                [
                    [10 + x_offset, 10 + y_offset],
                    [25 + x_offset, 10 + y_offset],
                    [25 + x_offset, 25 + y_offset],
                    [10 + x_offset, 25 + y_offset],
                ],
                dtype=np.float32,
            )
            polygons.append(polygon)

        # Measure performance
        start_time = time.time()
        result = transforms_instance(image, polygons)
        end_time = time.time()
        processing_time = end_time - start_time

        # Should process 50 polygons reasonably fast
        assert processing_time < 1.0  # Less than 1 second for 50 polygons

        # All polygons should be preserved
        assert len(result["polygons"]) == num_polygons

        # Each polygon should be valid
        for polygon in result["polygons"]:
            assert polygon.shape[0] == 1  # Batch dimension
            assert polygon.shape[1] == 4  # 4 points
            assert polygon.shape[2] == 2  # (x, y) coordinates
