"""
Tests for Document Flattening module.

Comprehensive test suite for document flattening functionality
including thin plate spline warping, surface normal estimation,
and quality assessment.
"""

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from ocr.datasets.preprocessing.document_flattening import (
    DocumentFlattener,
    FlatteningConfig,
    FlatteningMethod,
    FlatteningQualityMetrics,
    FlatteningResult,
    SurfaceNormals,
    WarpingTransform,
    flatten_crumpled_document,
)


class TestFlatteningConfig:
    """Test configuration validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FlatteningConfig()

        assert config.method == FlatteningMethod.THIN_PLATE_SPLINE
        assert config.grid_size == 20
        assert 0.0 <= config.smoothing_factor <= 1.0
        assert config.enable_quality_assessment is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = FlatteningConfig(method=FlatteningMethod.CYLINDRICAL, grid_size=30, smoothing_factor=0.2, edge_preservation_strength=0.9)

        assert config.method == FlatteningMethod.CYLINDRICAL
        assert config.grid_size == 30
        assert config.smoothing_factor == 0.2
        assert config.edge_preservation_strength == 0.9

    def test_grid_size_validation(self):
        """Test grid size validation."""
        # Valid grid sizes
        config = FlatteningConfig(grid_size=10)
        assert config.grid_size == 10

        # Invalid grid size (too small) - Pydantic ValidationError
        with pytest.raises(ValidationError):
            FlatteningConfig(grid_size=3)

        # Invalid grid size (too large) - Pydantic ValidationError
        with pytest.raises(ValidationError):
            FlatteningConfig(grid_size=150)

    def test_smoothing_factor_bounds(self):
        """Test smoothing factor is bounded."""
        # Valid smoothing factors
        config1 = FlatteningConfig(smoothing_factor=0.0)
        assert config1.smoothing_factor == 0.0

        config2 = FlatteningConfig(smoothing_factor=1.0)
        assert config2.smoothing_factor == 1.0

        # Invalid smoothing factor
        with pytest.raises(ValueError):
            FlatteningConfig(smoothing_factor=1.5)


class TestSurfaceNormals:
    """Test surface normal estimation data model."""

    def test_surface_normals_creation(self):
        """Test creating surface normals object."""
        normals = np.zeros((20, 20, 3))
        grid_points = np.zeros((20, 20, 2))
        curvature_map = np.zeros((20, 20))

        surface = SurfaceNormals(
            normals=normals, grid_points=grid_points, curvature_map=curvature_map, mean_curvature=0.1, max_curvature=0.3
        )

        assert surface.normals.shape == (20, 20, 3)
        assert surface.mean_curvature == 0.1
        assert surface.max_curvature == 0.3

    def test_normals_shape_validation(self):
        """Test surface normals shape validation."""
        grid_points = np.zeros((20, 20, 2))
        curvature_map = np.zeros((20, 20))

        # Invalid normal shape (missing 3rd dimension)
        with pytest.raises(ValueError, match="must be shape"):
            SurfaceNormals(
                normals=np.zeros((20, 20)),  # Missing 3rd dimension
                grid_points=grid_points,
                curvature_map=curvature_map,
                mean_curvature=0.1,
                max_curvature=0.3,
            )


class TestWarpingTransform:
    """Test warping transform data model."""

    def test_warping_transform_creation(self):
        """Test creating warping transform."""
        source_points = np.array([[0, 0], [10, 10], [20, 20]])
        target_points = np.array([[0, 0], [10, 12], [20, 24]])

        transform = WarpingTransform(
            source_points=source_points, target_points=target_points, method=FlatteningMethod.THIN_PLATE_SPLINE, confidence=0.85
        )

        assert transform.source_points.shape == (3, 2)
        assert transform.target_points.shape == (3, 2)
        assert transform.confidence == 0.85

    def test_points_shape_validation(self):
        """Test control points shape validation."""
        # Invalid points shape
        with pytest.raises(ValueError, match="must be shape"):
            WarpingTransform(
                source_points=np.array([0, 0]),  # Wrong shape
                target_points=np.array([[0, 0]]),
                method=FlatteningMethod.THIN_PLATE_SPLINE,
                confidence=0.8,
            )


class TestFlatteningQualityMetrics:
    """Test quality metrics data model."""

    def test_quality_metrics_creation(self):
        """Test creating quality metrics."""
        metrics = FlatteningQualityMetrics(
            distortion_score=0.15,
            edge_preservation_score=0.85,
            smoothness_score=0.90,
            overall_quality=0.87,
            residual_curvature=0.05,
            processing_successful=True,
        )

        assert metrics.distortion_score == 0.15
        assert metrics.overall_quality == 0.87
        assert metrics.processing_successful is True

    def test_metrics_bounds_validation(self):
        """Test metrics are bounded between 0 and 1."""
        # Invalid distortion score (> 1.0)
        with pytest.raises(ValueError):
            FlatteningQualityMetrics(
                distortion_score=1.5,
                edge_preservation_score=0.8,
                smoothness_score=0.9,
                overall_quality=0.8,
                residual_curvature=0.1,
                processing_successful=True,
            )


class TestFlatteningResult:
    """Test flattening result data model."""

    def test_flattening_result_creation(self):
        """Test creating flattening result."""
        flattened_image = np.zeros((100, 100, 3), dtype=np.uint8)
        transform = WarpingTransform(
            source_points=np.array([[0, 0]]), target_points=np.array([[0, 0]]), method=FlatteningMethod.THIN_PLATE_SPLINE, confidence=0.9
        )

        result = FlatteningResult(
            flattened_image=flattened_image,
            warping_transform=transform,
            method_used=FlatteningMethod.THIN_PLATE_SPLINE,
            processing_time_ms=50.0,
        )

        assert result.flattened_image.shape == (100, 100, 3)
        assert result.method_used == FlatteningMethod.THIN_PLATE_SPLINE
        assert result.processing_time_ms == 50.0

    def test_empty_image_validation(self):
        """Test empty image validation."""
        transform = WarpingTransform(
            source_points=np.array([[0, 0]]), target_points=np.array([[0, 0]]), method=FlatteningMethod.THIN_PLATE_SPLINE, confidence=0.9
        )

        # Empty image should fail validation
        with pytest.raises(ValueError, match="cannot be empty"):
            FlatteningResult(
                flattened_image=np.array([]),
                warping_transform=transform,
                method_used=FlatteningMethod.THIN_PLATE_SPLINE,
                processing_time_ms=50.0,
            )


class TestDocumentFlattener:
    """Test document flattening operations."""

    @pytest.fixture
    def flat_document_image(self) -> np.ndarray:
        """Create a flat document image for testing."""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        # Add some text/content
        cv2.rectangle(img, (50, 50), (250, 150), (0, 0, 0), 2)
        cv2.putText(img, "Test Document", (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return img

    @pytest.fixture
    def crumpled_document_image(self) -> np.ndarray:
        """Create a simulated crumpled document image."""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255

        # Add some content
        cv2.rectangle(img, (50, 50), (250, 150), (0, 0, 0), 2)
        cv2.putText(img, "Crumpled", (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Simulate crumpling with sinusoidal distortion
        h, w = img.shape[:2]
        for y in range(h):
            for x in range(w):
                # Add wave distortion
                offset_x = int(10 * np.sin(y * 0.1))
                offset_y = int(10 * np.sin(x * 0.1))

                new_x = min(max(x + offset_x, 0), w - 1)
                new_y = min(max(y + offset_y, 0), h - 1)

                if new_x != x or new_y != y:
                    img[y, x] = img[new_y, new_x]

        return img

    @pytest.fixture
    def flattener(self) -> DocumentFlattener:
        """Create flattener instance for testing."""
        config = FlatteningConfig(grid_size=15, smoothing_factor=0.1, enable_quality_assessment=True)
        return DocumentFlattener(config)

    def test_flattener_initialization(self, flattener):
        """Test flattener initializes correctly."""
        assert isinstance(flattener.config, FlatteningConfig)
        assert flattener.config.grid_size == 15

    def test_flatten_flat_document(self, flattener, flat_document_image):
        """Test flattening already flat document."""
        result = flattener.flatten_document(flat_document_image)

        assert isinstance(result, FlatteningResult)
        assert result.flattened_image.shape == flat_document_image.shape
        assert result.processing_time_ms > 0
        assert result.surface_normals is not None

    def test_flatten_crumpled_document(self, flattener, crumpled_document_image):
        """Test flattening crumpled document."""
        result = flattener.flatten_document(crumpled_document_image)

        assert isinstance(result, FlatteningResult)
        assert result.flattened_image.shape == crumpled_document_image.shape
        assert result.method_used == FlatteningMethod.THIN_PLATE_SPLINE
        assert result.processing_time_ms > 0

    def test_surface_normal_estimation(self, flattener, crumpled_document_image):
        """Test surface normal estimation."""
        surface_normals = flattener._estimate_surface_normals(crumpled_document_image)

        assert isinstance(surface_normals, SurfaceNormals)
        assert surface_normals.normals.shape[2] == 3  # 3D normals
        assert 0.0 <= surface_normals.mean_curvature
        assert surface_normals.max_curvature >= surface_normals.mean_curvature

    def test_thin_plate_spline_warping(self, flattener, crumpled_document_image):
        """Test thin plate spline warping."""
        surface_normals = flattener._estimate_surface_normals(crumpled_document_image)

        warped, transform = flattener._thin_plate_spline_warping(crumpled_document_image, surface_normals, corners=None)

        assert warped.shape == crumpled_document_image.shape
        assert isinstance(transform, WarpingTransform)
        assert transform.method == FlatteningMethod.THIN_PLATE_SPLINE
        assert 0.0 <= transform.confidence <= 1.0

    def test_cylindrical_warping(self, flattener, crumpled_document_image):
        """Test cylindrical warping."""
        surface_normals = flattener._estimate_surface_normals(crumpled_document_image)

        warped, transform = flattener._cylindrical_warping(crumpled_document_image, surface_normals)

        assert warped.shape == crumpled_document_image.shape
        assert transform.method == FlatteningMethod.CYLINDRICAL

    def test_spherical_warping(self, flattener, crumpled_document_image):
        """Test spherical warping."""
        surface_normals = flattener._estimate_surface_normals(crumpled_document_image)

        warped, transform = flattener._spherical_warping(crumpled_document_image, surface_normals)

        assert warped.shape == crumpled_document_image.shape
        assert transform.method == FlatteningMethod.SPHERICAL

    def test_adaptive_warping(self, flattener, crumpled_document_image):
        """Test adaptive warping."""
        surface_normals = flattener._estimate_surface_normals(crumpled_document_image)

        warped, transform = flattener._adaptive_warping(crumpled_document_image, surface_normals, corners=None)

        assert warped.shape == crumpled_document_image.shape
        assert isinstance(transform, WarpingTransform)

    def test_quality_assessment(self, flattener, crumpled_document_image):
        """Test flattening quality assessment."""
        # Flatten document
        result = flattener.flatten_document(crumpled_document_image)

        # Should have quality metrics if enabled
        assert result.quality_metrics is not None
        metrics = result.quality_metrics

        assert isinstance(metrics, FlatteningQualityMetrics)
        assert 0.0 <= metrics.distortion_score <= 1.0
        assert 0.0 <= metrics.edge_preservation_score <= 1.0
        assert 0.0 <= metrics.smoothness_score <= 1.0
        assert 0.0 <= metrics.overall_quality <= 1.0

    def test_different_flattening_methods(self, crumpled_document_image):
        """Test different flattening methods."""
        methods = [FlatteningMethod.THIN_PLATE_SPLINE, FlatteningMethod.CYLINDRICAL, FlatteningMethod.SPHERICAL, FlatteningMethod.ADAPTIVE]

        for method in methods:
            config = FlatteningConfig(method=method, grid_size=10)
            flattener = DocumentFlattener(config)

            result = flattener.flatten_document(crumpled_document_image)

            assert isinstance(result, FlatteningResult)
            assert result.method_used == method

    def test_grayscale_image_handling(self, flattener):
        """Test flattening grayscale images."""
        gray_img = np.ones((200, 300), dtype=np.uint8) * 200
        cv2.rectangle(gray_img, (50, 50), (250, 150), 0, 2)

        result = flattener.flatten_document(gray_img)

        assert isinstance(result, FlatteningResult)
        assert len(result.flattened_image.shape) in [2, 3]

    def test_min_curvature_threshold(self):
        """Test minimum curvature threshold."""
        # Create very flat image
        flat_img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        config = FlatteningConfig(min_curvature_threshold=0.5)
        flattener = DocumentFlattener(config)

        result = flattener.flatten_document(flat_img)

        # Should skip flattening for flat images
        assert "skipped_reason" in result.metadata
        assert result.metadata["skipped_reason"] == "already_flat"

    def test_rbf_warping_fallback(self, flattener):
        """Test RBF warping handles edge cases."""
        # Create minimal point set
        source_points = np.array([[0, 0], [10, 10]])
        target_points = np.array([[0, 0], [10, 10]])

        img = np.ones((50, 50, 3), dtype=np.uint8) * 255

        warped = flattener._apply_rbf_warping(img, source_points, target_points)

        # Should return image (possibly original if RBF fails)
        assert warped.shape == img.shape


class TestConvenienceFunction:
    """Test convenience function."""

    def test_flatten_crumpled_document_function(self):
        """Test convenience function for flattening."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (20, 20), (80, 80), (0, 0, 0), 2)

        result = flatten_crumpled_document(img)

        assert isinstance(result, FlatteningResult)
        assert result.flattened_image.shape == img.shape

    def test_convenience_function_with_config(self):
        """Test convenience function with custom config."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        config = FlatteningConfig(method=FlatteningMethod.CYLINDRICAL, grid_size=10)

        result = flatten_crumpled_document(img, config=config)

        assert result.method_used == FlatteningMethod.CYLINDRICAL

    def test_convenience_function_with_corners(self):
        """Test convenience function with document corners."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        corners = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])

        result = flatten_crumpled_document(img, corners=corners)

        assert isinstance(result, FlatteningResult)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_small_image(self):
        """Test flattening very small images."""
        tiny_img = np.ones((10, 10, 3), dtype=np.uint8) * 255

        config = FlatteningConfig(grid_size=5)
        flattener = DocumentFlattener(config)

        result = flattener.flatten_document(tiny_img)

        assert isinstance(result, FlatteningResult)

    def test_non_square_image(self):
        """Test flattening non-square images."""
        tall_img = np.ones((300, 100, 3), dtype=np.uint8) * 255
        wide_img = np.ones((100, 300, 3), dtype=np.uint8) * 255

        flattener = DocumentFlattener()

        result1 = flattener.flatten_document(tall_img)
        result2 = flattener.flatten_document(wide_img)

        assert result1.flattened_image.shape == tall_img.shape
        assert result2.flattened_image.shape == wide_img.shape

    def test_quality_assessment_disabled(self):
        """Test flattening with quality assessment disabled."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        config = FlatteningConfig(enable_quality_assessment=False)
        flattener = DocumentFlattener(config)

        result = flattener.flatten_document(img)

        assert result.quality_metrics is None


class TestIntegration:
    """Integration tests for document flattening."""

    def test_full_flattening_pipeline(self):
        """Test complete flattening pipeline."""
        # Create document with known distortion
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (50, 50), (250, 150), (0, 0, 0), 2)

        # Apply sinusoidal distortion
        h, w = img.shape[:2]
        distorted = img.copy()
        for y in range(h):
            shift = int(15 * np.sin(y * 0.05))
            distorted[y, :] = np.roll(img[y, :], shift, axis=0)

        # Flatten with different methods
        for method in FlatteningMethod:
            config = FlatteningConfig(method=method, grid_size=15, enable_quality_assessment=True)
            flattener = DocumentFlattener(config)

            result = flattener.flatten_document(distorted)

            # Verify all outputs
            assert isinstance(result, FlatteningResult)
            assert result.flattened_image.shape == distorted.shape
            assert result.method_used == method
            assert result.processing_time_ms > 0
            assert result.surface_normals is not None

            if result.quality_metrics is not None:
                assert result.quality_metrics.processing_successful is not None

    def test_multiple_consecutive_flattenings(self):
        """Test applying flattening multiple times."""
        img = np.ones((150, 200, 3), dtype=np.uint8) * 255

        flattener = DocumentFlattener()

        # First flattening
        result1 = flattener.flatten_document(img)

        # Second flattening on already flattened image
        result2 = flattener.flatten_document(result1.flattened_image)

        # Both should succeed
        assert isinstance(result1, FlatteningResult)
        assert isinstance(result2, FlatteningResult)

        # Second should have lower curvature
        if result2.surface_normals:
            if result1.surface_normals:
                assert result2.surface_normals.mean_curvature <= result1.surface_normals.mean_curvature
