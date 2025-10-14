"""
Tests for Advanced Corner Detection module.
"""

import cv2
import numpy as np
import pytest

from ocr.datasets.preprocessing.advanced_corner_detection import (
    AdvancedCornerDetector,
    CornerDetectionConfig,
    CornerDetectionMethod,
    validate_document_corners,
)


class TestAdvancedCornerDetector:
    @pytest.fixture
    def sample_document_image(self):
        """Create a synthetic document image for testing."""
        # Create a white rectangle on black background (simple document)
        img = np.zeros((200, 300), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 150), 255, -1)
        return img

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing."""
        config = CornerDetectionConfig(method=CornerDetectionMethod.COMBINED, harris_threshold=0.01, shi_tomasi_max_corners=50)
        return AdvancedCornerDetector(config)

    def test_detect_corners_harris(self, sample_document_image, detector):
        """Test Harris corner detection."""
        detector.config.method = CornerDetectionMethod.HARRIS
        result = detector.detect_corners(sample_document_image)

        assert isinstance(result.corners, np.ndarray)
        assert result.method == "harris"
        assert result.confidence >= 0.0
        assert result.subpixel_refined

    def test_detect_corners_shi_tomasi(self, sample_document_image, detector):
        """Test Shi-Tomasi corner detection."""
        detector.config.method = CornerDetectionMethod.SHI_TOMASI
        result = detector.detect_corners(sample_document_image)

        assert isinstance(result.corners, np.ndarray)
        assert result.method == "shi_tomasi"
        assert result.confidence >= 0.0

    def test_detect_corners_combined(self, sample_document_image, detector):
        """Test combined corner detection."""
        detector.config.method = CornerDetectionMethod.COMBINED
        result = detector.detect_corners(sample_document_image)

        assert isinstance(result.corners, np.ndarray)
        assert result.method == "combined"
        assert result.confidence >= 0.0
        assert result.subpixel_refined

    def test_subpixel_refinement(self, detector):
        """Test sub-pixel corner refinement."""
        # Create image with known corner at non-integer position
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 80), 255, -1)

        result = detector.detect_corners(img)

        # Check that corners are refined (may not be exactly integer)
        if len(result.corners) > 0:
            assert result.subpixel_refined

    def test_empty_image_handling(self, detector):
        """Test handling of images with no detectable corners."""
        img = np.zeros((100, 100), dtype=np.uint8)

        result = detector.detect_corners(img)

        assert len(result.corners) == 0
        assert result.confidence == 0.0
        assert not result.subpixel_refined


class TestCornerValidation:
    def test_validate_document_corners_valid(self):
        """Test validation of valid document corners."""
        # Create corners for a valid rectangle
        corners = np.array(
            [
                [10, 10],  # top-left
                [90, 10],  # top-right
                [90, 90],  # bottom-right
                [10, 90],  # bottom-left
            ]
        )
        image_shape = (100, 100, 3)

        assert validate_document_corners(corners, image_shape)

    def test_validate_document_corners_too_few(self):
        """Test validation fails with too few corners."""
        corners = np.array([[50, 50]])  # Only one corner
        image_shape = (100, 100, 3)

        assert not validate_document_corners(corners, image_shape)

    def test_validate_document_corners_out_of_bounds(self):
        """Test validation fails with out-of-bounds corners."""
        corners = np.array(
            [
                [-10, 10],  # Out of bounds
                [90, 10],
                [90, 90],
                [10, 90],
            ]
        )
        image_shape = (100, 100, 3)

        assert not validate_document_corners(corners, image_shape)

    def test_validate_document_corners_clustered(self):
        """Test validation fails with clustered corners."""
        # All corners too close together
        corners = np.array([[45, 45], [50, 45], [45, 50], [50, 50]])
        image_shape = (100, 100, 3)

        assert not validate_document_corners(corners, image_shape)
