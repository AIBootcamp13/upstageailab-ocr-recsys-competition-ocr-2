"""
Tests for Geometric Document Modeling module.
"""

import numpy as np
import pytest

from ocr.datasets.preprocessing.geometric_document_modeling import (
    GeometricDocumentModeler,
    GeometricModel,
    GeometricModelConfig,
    validate_document_geometry,
)


class TestGeometricDocumentModeler:
    @pytest.fixture
    def sample_rectangle_corners(self):
        """Create corners for a perfect rectangle."""
        return np.array(
            [
                [10, 10],  # top-left
                [90, 10],  # top-right
                [90, 90],  # bottom-right
                [10, 90],  # bottom-left
            ],
            dtype=np.float32,
        )

    @pytest.fixture
    def sample_quadrilateral_corners(self):
        """Create corners for a quadrilateral (not rectangle)."""
        return np.array(
            [
                [10, 10],  # top-left
                [85, 15],  # top-right (slightly skewed)
                [90, 85],  # bottom-right
                [15, 90],  # bottom-left
            ],
            dtype=np.float32,
        )

    @pytest.fixture
    def modeler(self):
        """Create modeler instance for testing."""
        config = GeometricModelConfig(model_type=GeometricModel.QUADRILATERAL, ransac_iterations=50, confidence_threshold=0.7)
        return GeometricDocumentModeler(config)

    def test_fit_quadrilateral_perfect_rectangle(self, sample_rectangle_corners, modeler):
        """Test fitting quadrilateral to perfect rectangle."""
        geometry = modeler.fit_document_geometry(sample_rectangle_corners)

        assert geometry is not None
        assert geometry.model_type == "quadrilateral"
        assert geometry.confidence > 0.7
        assert geometry.is_rectangle
        assert len(geometry.corners) == 4

    def test_fit_quadrilateral_skewed(self, sample_quadrilateral_corners, modeler):
        """Test fitting quadrilateral to skewed quadrilateral."""
        geometry = modeler.fit_document_geometry(sample_quadrilateral_corners)

        assert geometry is not None
        assert geometry.model_type == "quadrilateral"
        assert geometry.confidence > 0.5
        assert len(geometry.corners) == 4

    def test_fit_rectangle_model(self, sample_rectangle_corners):
        """Test rectangle-specific fitting."""
        config = GeometricModelConfig(model_type=GeometricModel.RECTANGLE, confidence_threshold=0.8)
        modeler = GeometricDocumentModeler(config)

        geometry = modeler.fit_document_geometry(sample_rectangle_corners)

        assert geometry is not None
        assert geometry.model_type == "rectangle"
        assert geometry.is_rectangle

    def test_insufficient_points(self, modeler):
        """Test handling of insufficient corner points."""
        few_corners = np.array([[10, 10], [20, 20]])

        geometry = modeler.fit_document_geometry(few_corners)

        assert geometry is None

    def test_calculate_polygon_area(self, modeler):
        """Test polygon area calculation."""
        # Rectangle 80x80 = 6400 area
        points = np.array([[0, 0], [80, 0], [80, 80], [0, 80]])

        area = modeler._calculate_polygon_area(points)
        assert abs(area - 6400) < 1

    def test_calculate_aspect_ratio(self, modeler):
        """Test aspect ratio calculation."""
        # Rectangle 80x40 = aspect ratio 2.0
        points = np.array([[0, 0], [80, 0], [80, 40], [0, 40]])

        ratio = modeler._calculate_aspect_ratio(points)
        assert abs(ratio - 2.0) < 0.1

    def test_is_rectangle_perfect(self, modeler):
        """Test rectangle detection on perfect rectangle."""
        perfect_rect = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])

        assert modeler._is_rectangle(perfect_rect)

    def test_is_rectangle_skewed(self, modeler):
        """Test rectangle detection fails on skewed quadrilateral."""
        skewed = np.array(
            [
                [0, 0],
                [12, 2],  # Not at 90 degrees
                [10, 12],
                [0, 10],
            ]
        )

        assert not modeler._is_rectangle(skewed)


class TestGeometryValidation:
    def test_validate_geometry_valid(self):
        """Test validation of valid document geometry."""
        from ocr.datasets.preprocessing.geometric_document_modeling import DocumentGeometry

        geometry = DocumentGeometry(
            corners=np.array([[10, 10], [90, 10], [90, 90], [10, 90]]),
            confidence=0.9,
            model_type="quadrilateral",
            area=6400,
            aspect_ratio=1.0,
            is_rectangle=True,
        )

        image_shape = (100, 100, 3)
        assert validate_document_geometry(geometry, image_shape)

    def test_validate_geometry_too_small(self):
        """Test validation fails for too small geometry."""
        from ocr.datasets.preprocessing.geometric_document_modeling import DocumentGeometry

        geometry = DocumentGeometry(
            corners=np.array([[45, 45], [55, 45], [55, 55], [45, 55]]),
            confidence=0.9,
            model_type="quadrilateral",
            area=100,  # Too small
            aspect_ratio=1.0,
            is_rectangle=True,
        )

        image_shape = (100, 100, 3)
        assert not validate_document_geometry(geometry, image_shape)

    def test_validate_geometry_out_of_bounds(self):
        """Test validation fails for out-of-bounds geometry."""
        from ocr.datasets.preprocessing.geometric_document_modeling import DocumentGeometry

        geometry = DocumentGeometry(
            corners=np.array(
                [
                    [-10, 10],  # Out of bounds
                    [90, 10],
                    [90, 90],
                    [10, 90],
                ]
            ),
            confidence=0.9,
            model_type="quadrilateral",
            area=6400,
            aspect_ratio=1.0,
            is_rectangle=True,
        )

        image_shape = (100, 100, 3)
        assert not validate_document_geometry(geometry, image_shape)
