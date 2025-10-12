"""
Unit tests for polygon filtering logic in OCRDataset

Tests the _filter_degenerate_polygons method that removes invalid polygons
based on size, point count, and span criteria.

This addresses Phase 1.3.1 in the data pipeline testing plan.
"""

from unittest.mock import patch

import numpy as np

from ocr.utils.polygon_utils import filter_degenerate_polygons


class TestPolygonFiltering:
    """Test cases for polygon filtering in OCRDataset."""

    def test_filter_valid_polygons(self):
        """Test that valid polygons are preserved."""
        # Valid triangle
        valid_triangle = np.array([[10, 20], [30, 40], [20, 60]], dtype=np.float32)

        # Valid quadrilateral
        valid_quad = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

        polygons = [valid_triangle, valid_quad]

        filtered = filter_degenerate_polygons(polygons)

        assert len(filtered) == 2
        np.testing.assert_array_equal(filtered[0], valid_triangle)
        np.testing.assert_array_equal(filtered[1], valid_quad)

    def test_filter_none_polygons(self):
        """Test filtering of None polygons."""
        polygons = [None, None, None]

        with patch("ocr.datasets.base.logging.getLogger") as mock_logger:
            mock_logger.return_value.isEnabledFor.return_value = True
            filtered = filter_degenerate_polygons(polygons)

        assert len(filtered) == 0

    def test_filter_empty_polygons(self):
        """Test filtering of empty polygons."""
        empty_poly = np.array([], dtype=np.float32)
        polygons = [empty_poly]

        with patch("ocr.datasets.base.logging.getLogger") as mock_logger:
            mock_logger.return_value.isEnabledFor.return_value = True
            filtered = filter_degenerate_polygons(polygons)

        assert len(filtered) == 0

    def test_filter_too_few_points(self):
        """Test filtering of polygons with too few points."""
        # Single point
        single_point = np.array([[10, 20]], dtype=np.float32)

        # Two points (line)
        two_points = np.array([[10, 20], [30, 40]], dtype=np.float32)

        polygons = [single_point, two_points]

        with patch("ocr.datasets.base.logging.getLogger") as mock_logger:
            mock_logger.return_value.isEnabledFor.return_value = True
            filtered = filter_degenerate_polygons(polygons)

        assert len(filtered) == 0

    def test_filter_too_small_polygons(self):
        """Test filtering of polygons smaller than min_side threshold."""
        # Very small triangle (all points very close)
        small_triangle = np.array([[10, 10], [10.5, 10], [10, 10.5]], dtype=np.float32)

        # Small quadrilateral
        small_quad = np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]], dtype=np.float32)

        polygons = [small_triangle, small_quad]

        with patch("ocr.datasets.base.logging.getLogger") as mock_logger:
            mock_logger.return_value.isEnabledFor.return_value = True
            filtered = filter_degenerate_polygons(polygons, min_side=2.0)

        assert len(filtered) == 0

    def test_filter_zero_span_polygons(self):
        """Test filtering of polygons with zero span after rounding."""
        # Horizontal line (zero height span)
        horizontal_line = np.array([[0, 10], [10, 10], [20, 10]], dtype=np.float32)

        # Vertical line (zero width span)
        vertical_line = np.array([[10, 0], [10, 10], [10, 20]], dtype=np.float32)

        polygons = [horizontal_line, vertical_line]

        with patch("ocr.datasets.base.logging.getLogger") as mock_logger:
            mock_logger.return_value.isEnabledFor.return_value = True
            filtered = filter_degenerate_polygons(polygons)

        assert len(filtered) == 0

    def test_filter_mixed_valid_invalid(self):
        """Test filtering with mix of valid and invalid polygons."""
        # Valid triangle
        valid = np.array([[0, 0], [100, 0], [50, 100]], dtype=np.float32)

        # Invalid: too small
        too_small = np.array([[10, 10], [10.1, 10], [10, 10.1]], dtype=np.float32)

        # Invalid: None
        none_poly = None

        polygons = [valid, too_small, none_poly]

        with patch("ocr.datasets.base.logging.getLogger") as mock_logger:
            mock_logger.return_value.isEnabledFor.return_value = True
            filtered = filter_degenerate_polygons(polygons, min_side=5.0)

        assert len(filtered) == 1
        np.testing.assert_array_equal(filtered[0], valid)

    def test_filter_different_shapes(self):
        """Test filtering with different polygon array shapes."""
        # (N, 2) shape
        shape_n2 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)

        # (1, N, 2) shape - should be handled
        shape_1n2 = np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=np.float32)

        polygons = [shape_n2, shape_1n2]

        with patch("ocr.datasets.base.logging.getLogger") as mock_logger:
            mock_logger.return_value.isEnabledFor.return_value = True
            filtered = filter_degenerate_polygons(polygons)

        assert len(filtered) == 2

    def test_filter_min_side_parameter(self):
        """Test that min_side parameter controls filtering threshold."""
        # Polygon with span of 5
        medium_poly = np.array([[0, 0], [5, 0], [5, 5], [0, 5]], dtype=np.float32)

        # Test with min_side = 10 (should be filtered)
        with patch("ocr.datasets.base.logging.getLogger") as mock_logger:
            mock_logger.return_value.isEnabledFor.return_value = True
            filtered_strict = filter_degenerate_polygons([medium_poly], min_side=10.0)

        assert len(filtered_strict) == 0

        # Test with min_side = 2 (should be kept)
        with patch("ocr.datasets.base.logging.getLogger") as mock_logger:
            mock_logger.return_value.isEnabledFor.return_value = True
            filtered_lenient = filter_degenerate_polygons([medium_poly], min_side=2.0)

        assert len(filtered_lenient) == 1

    def test_filter_logging(self):
        """Test that filtering statistics are logged correctly."""
        polygons = [
            None,  # none
            np.array([], dtype=np.float32),  # empty
            np.array([[10, 10]], dtype=np.float32),  # too_few_points
            np.array([[10, 10], [10.1, 10], [10, 10.1]], dtype=np.float32),  # too_small
            np.array([[0, 10], [10, 10], [20, 10]], dtype=np.float32),  # zero_span
            np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32),  # valid
        ]

        with patch("ocr.datasets.base.logging.getLogger") as mock_logger:
            mock_logger_instance = mock_logger.return_value
            mock_logger_instance.isEnabledFor.return_value = True

            filtered = filter_degenerate_polygons(polygons, min_side=5.0)

            # Should have logged the filtering statistics
            mock_logger_instance.info.assert_called_once()
            log_call_args = mock_logger_instance.info.call_args[0]

            # Check that the log message contains expected counts
            log_message = log_call_args[0]
            assert "Filtered %d degenerate polygons" in log_message
            assert "too_few_points=%d" in log_message
            assert "too_small=%d" in log_message
            assert "zero_span=%d" in log_message
            assert "empty=%d" in log_message
            assert "none=%d" in log_message

        assert len(filtered) == 1  # Only the valid polygon should remain

    def test_filter_no_logging_when_disabled(self):
        """Test that no logging occurs when logging level is below INFO."""
        polygons = [None, np.array([[10, 10]], dtype=np.float32)]

        with patch("ocr.datasets.base.logging.getLogger") as mock_logger:
            mock_logger_instance = mock_logger.return_value
            mock_logger_instance.isEnabledFor.return_value = False  # Logging disabled

            filtered = filter_degenerate_polygons(polygons)

            # Should not have logged anything
            mock_logger_instance.info.assert_not_called()

        assert len(filtered) == 0
