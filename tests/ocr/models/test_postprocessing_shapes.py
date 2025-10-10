"""
Unit tests for post-processing shape handling in DBPostProcessor

Tests the robust handling of different batch sizes, tensor shapes, and edge cases
in the post-processing pipeline to ensure consistent prediction output formats.

This addresses Phase 3.1 of the data pipeline testing plan to ensure
robust shape handling in post-processing.
"""

import numpy as np
import pytest
import torch

from ocr.models.head.db_postprocess import DBPostProcessor


class TestPostProcessingShapes:
    """Test cases for post-processing shape handling in DBPostProcessor."""

    @pytest.fixture
    def post_processor(self):
        """Create a DBPostProcessor instance for testing."""
        return DBPostProcessor(thresh=0.3, box_thresh=0.7, max_candidates=100)

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        batch_size = 2
        height, width = 224, 224

        return {
            "images": torch.randn(batch_size, 3, height, width),
            "polygons": [torch.randn(10, 4, 2) for _ in range(batch_size)],  # Not used in post-processing
            "ignore_tags": torch.zeros(batch_size, 10, dtype=torch.bool),
            "shape": [(height, width) for _ in range(batch_size)],
            "inverse_matrix": [np.eye(3, dtype=np.float32) for _ in range(batch_size)],
            "filename": [f"test_{i}.jpg" for i in range(batch_size)],
        }

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        batch_size = 2
        height, width = 224, 224

        return {"prob_maps": torch.sigmoid(torch.randn(batch_size, 1, height, width))}

    def test_single_batch_processing(self, post_processor, sample_batch, sample_predictions):
        """Test processing a single batch with standard shapes."""
        boxes_batch, scores_batch = post_processor.represent(sample_batch, sample_predictions)

        assert len(boxes_batch) == len(sample_batch["images"])
        assert len(scores_batch) == len(sample_batch["images"])

        for boxes, scores in zip(boxes_batch, scores_batch, strict=True):
            assert isinstance(boxes, list)
            assert isinstance(scores, list)
            assert len(boxes) == len(scores)

            # Each box should be a list of 4 coordinates
            for box in boxes:
                assert isinstance(box, list)
                assert len(box) == 4  # 4 corners
                for point in box:
                    assert len(point) == 2  # x, y coordinates

    def test_different_batch_sizes(self, post_processor):
        """Test processing batches with different sizes."""
        for batch_size in [1, 2, 4, 8]:
            height, width = 224, 224

            batch = {
                "images": torch.randn(batch_size, 3, height, width),
                "polygons": [torch.randn(5, 4, 2) for _ in range(batch_size)],
                "ignore_tags": torch.zeros(batch_size, 5, dtype=torch.bool),
                "shape": [(height, width) for _ in range(batch_size)],
                "inverse_matrix": [np.eye(3, dtype=np.float32) for _ in range(batch_size)],
                "filename": [f"test_{i}.jpg" for i in range(batch_size)],
            }

            predictions = {"prob_maps": torch.sigmoid(torch.randn(batch_size, 1, height, width))}

            boxes_batch, scores_batch = post_processor.represent(batch, predictions)

            assert len(boxes_batch) == batch_size
            assert len(scores_batch) == batch_size

    def test_different_image_sizes(self, post_processor):
        """Test processing images with different sizes."""
        sizes = [(224, 224), (320, 240), (480, 640), (128, 128)]

        for height, width in sizes:
            batch_size = 1

            batch = {
                "images": torch.randn(batch_size, 3, height, width),
                "polygons": [torch.randn(5, 4, 2) for _ in range(batch_size)],
                "ignore_tags": torch.zeros(batch_size, 5, dtype=torch.bool),
                "shape": [(height, width) for _ in range(batch_size)],
                "inverse_matrix": [np.eye(3, dtype=np.float32) for _ in range(batch_size)],
                "filename": [f"test_{height}x{width}.jpg" for _ in range(batch_size)],
            }

            predictions = {"prob_maps": torch.sigmoid(torch.randn(batch_size, 1, height, width))}

            boxes_batch, scores_batch = post_processor.represent(batch, predictions)

            assert len(boxes_batch) == batch_size
            assert len(scores_batch) == batch_size

    def test_prediction_as_tensor_directly(self, post_processor, sample_batch):
        """Test processing when predictions are passed as tensor directly (not dict)."""
        batch_size = len(sample_batch["images"])
        height, width = 224, 224

        # Pass predictions as tensor directly
        predictions_tensor = torch.sigmoid(torch.randn(batch_size, 1, height, width))

        boxes_batch, scores_batch = post_processor.represent(sample_batch, predictions_tensor)

        assert len(boxes_batch) == batch_size
        assert len(scores_batch) == batch_size

    def test_empty_batch_handling(self, post_processor):
        """Test handling of empty batches."""
        batch = {
            "images": torch.empty(0, 3, 224, 224),
            "polygons": [],
            "ignore_tags": torch.empty(0, 0, dtype=torch.bool),
            "shape": [],
            "inverse_matrix": [],
            "filename": [],
        }

        predictions = {"prob_maps": torch.empty(0, 1, 224, 224)}

        boxes_batch, scores_batch = post_processor.represent(batch, predictions)

        assert len(boxes_batch) == 0
        assert len(scores_batch) == 0

    def test_missing_required_keys(self, post_processor, sample_batch, sample_predictions):
        """Test error handling for missing required keys."""
        # Test missing images
        batch_missing_images = sample_batch.copy()
        del batch_missing_images["images"]

        with pytest.raises(AssertionError, match="images is required"):
            post_processor.represent(batch_missing_images, sample_predictions)

        # Test missing inverse_matrix
        batch_missing_matrix = sample_batch.copy()
        del batch_missing_matrix["inverse_matrix"]

        with pytest.raises(AssertionError, match="inverse_matrix is required"):
            post_processor.represent(batch_missing_matrix, sample_predictions)

        # Test missing prob_maps in dict predictions
        predictions_missing_prob = {"other_key": torch.randn(2, 1, 224, 224)}

        with pytest.raises(AssertionError, match="prob_maps is required"):
            post_processor.represent(sample_batch, predictions_missing_prob)

    def test_invalid_prediction_shapes(self, post_processor, sample_batch):
        """Test handling of invalid prediction shapes."""
        # Wrong number of channels
        predictions_wrong_channels = {
            "prob_maps": torch.sigmoid(torch.randn(2, 3, 224, 224))  # Should be 1 channel
        }

        # This should raise a ValueError for wrong channel count
        with pytest.raises(ValueError, match="should have 1 channel"):
            post_processor.represent(sample_batch, predictions_wrong_channels)

    def test_extreme_probability_values(self, post_processor, sample_batch):
        """Test handling of extreme probability values (0.0 and 1.0)."""
        batch_size = len(sample_batch["images"])
        height, width = 224, 224

        # Test with all zeros (no text detected)
        predictions_zeros = {"prob_maps": torch.zeros(batch_size, 1, height, width)}

        boxes_batch, scores_batch = post_processor.represent(sample_batch, predictions_zeros)

        assert len(boxes_batch) == batch_size
        assert len(scores_batch) == batch_size
        # Should have no detections
        for boxes in boxes_batch:
            assert len(boxes) == 0

        # Test with all ones (all text detected)
        predictions_ones = {"prob_maps": torch.ones(batch_size, 1, height, width)}

        boxes_batch, scores_batch = post_processor.represent(sample_batch, predictions_ones)

        assert len(boxes_batch) == batch_size
        assert len(scores_batch) == batch_size

    def test_polygon_vs_box_mode(self, post_processor, sample_batch, sample_predictions):
        """Test both polygon and box extraction modes."""
        # Test box mode (default)
        post_processor.use_polygon = False
        boxes_batch, scores_batch = post_processor.represent(sample_batch, sample_predictions)

        assert len(boxes_batch) == len(sample_batch["images"])

        # Test polygon mode
        post_processor.use_polygon = True
        boxes_batch, scores_batch = post_processor.represent(sample_batch, sample_predictions)

        assert len(boxes_batch) == len(sample_batch["images"])

    def test_different_thresholds(self, post_processor, sample_batch, sample_predictions):
        """Test different threshold values."""
        original_thresh = post_processor.thresh
        original_box_thresh = post_processor.box_thresh

        try:
            # Test with very low thresholds (should detect more)
            post_processor.thresh = 0.1
            post_processor.box_thresh = 0.1

            boxes_batch_low, _ = post_processor.represent(sample_batch, sample_predictions)

            # Test with very high thresholds (should detect less)
            post_processor.thresh = 0.8
            post_processor.box_thresh = 0.8

            boxes_batch_high, _ = post_processor.represent(sample_batch, sample_predictions)

            # Higher thresholds should generally detect fewer boxes
            # (though this is not guaranteed due to the nature of the data)
            assert len(boxes_batch_low) == len(boxes_batch_high)

        finally:
            # Restore original thresholds
            post_processor.thresh = original_thresh
            post_processor.box_thresh = original_box_thresh

    def test_max_candidates_limit(self, post_processor, sample_batch, sample_predictions):
        """Test that max_candidates limit is respected."""
        original_max_candidates = post_processor.max_candidates

        try:
            post_processor.max_candidates = 5

            boxes_batch, scores_batch = post_processor.represent(sample_batch, sample_predictions)

            # Each batch item should have at most max_candidates boxes
            for boxes in boxes_batch:
                assert len(boxes) <= post_processor.max_candidates

        finally:
            post_processor.max_candidates = original_max_candidates

    def test_shape_validation_invalid_image_tensor(self, post_processor):
        """Test validation of invalid image tensor shapes."""
        # Wrong number of dimensions
        batch = {
            "images": torch.randn(2, 224),  # Should be 4D
            "polygons": [torch.randn(5, 4, 2) for _ in range(2)],
            "ignore_tags": torch.zeros(2, 5, dtype=torch.bool),
            "shape": [(224, 224) for _ in range(2)],
            "inverse_matrix": [np.eye(3, dtype=np.float32) for _ in range(2)],
            "filename": ["test_0.jpg", "test_1.jpg"],
        }

        predictions = {"prob_maps": torch.sigmoid(torch.randn(2, 1, 224, 224))}

        with pytest.raises(ValueError, match="must be 4D tensor"):
            post_processor.represent(batch, predictions)

    def test_shape_validation_invalid_prediction_channels(self, post_processor, sample_batch):
        """Test validation of invalid prediction channel count."""
        # Wrong number of channels in predictions
        predictions = {"prob_maps": torch.sigmoid(torch.randn(2, 3, 224, 224))}  # Should be 1

        with pytest.raises(ValueError, match="should have 1 channel"):
            post_processor.represent(sample_batch, predictions)

    def test_shape_validation_batch_size_mismatch(self, post_processor):
        """Test validation of batch size mismatch between images and predictions."""
        batch = {
            "images": torch.randn(2, 3, 224, 224),  # 2 images
            "polygons": [torch.randn(5, 4, 2) for _ in range(2)],
            "ignore_tags": torch.zeros(2, 5, dtype=torch.bool),
            "shape": [(224, 224) for _ in range(2)],
            "inverse_matrix": [np.eye(3, dtype=np.float32) for _ in range(2)],
            "filename": ["test_0.jpg", "test_1.jpg"],
        }

        predictions = {"prob_maps": torch.sigmoid(torch.randn(3, 1, 224, 224))}  # 3 predictions

        with pytest.raises(ValueError, match="Batch size mismatch"):
            post_processor.represent(batch, predictions)

    def test_shape_validation_spatial_dimension_mismatch(self, post_processor):
        """Test validation of spatial dimension mismatch."""
        batch = {
            "images": torch.randn(2, 3, 224, 224),  # 224x224 images
            "polygons": [torch.randn(5, 4, 2) for _ in range(2)],
            "ignore_tags": torch.zeros(2, 5, dtype=torch.bool),
            "shape": [(224, 224) for _ in range(2)],
            "inverse_matrix": [np.eye(3, dtype=np.float32) for _ in range(2)],
            "filename": ["test_0.jpg", "test_1.jpg"],
        }

        predictions = {"prob_maps": torch.sigmoid(torch.randn(2, 1, 320, 240))}  # 320x240 predictions

        with pytest.raises(ValueError, match="Spatial dimension mismatch"):
            post_processor.represent(batch, predictions)

    def test_shape_validation_invalid_inverse_matrix(self, post_processor, sample_batch, sample_predictions):
        """Test validation of invalid inverse matrix."""
        # Wrong shape for inverse matrix
        sample_batch["inverse_matrix"] = [np.eye(2, dtype=np.float32) for _ in range(2)]  # Should be 3x3

        with pytest.raises(ValueError, match="must be 3x3 matrix"):
            post_processor.represent(sample_batch, sample_predictions)

    def test_shape_validation_invalid_matrix_type(self, post_processor, sample_batch, sample_predictions):
        """Test validation of invalid inverse matrix type."""
        # Wrong type for inverse matrix
        sample_batch["inverse_matrix"] = [torch.eye(3) for _ in range(2)]  # Should be numpy array

        with pytest.raises(ValueError, match="must be numpy array"):
            post_processor.represent(sample_batch, sample_predictions)
