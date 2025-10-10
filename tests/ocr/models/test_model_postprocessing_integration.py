"""
Integration tests for the complete OCR pipeline: model → post-processing

Tests the end-to-end flow from model predictions through post-processing
to ensure the entire inference pipeline works correctly with proper shape handling.

This addresses Task 3.1.3 of the data pipeline testing plan.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from ocr.models.head.db_postprocess import DBPostProcessor


class MockDBModel(nn.Module):
    """Mock DB model for integration testing."""

    def __init__(self, input_channels=3, output_channels=1):
        super().__init__()
        self.backbone = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.neck = nn.Conv2d(64, 64, 3, padding=1)
        self.head = nn.Conv2d(64, output_channels, 1)

    def forward(self, x):
        x = torch.relu(self.backbone(x))
        x = torch.relu(self.neck(x))
        prob_map = torch.sigmoid(self.head(x))
        return {"prob_maps": prob_map}


class TestModelPostProcessingIntegration:
    """Integration tests for model → post-processing pipeline."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock DB model for testing."""
        return MockDBModel()

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
            "polygons": [torch.randn(10, 4, 2) for _ in range(batch_size)],
            "ignore_tags": torch.zeros(batch_size, 10, dtype=torch.bool),
            "shape": [(height, width) for _ in range(batch_size)],
            "inverse_matrix": [np.eye(3, dtype=np.float32) for _ in range(batch_size)],
            "filename": [f"test_{i}.jpg" for i in range(batch_size)],
        }

    def test_end_to_end_pipeline_single_image(self, mock_model, post_processor):
        """Test complete pipeline from single image through post-processing."""
        # Create single image batch
        image = torch.randn(1, 3, 224, 224)
        batch = {
            "images": image,
            "polygons": [torch.randn(5, 4, 2)],
            "ignore_tags": torch.zeros(1, 5, dtype=torch.bool),
            "shape": [(224, 224)],
            "inverse_matrix": [np.eye(3, dtype=np.float32)],
            "filename": ["test_single.jpg"],
        }

        # Run model inference
        with torch.no_grad():
            predictions = mock_model(image)

        # Run post-processing
        boxes_batch, scores_batch = post_processor.represent(batch, predictions)

        # Validate results structure
        assert isinstance(boxes_batch, list)
        assert isinstance(scores_batch, list)
        assert len(boxes_batch) == 1  # Single image
        assert len(scores_batch) == 1  # Single image

        boxes = boxes_batch[0]
        scores = scores_batch[0]
        assert isinstance(boxes, list)
        assert isinstance(scores, list)
        assert len(boxes) == len(scores)

        # Each box should be a valid quadrilateral
        for box in boxes:
            assert isinstance(box, np.ndarray)
            assert box.shape == (4, 2)  # 4 corners, 2 coordinates each

    def test_end_to_end_pipeline_batch_processing(self, mock_model, post_processor, sample_batch):
        """Test complete pipeline with batch processing."""
        # Run model inference
        with torch.no_grad():
            predictions = mock_model(sample_batch["images"])

        # Run post-processing
        boxes_batch, scores_batch = post_processor.represent(sample_batch, predictions)

        # Validate results structure
        assert isinstance(boxes_batch, list)
        assert isinstance(scores_batch, list)
        assert len(boxes_batch) == len(sample_batch["images"])  # Match batch size
        assert len(scores_batch) == len(sample_batch["images"])  # Match batch size

        for boxes, scores in zip(boxes_batch, scores_batch, strict=True):
            assert isinstance(boxes, list)
            assert isinstance(scores, list)
            assert len(boxes) == len(scores)

            # Each box should be a valid quadrilateral
            for box in boxes:
                assert isinstance(box, np.ndarray)
                assert box.shape == (4, 2)  # 4 corners, 2 coordinates each

    def test_pipeline_consistency_across_runs(self, mock_model, post_processor, sample_batch):
        """Test that the pipeline produces consistent results across multiple runs."""
        # Run pipeline multiple times
        results_runs = []
        for _ in range(3):
            with torch.no_grad():
                predictions = mock_model(sample_batch["images"])
            results = post_processor.represent(sample_batch, predictions)
            results_runs.append(results)

        # All runs should produce the same structure
        for results in results_runs:
            boxes_batch, scores_batch = results
            assert len(boxes_batch) == len(sample_batch["images"])
            assert len(scores_batch) == len(sample_batch["images"])
            for boxes, scores in zip(boxes_batch, scores_batch, strict=True):
                assert isinstance(boxes, list)
                assert isinstance(scores, list)

        # Note: Due to random model weights, exact polygon coordinates may differ
        # but the structure and types should be consistent

    def test_pipeline_with_different_image_sizes(self, mock_model, post_processor):
        """Test pipeline with different image sizes in the same batch."""
        # Create batch with different sizes
        sizes = [(224, 224), (320, 240), (160, 160)]
        batch_size = len(sizes)

        batch = {
            "images": torch.randn(batch_size, 3, max(s[0] for s in sizes), max(s[1] for s in sizes)),
            "polygons": [torch.randn(5, 4, 2) for _ in range(batch_size)],
            "ignore_tags": torch.zeros(batch_size, 5, dtype=torch.bool),
            "shape": sizes,
            "inverse_matrix": [np.eye(3, dtype=np.float32) for _ in range(batch_size)],
            "filename": [f"test_{i}.jpg" for i in range(batch_size)],
        }

        # Run model inference
        with torch.no_grad():
            predictions = mock_model(batch["images"])

        # Run post-processing
        boxes_batch, scores_batch = post_processor.represent(batch, predictions)

        # Validate results
        assert len(boxes_batch) == batch_size
        assert len(scores_batch) == batch_size
        for boxes, scores in zip(boxes_batch, scores_batch, strict=True):
            assert isinstance(boxes, list)
            assert isinstance(scores, list)

    def test_pipeline_error_propagation(self, mock_model, post_processor, sample_batch):
        """Test that pipeline properly propagates and handles errors."""
        # Test with invalid predictions (wrong shape)
        invalid_predictions = {"prob_maps": torch.randn(1, 1, 224, 224)}  # Wrong batch size

        with pytest.raises(ValueError, match="Batch size mismatch"):
            post_processor.represent(sample_batch, invalid_predictions)

    def test_pipeline_with_empty_predictions(self, post_processor):
        """Test pipeline behavior with predictions that result in no detections."""
        batch_size = 2
        height, width = 224, 224

        # Create batch
        batch = {
            "images": torch.randn(batch_size, 3, height, width),
            "polygons": [torch.randn(5, 4, 2) for _ in range(batch_size)],
            "ignore_tags": torch.zeros(batch_size, 5, dtype=torch.bool),
            "shape": [(height, width) for _ in range(batch_size)],
            "inverse_matrix": [np.eye(3, dtype=np.float32) for _ in range(batch_size)],
            "filename": [f"test_{i}.jpg" for i in range(batch_size)],
        }

        # Create predictions with very low probabilities (should result in no detections)
        predictions = {"prob_maps": torch.zeros(batch_size, 1, height, width)}

        # Run post-processing
        boxes_batch, scores_batch = post_processor.represent(batch, predictions)

        # Should still return valid structure, but potentially empty results
        assert isinstance(boxes_batch, list)
        assert isinstance(scores_batch, list)
        assert len(boxes_batch) == batch_size
        assert len(scores_batch) == batch_size

        for boxes, scores in zip(boxes_batch, scores_batch, strict=True):
            assert isinstance(boxes, list)
            assert isinstance(scores, list)
            # Boxes and scores lists might be empty if no detections above threshold

    def test_pipeline_memory_efficiency(self, mock_model, post_processor):
        """Test that pipeline doesn't have obvious memory leaks or excessive usage."""
        import gc
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run multiple batches
        for i in range(5):
            batch = {
                "images": torch.randn(2, 3, 224, 224),
                "polygons": [torch.randn(10, 4, 2) for _ in range(2)],
                "ignore_tags": torch.zeros(2, 10, dtype=torch.bool),
                "shape": [(224, 224) for _ in range(2)],
                "inverse_matrix": [np.eye(3, dtype=np.float32) for _ in range(2)],
                "filename": [f"test_{i}_{j}.jpg" for j in range(2)],
            }

            with torch.no_grad():
                predictions = mock_model(batch["images"])
            _ = post_processor.represent(batch, predictions)

            # Force garbage collection
            gc.collect()

        # Check memory usage after runs
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB for this test)
        # This is a basic check - in practice, more sophisticated memory profiling would be used
        assert memory_increase < 100, f"Memory increase too large: {memory_increase}MB"
