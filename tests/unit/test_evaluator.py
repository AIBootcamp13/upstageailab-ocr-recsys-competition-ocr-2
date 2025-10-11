"""Unit tests for evaluator module."""

from unittest.mock import MagicMock, patch

import numpy as np

from ocr.evaluation.evaluator import CLEvalEvaluator
from ocr.validation.models import LightningStepPrediction


class MockDataset:
    """Mock dataset for testing purposes."""

    def __init__(self, annotations=None):
        self.anns = annotations or {}
        self.image_path = "/mock/path"


class MockSubsetDataset:
    """Mock subset dataset that has indices and a base dataset."""

    def __init__(self, base_dataset, indices):
        self.dataset = base_dataset
        self.indices = indices


class MockCLEvalMetric:
    """Mock CLEvalMetric for testing."""

    def __init__(self, **kwargs):
        # Don't pass kwargs to the actual CLEvalMetric since it might not accept them
        self.reset_called = False
        self.call_args = None
        # Return tensors as expected by the real implementation
        import torch

        self.compute_result = {"recall": torch.tensor(0.9), "precision": torch.tensor(0.8), "f1": torch.tensor(0.85)}

    def reset(self):
        self.reset_called = True

    def __call__(self, det_quads, gt_quads):
        self.call_args = (det_quads, gt_quads)

    def compute(self):
        return self.compute_result


class TestCLEvalEvaluator:
    """Tests for CLEvalEvaluator class."""

    def test_init_with_default_values(self):
        """Test initialization with default values."""
        dataset = MockDataset()
        evaluator = CLEvalEvaluator(dataset)

        assert evaluator.dataset == dataset
        assert evaluator.mode == "val"
        assert evaluator.metric_cfg == {}
        assert evaluator.enable_validation is True
        assert len(evaluator.predictions) == 0

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        dataset = MockDataset()
        # Use parameters that are valid for CLEvalMetric
        metric_cfg = {
            "case_sensitive": False,
            "recall_gran_penalty": 0.5,
            "precision_gran_penalty": 0.5,
            "vertical_aspect_ratio_thresh": 0.7,
            "ap_constraint": 0.4,
        }
        evaluator = CLEvalEvaluator(dataset, metric_cfg=metric_cfg, mode="test", enable_validation=False)

        assert evaluator.dataset == dataset
        assert evaluator.mode == "test"
        assert evaluator.metric_cfg == metric_cfg
        assert evaluator.enable_validation is False
        assert len(evaluator.predictions) == 0

    def test_update_with_validation_enabled(self):
        """Test update method with validation enabled."""
        dataset = MockDataset()
        evaluator = CLEvalEvaluator(dataset, enable_validation=True)

        filenames = ["img1.jpg", "img2.jpg"]
        predictions = [
            {"boxes": [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])], "orientation": 0, "raw_size": (100, 100)},
            {"boxes": [np.array([[2, 2], [3, 2], [3, 3], [2, 3]])], "orientation": 0, "raw_size": (100, 100)},
        ]

        with patch("ocr.evaluation.evaluator.validate_predictions") as mock_validate:
            # Mock the validation to return LightningStepPrediction objects
            mock_validate.return_value = [
                LightningStepPrediction(boxes=[np.array([[0, 0], [1, 0], [1, 1], [0, 1]])], orientation=0, raw_size=(100, 100)),
                LightningStepPrediction(boxes=[np.array([[2, 2], [3, 2], [3, 3], [2, 3]])], orientation=0, raw_size=(100, 100)),
            ]

            evaluator.update(filenames, predictions)

        assert len(evaluator.predictions) == 2
        assert "img1.jpg" in evaluator.predictions
        assert "img2.jpg" in evaluator.predictions
        mock_validate.assert_called_once()

    def test_update_with_validation_disabled(self):
        """Test update method with validation disabled."""
        dataset = MockDataset()
        evaluator = CLEvalEvaluator(dataset, enable_validation=False)

        filenames = ["img1.jpg", "img2.jpg"]
        predictions = [
            {"boxes": [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])], "orientation": 0, "raw_size": (100, 100)},
            {"boxes": [np.array([[2, 2], [3, 2], [3, 3], [2, 3]])], "orientation": 0, "raw_size": (100, 100)},
        ]

        evaluator.update(filenames, predictions)

        assert len(evaluator.predictions) == 2
        assert "img1.jpg" in evaluator.predictions
        assert "img2.jpg" in evaluator.predictions

    def test_reset_clears_predictions(self):
        """Test that reset method clears all predictions."""
        dataset = MockDataset()
        evaluator = CLEvalEvaluator(dataset)

        # Add some predictions
        filenames = ["img1.jpg", "img2.jpg"]
        predictions = [
            {"boxes": [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])], "orientation": 0, "raw_size": (100, 100)},
            {"boxes": [np.array([[2, 2], [3, 2], [3, 3], [2, 3]])], "orientation": 0, "raw_size": (100, 100)},
        ]
        evaluator.update(filenames, predictions)

        assert len(evaluator.predictions) == 2

        evaluator.reset()

        assert len(evaluator.predictions) == 0

    @patch("ocr.evaluation.evaluator.CLEvalMetric", MockCLEvalMetric)
    def test_compute_with_predictions_and_ground_truth(self):
        """Test compute method with predictions and ground truth data."""
        # Create mock dataset with annotations
        annotations = {
            "img1.jpg": [{"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "text": "test"}],
            "img2.jpg": [{"polygon": [[20, 20], [30, 20], [30, 30], [20, 30]], "text": "sample"}],
        }
        dataset = MockDataset(annotations)

        evaluator = CLEvalEvaluator(dataset, mode="val")

        # Add predictions
        filenames = ["img1.jpg", "img2.jpg"]
        predictions = [
            {
                "boxes": [np.array([[1, 1], [9, 1], [9, 9], [1, 9]])],
                "orientation": 0,
                "raw_size": (100, 100),
                "image_path": "/path/img1.jpg",
            },
            {
                "boxes": [np.array([[21, 21], [29, 21], [29, 29], [21, 29]])],
                "orientation": 0,
                "raw_size": (100, 100),
                "image_path": "/path/img2.jpg",
            },
        ]
        evaluator.update(filenames, predictions)

        # Mock PIL Image.open to return a mock image with size
        with patch("PIL.Image.open") as mock_image_open:
            mock_image = MagicMock()
            mock_image.size = (100, 100)
            mock_image_open.return_value.__enter__.return_value = mock_image

            with patch("ocr.evaluation.evaluator.remap_polygons") as mock_remap:
                mock_remap.return_value = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)]

                results = evaluator.compute()

        assert "val/recall" in results
        assert "val/precision" in results
        assert "val/hmean" in results
        assert abs(results["val/recall"] - 0.9) < 0.01  # Allow small floating point differences
        assert abs(results["val/precision"] - 0.8) < 0.01
        assert abs(results["val/hmean"] - 0.85) < 0.01

    @patch("ocr.evaluation.evaluator.CLEvalMetric", MockCLEvalMetric)
    def test_compute_with_no_predictions(self):
        """Test compute method when no predictions have been made."""
        annotations = {"img1.jpg": [{"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "text": "test"}]}
        dataset = MockDataset(annotations)

        evaluator = CLEvalEvaluator(dataset, mode="val")

        # Don't add any predictions

        results = evaluator.compute()

        assert results["val/recall"] == 0.0
        assert results["val/precision"] == 0.0
        assert results["val/hmean"] == 0.0

    @patch("ocr.evaluation.evaluator.CLEvalMetric", MockCLEvalMetric)
    def test_compute_with_subset_dataset(self):
        """Test compute method with a subset dataset (has indices)."""
        base_annotations = {
            "img1.jpg": [{"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "text": "test"}],
            "img2.jpg": [{"polygon": [[20, 20], [30, 20], [30, 30], [20, 30]], "text": "sample"}],
            "img3.jpg": [{"polygon": [[40, 40], [50, 40], [50, 50], [40, 50]], "text": "example"}],
        }
        base_dataset = MockDataset(base_annotations)
        subset_dataset = MockSubsetDataset(base_dataset, [0, 2])  # Only img1 and img3

        evaluator = CLEvalEvaluator(subset_dataset, mode="test")

        # Add predictions for the subset images
        filenames = ["img1.jpg", "img3.jpg"]
        predictions = [
            {
                "boxes": [np.array([[1, 1], [9, 1], [9, 9], [1, 9]])],
                "orientation": 0,
                "raw_size": (100, 100),
                "image_path": "/path/img1.jpg",
            },
            {
                "boxes": [np.array([[41, 41], [49, 41], [49, 49], [41, 49]])],
                "orientation": 0,
                "raw_size": (100, 100),
                "image_path": "/path/img3.jpg",
            },
        ]
        evaluator.update(filenames, predictions)

        # Mock PIL Image.open to return a mock image with size
        with patch("PIL.Image.open") as mock_image_open:
            mock_image = MagicMock()
            mock_image.size = (100, 100)
            mock_image_open.return_value.__enter__.return_value = mock_image

            with patch("ocr.evaluation.evaluator.remap_polygons") as mock_remap:
                mock_remap.return_value = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)]

                results = evaluator.compute()

        assert "test/recall" in results
        assert "test/precision" in results
        assert "test/hmean" in results

    def test_validation_toggle_with_enable_validation_true(self):
        """Test that validation path is executed when enable_validation=True."""
        dataset = MockDataset()
        evaluator = CLEvalEvaluator(dataset, enable_validation=True)

        filenames = ["img1.jpg"]
        predictions = [{"boxes": [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])], "orientation": 0, "raw_size": (100, 100)}]

        with patch("ocr.evaluation.evaluator.validate_predictions") as mock_validate:
            mock_validate.return_value = [
                LightningStepPrediction(boxes=[np.array([[0, 0], [1, 0], [1, 1], [0, 1]])], orientation=0, raw_size=(100, 100))
            ]

            evaluator.update(filenames, predictions)

        # Verify that validation was called
        mock_validate.assert_called_once()

    def test_validation_toggle_with_enable_validation_false(self):
        """Test that validation path is skipped when enable_validation=False."""
        dataset = MockDataset()
        evaluator = CLEvalEvaluator(dataset, enable_validation=False)

        filenames = ["img1.jpg"]
        predictions = [{"boxes": [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])], "orientation": 0, "raw_size": (100, 100)}]

        with patch("ocr.evaluation.evaluator.validate_predictions") as mock_validate:
            evaluator.update(filenames, predictions)

        # Verify that validation was not called
        mock_validate.assert_not_called()

    @patch("ocr.evaluation.evaluator.CLEvalMetric", MockCLEvalMetric)
    def test_compute_with_empty_detection_boxes(self):
        """Test compute method with empty detection boxes."""
        annotations = {"img1.jpg": [{"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "text": "test"}]}
        dataset = MockDataset(annotations)

        evaluator = CLEvalEvaluator(dataset, mode="val")

        # Add prediction with empty boxes
        filenames = ["img1.jpg"]
        predictions = [{"boxes": [], "orientation": 0, "raw_size": (100, 100), "image_path": "/path/img1.jpg"}]
        evaluator.update(filenames, predictions)

        # Mock PIL Image.open to return a mock image with size
        with patch("PIL.Image.open") as mock_image_open:
            mock_image = MagicMock()
            mock_image.size = (100, 100)
            mock_image_open.return_value.__enter__.return_value = mock_image

            with patch("ocr.evaluation.evaluator.remap_polygons") as mock_remap:
                mock_remap.return_value = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)]

                results = evaluator.compute()

        assert "val/recall" in results
        assert "val/precision" in results
        assert "val/hmean" in results

    def test_rich_iterator_fallback_when_rich_unavailable(self):
        """Test that _rich_iterator falls back to simple iteration when rich is unavailable."""
        filenames = ["img1.jpg", "img2.jpg", "img3.jpg"]

        # Test the iterator method directly
        result = list(CLEvalEvaluator._rich_iterator(filenames, "Test"))

        assert result == filenames
