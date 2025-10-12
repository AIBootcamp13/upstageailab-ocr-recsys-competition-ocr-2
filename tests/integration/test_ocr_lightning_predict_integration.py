"""Integration test for OCRLightningModule predict loop with evaluator happy-path."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from ocr.lightning_modules.ocr_pl import OCRPLModule


class MockModel:
    """Mock model that simulates the OCR model interface."""

    def __call__(self, return_loss=False, **batch):
        # Simulate model output
        batch_size = batch["images"].shape[0]
        return {
            "loss": torch.tensor(0.1),
            "loss_dict": {"loss_cls": torch.tensor(0.05), "loss_bbox": torch.tensor(0.05)},
            "pred_maps": torch.randn(batch_size, 6, 64, 64),  # Simulated prediction maps
        }

    def get_polygons_from_maps(self, batch, pred):
        # Simulate polygon extraction from prediction maps
        batch_size = batch["images"].shape[0]
        boxes_batch = []
        scores_batch = []

        for i in range(batch_size):
            # Create some mock polygons (4 corners of a rectangle)
            boxes = [
                np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32),  # Box 1
                np.array([[70, 70], [100, 70], [100, 90], [70, 90]], dtype=np.float32),  # Box 2
            ]
            scores = [0.95, 0.87]  # Confidence scores

            boxes_batch.append(boxes)
            scores_batch.append(scores)

        return boxes_batch, scores_batch


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, split="predict"):
        self.split = split
        if split == "predict":
            self.anns = {}  # No annotations needed for prediction


@pytest.fixture
def mock_config():
    """Create a mock configuration for the OCR module."""
    config = OmegaConf.create(
        {"paths": {"submission_dir": "/tmp/test_submissions"}, "minified_json": False, "include_confidence": True, "compile_model": False}
    )
    return config


@pytest.fixture
def mock_batch():
    """Create a mock batch for testing."""
    batch = {
        "images": torch.randn(2, 3, 224, 224),  # 2 images, 3 channels, 224x224
        "image_filename": ["img1.jpg", "img2.jpg"],
        "raw_size": [(224, 224), (224, 224)],
        "orientation": [1, 1],
        "image_path": ["/path/img1.jpg", "/path/img2.jpg"],
        "inverse_matrix": [np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)],  # Identity matrices
        "shape": [(224, 224), (224, 224)],  # Image shapes
        "polygons": [[], []],  # Empty polygons for predict
        "prob_maps": torch.randn(2, 1, 224, 224),  # Probability maps
        "thresh_maps": torch.randn(2, 1, 224, 224),  # Threshold maps
    }
    return batch


def test_ocr_lightning_module_predict_step_happy_path(mock_config, mock_batch):
    """Test the predict_step method of OCRPLModule with mock data."""
    # Setup
    model = MockModel()
    dataset = {"predict": MockDataset("predict")}
    metric_cfg = OmegaConf.create({"_target_": "ocr.metrics.CLEvalMetric", "case_sensitive": False})

    module = OCRPLModule(model, dataset, mock_config, metric_cfg)

    # Execute predict step
    result = module.predict_step(mock_batch)

    # Assertions
    assert result is not None  # Model should return predictions
    assert len(module.predict_step_outputs) == 2  # Should have 2 entries for 2 images

    # Check that outputs are stored correctly
    assert "img1.jpg" in module.predict_step_outputs
    assert "img2.jpg" in module.predict_step_outputs

    # Check structure of stored predictions (with confidence)
    img1_pred = module.predict_step_outputs["img1.jpg"]
    assert "boxes" in img1_pred
    assert "scores" in img1_pred
    assert len(img1_pred["boxes"]) == 2  # 2 boxes per image
    assert len(img1_pred["scores"]) == 2  # 2 scores per image


def test_ocr_lightning_module_on_predict_epoch_end_creates_submission(mock_config, mock_batch):
    """Test that on_predict_epoch_end creates a proper submission file."""
    # Setup
    model = MockModel()
    dataset = {"predict": MockDataset("predict")}
    metric_cfg = OmegaConf.create({"_target_": "ocr.metrics.CLEvalMetric", "case_sensitive": False})

    module = OCRPLModule(model, dataset, mock_config, metric_cfg)

    # Simulate some predictions being stored
    module.predict_step_outputs["img1.jpg"] = {
        "boxes": [
            np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32),
            np.array([[70, 70], [100, 70], [100, 90], [70, 90]], dtype=np.float32),
        ],
        "scores": [0.95, 0.87],
    }

    module.predict_step_outputs["img2.jpg"] = {
        "boxes": [np.array([[20, 20], [60, 20], [60, 40], [20, 40]], dtype=np.float32)],
        "scores": [0.92],
    }

    # Execute on_predict_epoch_end
    with patch("json.dump") as mock_json_dump:
        module.on_predict_epoch_end()

        # Check that json.dump was called (meaning submission was created)
        assert mock_json_dump.called
        assert len(module.predict_step_outputs) == 0  # Should be cleared after epoch end


def test_ocr_lightning_module_predict_loop_integration(mock_config, mock_batch):
    """Integration test for the complete predict loop."""
    # Setup
    model = MockModel()
    dataset = {"predict": MockDataset("predict")}
    metric_cfg = OmegaConf.create({"_target_": "ocr.metrics.CLEvalMetric", "case_sensitive": False})

    module = OCRPLModule(model, dataset, mock_config, metric_cfg)

    # Simulate multiple predict steps (like a dataloader would)
    module.predict_step(mock_batch)

    # Verify state after prediction
    assert len(module.predict_step_outputs) == 2
    for filename in ["img1.jpg", "img2.jpg"]:
        assert filename in module.predict_step_outputs
        pred_data = module.predict_step_outputs[filename]
        assert "boxes" in pred_data
        assert "scores" in pred_data

    # Execute epoch end to finalize submission
    with patch("json.dump") as mock_json_dump, patch("pathlib.Path.open"), patch("pathlib.Path.mkdir"):
        module.on_predict_epoch_end()

        # Verify outputs were processed and cleared
        assert len(module.predict_step_outputs) == 0
        assert mock_json_dump.called

        # Check the structure of the dumped data
        dumped_data = mock_json_dump.call_args[0][0]  # First argument to json.dump
        assert "images" in dumped_data
        assert "img1.jpg" in dumped_data["images"]
        assert "img2.jpg" in dumped_data["images"]

        # Verify image structure
        img1_data = dumped_data["images"]["img1.jpg"]
        assert "words" in img1_data
        assert len(img1_data["words"]) == 2  # 2 words detected in img1


def test_ocr_lightning_module_predict_without_confidence(mock_config, mock_batch):
    """Test predict loop when confidence scores are not included."""
    # Modify config to not include confidence
    mock_config.include_confidence = False

    # Setup
    model = MockModel()
    dataset = {"predict": MockDataset("predict")}
    metric_cfg = OmegaConf.create({"_target_": "ocr.metrics.CLEvalMetric", "case_sensitive": False})

    module = OCRPLModule(model, dataset, mock_config, metric_cfg)

    # Execute predict step
    module.predict_step(mock_batch)

    # Check that outputs are stored without confidence scores
    img1_pred = module.predict_step_outputs["img1.jpg"]
    assert isinstance(img1_pred, list)  # Should be a list of boxes, not a dict
    assert len(img1_pred) == 2  # 2 boxes

    # Execute epoch end
    with patch("json.dump") as mock_json_dump, patch("pathlib.Path.open"), patch("pathlib.Path.mkdir"):
        module.on_predict_epoch_end()

        dumped_data = mock_json_dump.call_args[0][0]
        # Verify confidence is not included in output when include_confidence=False
        img1_words = dumped_data["images"]["img1.jpg"]["words"]
        for word_key, word_data in img1_words.items():
            assert "confidence" not in word_data


def test_ocr_lightning_module_predict_with_evaluator_integration(mock_config, mock_batch):
    """Test predict integration with evaluator components."""
    # Setup with evaluator
    model = MockModel()
    dataset = {"predict": MockDataset("predict")}
    metric_cfg = OmegaConf.create({"_target_": "ocr.metrics.CLEvalMetric", "case_sensitive": False})

    module = OCRPLModule(model, dataset, mock_config, metric_cfg)

    # Verify that evaluator is not created for predict split (only val/test)
    # The predict loop doesn't use the evaluator in the same way as val/test
    assert module.test_evaluator is None  # No test evaluator since dataset["test"] doesn't exist
    assert module.valid_evaluator is None  # No val evaluator since dataset["val"] doesn't exist

    # Execute predict step
    result = module.predict_step(mock_batch)

    # Verify the prediction was processed correctly
    assert result is not None
    assert len(module.predict_step_outputs) == 2

    # Verify the structure of predictions
    for filename in ["img1.jpg", "img2.jpg"]:
        pred_data = module.predict_step_outputs[filename]
        assert "boxes" in pred_data
        assert "scores" in pred_data
        assert len(pred_data["boxes"]) == 2  # 2 boxes per image
        assert len(pred_data["scores"]) == 2  # 2 scores per image
