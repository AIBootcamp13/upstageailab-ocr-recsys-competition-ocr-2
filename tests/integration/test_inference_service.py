from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest

from ui.apps.inference.models.data_contracts import Predictions
from ui.apps.inference.services import inference_runner


@pytest.fixture(autouse=True)
def disable_real_inference(monkeypatch):
    """Ensure unit tests never call the real inference engine."""
    monkeypatch.setattr(inference_runner, "ENGINE_AVAILABLE", False)
    monkeypatch.setattr(inference_runner, "run_inference_on_image", None)


def _write_rgb_image(tmp_path: Path, rgb_image: np.ndarray) -> Path:
    path = tmp_path / "sample.png"
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr_image):
        raise RuntimeError("Failed to serialize test image")
    return path


def _hyperparams() -> dict[str, float]:
    return {
        "binarization_thresh": 0.3,
        "box_thresh": 0.5,
        "max_candidates": 300,
        "min_detection_size": 5,
    }


def test_perform_inference_with_doc_tr_preprocessing(tmp_path):
    service = inference_runner.InferenceService()
    rgb_image = np.full((64, 80, 3), 220, dtype=np.uint8)
    image_path = _write_rgb_image(tmp_path, rgb_image)
    processed_image = np.full_like(rgb_image, 123)

    class StubPreprocessor:
        def __init__(self) -> None:
            self.called = False

        def __call__(self, image: np.ndarray) -> dict[str, np.ndarray | dict]:
            self.called = True
            assert np.array_equal(image, rgb_image)
            return {
                "image": processed_image.copy(),
                "metadata": {"processing_steps": ["document_detection", "perspective_correction"]},
            }

    preprocessor = StubPreprocessor()

    result = service._perform_inference(
        image_path=image_path,
        model_path=tmp_path / "model",
        filename=image_path.name,
        hyperparams=_hyperparams(),
        use_preprocessing=True,
        preprocessor=cast(inference_runner.DocumentPreprocessorType, preprocessor),
    )

    assert result.success is True
    assert preprocessor.called is True
    preprocess_info = result.preprocessing
    assert preprocess_info.enabled is True
    assert preprocess_info.metadata == {"processing_steps": ["document_detection", "perspective_correction"]}
    assert preprocess_info.mode == "docTR:on"
    assert isinstance(preprocess_info.processed, np.ndarray)
    assert np.array_equal(preprocess_info.processed, processed_image)
    assert np.array_equal(result.image, processed_image)
    assert preprocess_info.original.shape == rgb_image.shape
    assert isinstance(result.predictions, Predictions)


def test_perform_inference_preprocessing_failure_falls_back(tmp_path):
    service = inference_runner.InferenceService()
    rgb_image = np.full((64, 80, 3), 200, dtype=np.uint8)
    image_path = _write_rgb_image(tmp_path, rgb_image)

    class FailingPreprocessor:
        def __call__(self, image: np.ndarray) -> dict[str, np.ndarray | dict]:
            raise RuntimeError("docTR failure")

    result = service._perform_inference(
        image_path=image_path,
        model_path=tmp_path / "model",
        filename=image_path.name,
        hyperparams=_hyperparams(),
        use_preprocessing=True,
        preprocessor=cast(inference_runner.DocumentPreprocessorType, FailingPreprocessor()),
    )

    assert result.success is True
    preprocess_info = result.preprocessing
    assert preprocess_info.enabled is False
    assert preprocess_info.processed is None
    assert preprocess_info.metadata is None
    assert preprocess_info.mode == "docTR:on"
    assert preprocess_info.original is not None and preprocess_info.original.shape == rgb_image.shape
    assert preprocess_info.error == "docTR failure"
    assert np.array_equal(result.image, rgb_image)
    assert isinstance(result.predictions, Predictions)
