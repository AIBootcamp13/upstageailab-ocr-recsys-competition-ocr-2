from __future__ import annotations

"""Service layer orchestrating inference requests.

Before modifying behaviour, consult the Streamlit protocols in
``docs/ai_handbook/02_protocols/`` and the UI configuration in
``configs/ui/inference.yaml``. Hyperparameter defaults originate from that YAML
and related schemas; keep them authoritative instead of adding ad-hoc logic
here.
"""

import logging
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import streamlit as st

from ..models.config import PreprocessingConfig
from ..models.ui_events import InferenceRequest
from ..state import InferenceState

LOGGER = logging.getLogger(__name__)

try:
    from ui.utils.inference_engine import run_inference_on_image
except ImportError:  # pragma: no cover - mocked in dev environments
    run_inference_on_image = None

ENGINE_AVAILABLE = run_inference_on_image is not None

try:
    from ocr.datasets.preprocessing import (
        DOCTR_AVAILABLE,
        DocumentPreprocessor,
    )
except ImportError:  # pragma: no cover - optional dependency guard
    DOCTR_AVAILABLE = False
    DocumentPreprocessor = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from ocr.datasets.preprocessing import (
        DocumentPreprocessor as DocumentPreprocessorType,
    )
else:  # pragma: no cover - runtime fallback for optional dependency
    DocumentPreprocessorType = Any


class InferenceService:
    def run(self, state: InferenceState, request: InferenceRequest, hyperparams: dict[str, float]) -> None:
        mode_key = "docTR:on" if request.use_preprocessing else "docTR:off"
        state.ensure_processed_bucket(request.model_path, mode_key)
        total_files = len(request.files)
        new_results: list[dict[str, Any]] = []

        progress = st.progress(0.0, text=f"Starting inference for {total_files} images...")

        preprocessor = None
        if request.use_preprocessing and DocumentPreprocessor is not None:
            preprocessor = self._build_preprocessor(request.preprocessing_config)

        for index, uploaded_file in enumerate(request.files):
            filename = uploaded_file.name
            processed_bucket = state.processed_images[request.model_path][mode_key]
            if filename in processed_bucket:
                progress.progress(
                    (index + 1) / total_files,
                    text=f"Skipped {filename} (already processed)... ({index + 1}/{total_files})",
                )
                continue

            progress.progress(index / total_files, text=f"Processing {filename}... ({index + 1}/{total_files})")

            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = Path(tmp_file.name)

            try:
                result = self._perform_inference(
                    temp_path,
                    Path(request.model_path),
                    filename,
                    hyperparams,
                    request.use_preprocessing,
                    preprocessor,
                )
                new_results.append(result)
                processed_bucket.add(filename)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

        state.inference_results.extend(new_results)
        state.persist()

        progress.progress(1.0, text=f"✅ Inference complete! Processed {len(new_results)} new images.")
        time.sleep(1)
        progress.empty()

    def _perform_inference(
        self,
        image_path: Path,
        model_path: Path,
        filename: str,
        hyperparams: dict[str, float],
        use_preprocessing: bool,
        preprocessor: DocumentPreprocessorType | None,
    ) -> dict[str, Any]:
        processed_temp_path: Path | None = None
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            inference_rgb = image_rgb
            preprocessing_info: dict[str, Any] = {
                "enabled": use_preprocessing,
                "metadata": None,
                "original": image_rgb,
                "processed": None,
                "doctr_available": DOCTR_AVAILABLE,
                "mode": "docTR:on" if use_preprocessing else "docTR:off",
            }
            inference_target_path = image_path

            if use_preprocessing and preprocessor is not None:
                try:
                    processed = preprocessor(image_rgb.copy())
                    processed_image = processed.get("image")
                    if processed_image is None:
                        raise ValueError("DocumentPreprocessor returned no image result.")
                    inference_rgb = np.asarray(processed_image)
                    preprocessing_info["metadata"] = processed.get("metadata")
                    preprocessing_info["processed"] = inference_rgb
                    inference_target_path = self._write_temp_image(inference_rgb, suffix=image_path.suffix)
                    processed_temp_path = inference_target_path
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("docTR preprocessing failed for %s: %s", filename, exc)
                    preprocessing_info["error"] = str(exc)
                    preprocessing_info["enabled"] = False
                    inference_rgb = image_rgb
                    inference_target_path = image_path

            predictions = None
            if ENGINE_AVAILABLE and run_inference_on_image:
                inference_fn = run_inference_on_image
                try:
                    predictions = inference_fn(
                        str(inference_target_path),
                        str(model_path),
                        hyperparams.get("binarization_thresh"),
                        hyperparams.get("box_thresh"),
                        int(hyperparams.get("max_candidates", 300)),
                        int(hyperparams.get("min_detection_size", 5)),
                    )
                    if predictions is None:
                        raise ValueError("Inference engine returned no results.")
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Real inference failed; using mock predictions fallback: %s", exc)
                    st.warning(f"⚠️ Real inference failed ({exc}), using mock predictions as a fallback.")
                    predictions = None

            if predictions is None:
                predictions = self._generate_mock_predictions(inference_rgb.shape)

            return {
                "filename": filename,
                "success": True,
                "image": inference_rgb,
                "predictions": predictions,
                "preprocessing": preprocessing_info,
            }
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Inference failed for %s", filename)
            return {"filename": filename, "success": False, "error": str(exc)}
        finally:
            if processed_temp_path and processed_temp_path.exists():
                processed_temp_path.unlink()

    @staticmethod
    def _generate_mock_predictions(image_shape: Sequence[int]) -> dict[str, Any]:
        height, width, _ = image_shape
        box1 = [int(width * 0.1), int(height * 0.1), int(width * 0.4), int(height * 0.2)]
        box2 = [int(width * 0.5), int(height * 0.4), int(width * 0.9), int(height * 0.5)]
        box3 = [int(width * 0.2), int(height * 0.7), int(width * 0.7), int(height * 0.8)]
        mock_boxes = [box1, box2, box3]

        return {
            "polygons": "|".join(f"{b[0]},{b[1]},{b[2]},{b[1]},{b[2]},{b[3]},{b[0]},{b[3]}" for b in mock_boxes),
            "texts": ["Sample Text 1", "Another Example", "Third Line"],
            "confidences": [0.95, 0.87, 0.92],
        }

    @staticmethod
    def _write_temp_image(image_rgb: np.ndarray, suffix: str) -> Path:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(tmp_file.name, image_bgr):
                raise ValueError("Failed to serialize preprocessed image for inference.")
            return Path(tmp_file.name)

    @staticmethod
    def _build_preprocessor(config: PreprocessingConfig | None) -> DocumentPreprocessorType | None:
        if DocumentPreprocessor is None:
            return None

        if config is None:
            return DocumentPreprocessor()

        kwargs = config.to_kwargs()
        return DocumentPreprocessor(**kwargs)
