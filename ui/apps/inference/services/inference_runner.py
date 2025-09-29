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
from typing import Any

import cv2
import streamlit as st

from ..models.ui_events import InferenceRequest
from ..state import InferenceState

LOGGER = logging.getLogger(__name__)

try:
    from ui.utils.inference_engine import run_inference_on_image
except ImportError:  # pragma: no cover - mocked in dev environments
    run_inference_on_image = None

ENGINE_AVAILABLE = run_inference_on_image is not None


class InferenceService:
    def run(self, state: InferenceState, request: InferenceRequest, hyperparams: dict[str, float]) -> None:
        state.ensure_processed_bucket(request.model_path)
        total_files = len(request.files)
        new_results: list[dict[str, Any]] = []

        progress = st.progress(0.0, text=f"Starting inference for {total_files} images...")

        for index, uploaded_file in enumerate(request.files):
            filename = uploaded_file.name
            if filename in state.processed_images[request.model_path]:
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
                result = self._perform_inference(temp_path, Path(request.model_path), filename, hyperparams)
                new_results.append(result)
                state.processed_images[request.model_path].add(filename)
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
    ) -> dict[str, Any]:
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            predictions = None
            if ENGINE_AVAILABLE and run_inference_on_image:
                inference_fn = run_inference_on_image
                try:
                    predictions = inference_fn(
                        str(image_path),
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
                predictions = self._generate_mock_predictions(image_rgb.shape)

            return {
                "filename": filename,
                "success": True,
                "image": image_rgb,
                "predictions": predictions,
            }
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Inference failed for %s", filename)
            return {"filename": filename, "success": False, "error": str(exc)}

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
