from __future__ import annotations

"""High-level inference engine orchestration."""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ocr.utils.orientation import normalize_pil_image, orientation_requires_rotation, remap_polygons

from .config_loader import ModelConfigBundle, PostprocessSettings, load_model_config, resolve_config_path
from .dependencies import OCR_MODULES_AVAILABLE, torch
from .model_loader import instantiate_model, load_checkpoint, load_state_dict
from .postprocess import decode_polygons_with_head, fallback_postprocess
from .preprocess import build_transform, preprocess_image
from .utils import generate_mock_predictions
from .utils import get_available_checkpoints as scan_checkpoints

LOGGER = logging.getLogger(__name__)

_ORIENTATION_INVERSE = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 8, 7: 7, 8: 6}


class InferenceEngine:
    """OCR Inference Engine for real-time predictions."""

    def __init__(self) -> None:
        self.model = None
        self.trainer = None
        self.config: Any | None = None
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

        self._config_bundle: ModelConfigBundle | None = None
        self._transform = None
        self._postprocess_settings: PostprocessSettings | None = None
        LOGGER.info("Using device: %s", self.device)

    # Public API ---------------------------------------------------------
    def load_model(self, checkpoint_path: str, config_path: str | None = None) -> bool:
        if not OCR_MODULES_AVAILABLE:
            LOGGER.error("OCR modules are not installed. Cannot load a real model.")
            return False

        search_dirs: tuple[Path, ...] = ()
        resolved_config = resolve_config_path(checkpoint_path, config_path, search_dirs)
        if resolved_config is None:
            LOGGER.error("Could not find a valid config file for checkpoint: %s", checkpoint_path)
            return False

        LOGGER.info("Loading model from checkpoint: %s", checkpoint_path)
        bundle = load_model_config(resolved_config)
        self._apply_config_bundle(bundle)

        model_config = getattr(bundle.raw_config, "model", None)
        if model_config is None:
            LOGGER.error("Configuration missing 'model' section: %s", resolved_config)
            return False

        try:
            model = instantiate_model(model_config)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to instantiate model from config %s", resolved_config)
            return False

        checkpoint = load_checkpoint(checkpoint_path, self.device)
        if checkpoint is None:
            LOGGER.error("Failed to load checkpoint %s", checkpoint_path)
            return False

        if not load_state_dict(model, checkpoint):
            LOGGER.error("Failed to load state dictionary for checkpoint %s", checkpoint_path)
            return False

        self.model = model.to(self.device)
        self.model.eval()
        self.config = bundle.raw_config
        return True

    def update_postprocessor_params(
        self,
        binarization_thresh: float | None = None,
        box_thresh: float | None = None,
        max_candidates: int | None = None,
        min_detection_size: int | None = None,
    ) -> None:
        if self.model is None:
            LOGGER.warning("Model not loaded, cannot update postprocessor parameters.")
            return

        settings = self._postprocess_settings
        if settings is None:
            return

        updated_settings = PostprocessSettings(
            binarization_thresh=binarization_thresh if binarization_thresh is not None else settings.binarization_thresh,
            box_thresh=box_thresh if box_thresh is not None else settings.box_thresh,
            max_candidates=int(max_candidates) if max_candidates is not None else settings.max_candidates,
            min_detection_size=int(min_detection_size) if min_detection_size is not None else settings.min_detection_size,
        )
        self._postprocess_settings = updated_settings
        if self._config_bundle is not None:
            self._config_bundle = ModelConfigBundle(
                raw_config=self._config_bundle.raw_config,
                preprocess=self._config_bundle.preprocess,
                postprocess=updated_settings,
            )

        head = getattr(self.model, "head", None)
        postprocess = getattr(head, "postprocess", None)
        if postprocess is None:
            LOGGER.info("Inference model lacks a postprocess module; using engine fallbacks.")
            return

        if hasattr(postprocess, "thresh") and binarization_thresh is not None:
            postprocess.thresh = binarization_thresh
        if hasattr(postprocess, "box_thresh") and box_thresh is not None:
            postprocess.box_thresh = box_thresh
        if hasattr(postprocess, "max_candidates") and max_candidates is not None:
            postprocess.max_candidates = int(max_candidates)
        if hasattr(postprocess, "min_size") and min_detection_size is not None:
            postprocess.min_size = int(min_detection_size)

    def predict_image(
        self,
        image_path: str,
        binarization_thresh: float | None = None,
        box_thresh: float | None = None,
        max_candidates: int | None = None,
        min_detection_size: int | None = None,
    ) -> dict[str, Any] | None:
        if self.model is None:
            LOGGER.warning("Model not loaded. Returning mock predictions.")
            return generate_mock_predictions()

        if any(value is not None for value in (binarization_thresh, box_thresh, max_candidates, min_detection_size)):
            self.update_postprocessor_params(
                binarization_thresh=binarization_thresh,
                box_thresh=box_thresh,
                max_candidates=max_candidates,
                min_detection_size=min_detection_size,
            )

        orientation = 1
        canonical_width = 0
        canonical_height = 0

        try:
            with Image.open(image_path) as pil_image:
                normalized_image, orientation = normalize_pil_image(pil_image)

                rgb_image = normalized_image
                if normalized_image.mode != "RGB":
                    rgb_image = normalized_image.convert("RGB")

                image_array = np.asarray(rgb_image)
                canonical_height, canonical_width = image_array.shape[:2]
                image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

                if rgb_image is not normalized_image:
                    rgb_image.close()
                if normalized_image is not pil_image:
                    normalized_image.close()
        except OSError:
            LOGGER.error("Failed to read image at path: %s", image_path)
            return None

        bundle = self._config_bundle
        if bundle is None:
            LOGGER.error("Model configuration bundle missing; load_model must be called first.")
            return None

        if self._transform is None:
            self._transform = build_transform(bundle.preprocess)

        try:
            batch = preprocess_image(image, self._transform)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to preprocess image %s", image_path)
            return None

        if torch is None:
            LOGGER.error("Torch is not available to run inference.")
            return None

        with torch.no_grad():
            predictions = self.model(return_loss=False, images=batch.to(self.device))

        decoded = decode_polygons_with_head(self.model, batch, predictions, image.shape)
        if decoded is not None:
            return self._remap_predictions_if_needed(
                decoded,
                orientation,
                canonical_width,
                canonical_height,
            )

        try:
            result = fallback_postprocess(predictions, image.shape, bundle.postprocess)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Error in post-processing for %s", image_path)
            return generate_mock_predictions()
        return self._remap_predictions_if_needed(
            result,
            orientation,
            canonical_width,
            canonical_height,
        )

    # Internal helpers ---------------------------------------------------
    def _apply_config_bundle(self, bundle: ModelConfigBundle) -> None:
        self._config_bundle = bundle
        self._postprocess_settings = bundle.postprocess
        self._transform = None

    @staticmethod
    def _remap_predictions_if_needed(
        result: dict[str, Any],
        orientation: int,
        canonical_width: int,
        canonical_height: int,
    ) -> dict[str, Any]:
        if not result:
            return result

        if not orientation_requires_rotation(orientation):
            return result

        polygons_str = result.get("polygons")
        if not polygons_str:
            return result

        inverse_orientation = _ORIENTATION_INVERSE.get(orientation, 1)
        if inverse_orientation == 1:
            return result

        remapped_polygons = []
        for polygon_entry in polygons_str.split("|"):
            tokens = [token for token in polygon_entry.split(",") if token]
            if len(tokens) < 8:
                continue
            try:
                coords = np.array([float(value) for value in tokens], dtype=np.float32).reshape(-1, 2)
            except ValueError:
                continue
            remapped_polygons.append(coords)

        if not remapped_polygons:
            return result

        transformed = remap_polygons(remapped_polygons, canonical_width, canonical_height, inverse_orientation)
        serialised: list[str] = []
        for polygon in transformed:
            flat = polygon.reshape(-1)
            serialised.append(",".join(str(int(round(value))) for value in flat))

        updated = dict(result)
        updated["polygons"] = "|".join(serialised)
        return updated


def run_inference_on_image(
    image_path: str,
    checkpoint_path: str,
    binarization_thresh: float | None = None,
    box_thresh: float | None = None,
    max_candidates: int | None = None,
    min_detection_size: int | None = None,
) -> dict[str, Any] | None:
    engine = InferenceEngine()
    if not engine.load_model(checkpoint_path):
        LOGGER.error("Failed to load model in convenience function.")
        return None
    return engine.predict_image(
        image_path=image_path,
        binarization_thresh=binarization_thresh,
        box_thresh=box_thresh,
        max_candidates=max_candidates,
        min_detection_size=min_detection_size,
    )


def get_available_checkpoints() -> list[str]:
    return scan_checkpoints()
