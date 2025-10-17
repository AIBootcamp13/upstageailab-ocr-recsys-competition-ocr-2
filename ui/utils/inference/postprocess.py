from __future__ import annotations

"""Prediction post-processing utilities."""

import logging
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from .config_loader import PostprocessSettings
from .dependencies import torch

LOGGER = logging.getLogger(__name__)


def compute_inverse_matrix(processed_tensor, original_shape: Sequence[int]):
    if torch is None:
        return [np.eye(3, dtype=np.float32)]

    model_height = int(processed_tensor.shape[-2])
    model_width = int(processed_tensor.shape[-1])
    original_height = int(original_shape[0])
    original_width = int(original_shape[1])

    if not model_width or not model_height:
        return [np.eye(3, dtype=np.float32)]

    scale_x = original_width / model_width
    scale_y = original_height / model_height

    matrix = np.array(
        [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return [matrix]


def decode_polygons_with_head(model, processed_tensor, predictions, original_shape: Sequence[int]) -> dict[str, Any] | None:
    head = getattr(model, "head", None)
    if head is None or not hasattr(head, "get_polygons_from_maps"):
        return None

    inverse_matrix = compute_inverse_matrix(processed_tensor, original_shape)
    batch = {
        "images": processed_tensor,
        "shape": [tuple(original_shape)],
        "filename": ["input"],
        "inverse_matrix": inverse_matrix,
    }

    polygons_result = head.get_polygons_from_maps(batch, predictions)

    if not polygons_result:
        return None

    boxes_batch, scores_batch = polygons_result
    if not boxes_batch:
        return None

    polygons: list[str] = []
    texts: list[str] = []
    confidences: list[float] = []

    for index, box in enumerate(boxes_batch[0]):
        if not box or len(box) < 4:
            continue
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        polygon_coords = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
        # Competition format uses space-separated coordinates
        polygons.append(" ".join(map(str, polygon_coords)))
        texts.append(f"Text_{index + 1}")
        score = scores_batch[0][index] if scores_batch and index < len(scores_batch[0]) else 0.0
        confidences.append(float(score))

    return {
        "polygons": "|".join(polygons) if polygons else "",
        "texts": texts,
        "confidences": confidences,
    }


def fallback_postprocess(predictions: Any, original_shape: Sequence[int], settings: PostprocessSettings) -> dict[str, Any]:
    prob_map = predictions.get("prob_maps")
    if prob_map is None:
        raise ValueError("'prob_maps' key not found in model predictions.")

    if torch is not None and isinstance(prob_map, torch.Tensor):
        prob_map = prob_map.detach().cpu().numpy()

    prob_map = np.squeeze(prob_map)
    binary_map = (prob_map > settings.binarization_thresh).astype(np.uint8)

    original_height, original_width, _ = original_shape
    model_height, model_width = prob_map.shape[:2]
    if not model_width or not model_height:
        raise ValueError(f"Invalid model dimensions: height={model_height}, width={model_width}")

    scale_x = original_width / model_width
    scale_y = original_height / model_height

    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: list[str] = []
    texts: list[str] = []
    confidences: list[float] = []

    for index, contour in enumerate(contours):
        if len(polygons) >= settings.max_candidates:
            LOGGER.info("Reached maximum candidates limit: %s", settings.max_candidates)
            break

        x, y, w, h = cv2.boundingRect(contour)
        if w < settings.min_detection_size or h < settings.min_detection_size:
            continue

        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)
        orig_w = int(w * scale_x)
        orig_h = int(h * scale_y)

        polygon_coords = [
            orig_x,
            orig_y,
            orig_x + orig_w,
            orig_y,
            orig_x + orig_w,
            orig_y + orig_h,
            orig_x,
            orig_y + orig_h,
        ]
        # Competition format uses space-separated coordinates
        polygons.append(" ".join(map(str, polygon_coords)))
        texts.append(f"Text_{index + 1}")

        prob_slice = prob_map[y : y + h, x : x + w]
        confidence = float(prob_slice.mean()) if prob_slice.size else 0.0
        confidences.append(confidence)

    return {
        "polygons": "|".join(polygons),
        "texts": texts,
        "confidences": confidences,
    }
