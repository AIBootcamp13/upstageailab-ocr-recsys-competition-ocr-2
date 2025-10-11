"""Geometry helper functions extracted from dataset transforms."""

from __future__ import annotations

import numpy as np


def calculate_inverse_transform(
    original_size: tuple[int, int],
    transformed_size: tuple[int, int],
    *,
    crop_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Compute inverse transform matrix mapping transformed coordinates back to original space."""

    ox, oy = original_size
    tx, ty = transformed_size
    cx, cy = 0, 0

    if crop_box:
        cx, cy, tx, ty = crop_box

    # Scale back to the original size
    scale_x = ox / tx
    scale_y = oy / ty
    scale_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float32)

    # Padding back to the original size
    translation_matrix = np.eye(3, dtype=np.float32)
    translation_matrix[0, 2] = -cx
    translation_matrix[1, 2] = -cy

    inverse_matrix = scale_matrix @ translation_matrix
    return inverse_matrix.astype(np.float32)


def calculate_cropbox(original_size: tuple[int, int], target_size: int = 640) -> tuple[int, int, int, int]:
    """Determine crop box applied during resize letterboxing."""

    ox, oy = original_size
    scale = target_size / max(ox, oy)
    new_width, new_height = int(ox * scale), int(oy * scale)
    delta_w = target_size - new_width
    delta_h = target_size - new_height
    x, y = delta_w // 2, delta_h // 2
    w, h = new_width, new_height
    return x, y, w, h
