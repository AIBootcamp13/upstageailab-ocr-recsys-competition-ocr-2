"""Orientation correction component."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from .detector import DocumentDetector
from .external import doctr_rotate_image, estimate_page_angle


class OrientationCorrector:
    """Orientation correction using docTR utilities."""

    def __init__(
        self,
        logger: logging.Logger,
        ensure_doctr: Callable[[str], bool],
        detector: DocumentDetector,
        angle_threshold: float,
        expand_canvas: bool,
        preserve_origin_shape: bool,
    ) -> None:
        self.logger = logger
        self.ensure_doctr = ensure_doctr
        self.detector = detector
        self.angle_threshold = angle_threshold
        self.expand_canvas = expand_canvas
        self.preserve_origin_shape = preserve_origin_shape

    def correct(
        self,
        image: np.ndarray,
        corners: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any] | None]:
        if not self.ensure_doctr("orientation_correction"):
            return image, corners, None
        if estimate_page_angle is None or doctr_rotate_image is None:
            return image, corners, None
        if corners.shape != (4, 2):
            return image, corners, None

        height, width = image.shape[:2]
        rel_corners = corners.astype(np.float32).copy()
        rel_corners[:, 0] /= max(width, 1)
        rel_corners[:, 1] /= max(height, 1)

        angle = float(estimate_page_angle(np.expand_dims(rel_corners, axis=0)))
        if not np.isfinite(angle):
            return image, corners, None

        metadata: dict[str, Any] = {
            "original_angle": angle,
            "angle_correction": 0.0,
        }

        if abs(angle) < self.angle_threshold:
            metadata["skipped_reason"] = "below_threshold"
            return image, corners, metadata

        rotated = doctr_rotate_image(  # type: ignore[call-arg]
            image,
            -angle,
            expand=self.expand_canvas,
            preserve_origin_shape=self.preserve_origin_shape,
        )

        new_corners, method = self.detector.detect(rotated)
        metadata["angle_correction"] = -angle
        metadata["processed_shape"] = tuple(int(dim) for dim in rotated.shape)

        if new_corners is None:
            self.logger.debug(
                "Orientation correction applied (angle=%.2f°) but redetection failed; keeping original corners.",
                angle,
            )
            metadata["redetection_success"] = False
            return rotated, corners, metadata

        metadata["redetection_success"] = True
        metadata["redetection_method"] = method
        return rotated, new_corners, metadata


__all__ = ["OrientationCorrector"]
