"""Perspective correction component."""

from __future__ import annotations

import logging
from collections.abc import Callable

import cv2
import numpy as np

from .external import extract_rcrops


class PerspectiveCorrector:
    """Perspective correction via docTR or OpenCV."""

    def __init__(
        self,
        logger: logging.Logger,
        ensure_doctr: Callable[[str], bool],
        use_doctr_geometry: bool,
        doctr_assume_horizontal: bool,
    ) -> None:
        self.logger = logger
        self.ensure_doctr = ensure_doctr
        self.use_doctr_geometry = use_doctr_geometry
        self.doctr_assume_horizontal = doctr_assume_horizontal

    def correct(self, image: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        if self.use_doctr_geometry:
            try:
                corrected, matrix = self._doctr_perspective_correction(image, corners)
                return corrected, matrix, "doctr_rcrop"
            except Exception as error:  # pragma: no cover - best-effort fallback
                self.logger.warning(
                    "docTR perspective correction failed (%s); falling back to OpenCV.",
                    error,
                )
        corrected, matrix = self._opencv_perspective_correction(image, corners)
        return corrected, matrix, "opencv"

    def _doctr_perspective_correction(self, image: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.ensure_doctr("doctr_geometry"):
            raise RuntimeError("docTR geometry helpers unavailable")
        if extract_rcrops is None:
            raise RuntimeError("docTR extract_rcrops not available")

        norm_corners = self._normalize_corners(corners, image)
        crops = extract_rcrops(  # type: ignore[call-arg]
            image,
            np.expand_dims(norm_corners, axis=0),
            assume_horizontal=self.doctr_assume_horizontal,
        )
        if not crops:
            raise RuntimeError("docTR extract_rcrops returned no crops")

        corrected = crops[0]
        src_points, (max_width, max_height), dst_points = self._compute_perspective_targets(corners)
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        if corrected.shape[0] != max_height or corrected.shape[1] != max_width:
            corrected = cv2.resize(corrected, (max_width, max_height), interpolation=cv2.INTER_LINEAR)

        return corrected, perspective_matrix

    def _opencv_perspective_correction(self, image: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        src_points, (max_width, max_height), dst_points = self._compute_perspective_targets(corners)
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        corrected = cv2.warpPerspective(image, perspective_matrix, (max_width, max_height), flags=cv2.INTER_LINEAR)
        return corrected, perspective_matrix

    @staticmethod
    def _normalize_corners(corners: np.ndarray, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        norm_corners = corners.astype(np.float32).copy()
        norm_corners[:, 0] /= max(width, 1)
        norm_corners[:, 1] /= max(height, 1)
        return norm_corners

    @staticmethod
    def _compute_perspective_targets(
        corners: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
        tl, tr, br, bl = corners.astype(np.float32)

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(round(width_a)), int(round(width_b)), 1)

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(round(height_a)), int(round(height_b)), 1)

        dst_points = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype=np.float32,
        )

        src_points = np.array([tl, tr, br, bl], dtype=np.float32)
        return src_points, (max_width, max_height), dst_points


__all__ = ["PerspectiveCorrector"]
