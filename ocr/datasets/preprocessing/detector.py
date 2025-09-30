"""Document boundary detection utilities."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import cv2
import numpy as np


class DocumentDetector:
    """Detect document boundaries using configurable strategies."""

    def __init__(
        self,
        logger: logging.Logger,
        min_area_ratio: float,
        use_adaptive: bool,
        use_fallback: bool,
    ) -> None:
        self.logger = logger
        self.min_area_ratio = min_area_ratio
        self.use_adaptive = use_adaptive
        self.use_fallback = use_fallback

    def detect(self, image: np.ndarray) -> tuple[np.ndarray | None, str | None]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape[:2]
        min_area = self.min_area_ratio * float(height * width)

        corners = self._detect_document_from_edges(gray, min_area)
        if corners is not None:
            return corners, "canny_contour"

        if self.use_adaptive:
            corners = self._detect_document_with_adaptive(gray, min_area)
            if corners is not None:
                return corners, "adaptive_threshold"

        if self.use_fallback:
            corners = self._fallback_document_bounding_box(gray, min_area)
            if corners is not None:
                return corners, "bounding_box"

        return None, "failed"

    def _detect_document_from_edges(self, gray: np.ndarray, min_area: float) -> np.ndarray | None:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        return self._extract_corners_from_binary(dilated, min_area)

    def _detect_document_with_adaptive(self, gray: np.ndarray, min_area: float) -> np.ndarray | None:
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            15,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
        return self._extract_corners_from_binary(closed, min_area)

    def _extract_corners_from_binary(self, binary: np.ndarray, min_area: float) -> np.ndarray | None:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = self._select_document_contour(contours, min_area)
        if contour is None:
            return None

        corners = self._approximate_corners(contour)
        return self._order_corners(corners) if corners is not None else None

    def _select_document_contour(self, contours: Sequence[np.ndarray], min_area: float) -> np.ndarray | None:
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= max(min_area, 1.0)]
        return max(valid_contours, key=cv2.contourArea) if valid_contours else None

    def _approximate_corners(self, contour: np.ndarray) -> np.ndarray | None:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None

        approximations = [0.02, 0.03, 0.04, 0.05]
        for epsilon_factor in approximations:
            approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)
            if approx.shape[0] == 4:
                return approx.reshape(4, 2)

        hull = cv2.convexHull(contour)
        if hull is not None and hull.shape[0] >= 4:
            hull_perimeter = cv2.arcLength(hull, True)
            if hull_perimeter > 0:
                for epsilon_factor in approximations:
                    approx = cv2.approxPolyDP(hull, epsilon_factor * hull_perimeter, True)
                    if approx.shape[0] == 4:
                        return approx.reshape(4, 2)

        return None

    def _fallback_document_bounding_box(self, gray: np.ndarray, min_area: float) -> np.ndarray | None:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        coords = cv2.findNonZero(cleaned)
        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)
        if w <= 10 or h <= 10:
            return None

        if float(w * h) < max(min_area * 0.6, 1.0):
            return None

        return np.array(
            [
                [x, y],
                [x + w - 1, y],
                [x + w - 1, y + h - 1],
                [x, y + h - 1],
            ],
            dtype=np.float32,
        )

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        sums = corners.sum(axis=1)
        diffs = np.diff(corners, axis=1).flatten()

        ordered_corners = np.zeros((4, 2), dtype=np.float32)
        top_left_idx = int(np.argmin(sums))
        bottom_right_idx = int(np.argmax(sums))

        ordered_corners[0] = corners[top_left_idx]
        ordered_corners[2] = corners[bottom_right_idx]

        remaining = [idx for idx in range(4) if idx not in (top_left_idx, bottom_right_idx)]
        ordered_corners[1] = corners[remaining[int(np.argmin(diffs[remaining]))]]
        ordered_corners[3] = corners[remaining[int(np.argmax(diffs[remaining]))]]

        return ordered_corners


__all__ = ["DocumentDetector"]
