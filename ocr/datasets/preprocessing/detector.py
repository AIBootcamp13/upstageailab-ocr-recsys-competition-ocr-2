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
        use_advanced_scanner: bool = False,
    ) -> None:
        self.logger = logger
        self.min_area_ratio = min_area_ratio
        self.use_adaptive = use_adaptive
        self.use_fallback = use_fallback
        self.use_advanced_scanner = use_advanced_scanner

    def detect(self, image: np.ndarray) -> tuple[np.ndarray | None, str | None]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape[:2]
        min_area = self.min_area_ratio * float(height * width)

        # Try advanced scanner first if enabled
        if self.use_advanced_scanner:
            corners = self._detect_document_with_advanced_scanner(image, min_area)
            if corners is not None:
                return corners, "advanced_scanner"

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

    def _detect_document_with_advanced_scanner(self, image: np.ndarray, min_area: float) -> np.ndarray | None:
        """Advanced document detection using multi-threshold edge detection and morphological processing."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)

            # Try different Canny thresholds optimized for documents
            edges1 = cv2.Canny(filtered, 30, 100)  # Lower thresholds for document edges
            edges2 = cv2.Canny(filtered, 50, 150)  # Standard thresholds
            edges3 = cv2.Canny(filtered, 75, 200)  # Higher thresholds for cleaner edges

            # Combine edges
            edges = cv2.bitwise_or(edges1, edges2)
            edges = cv2.bitwise_or(edges, edges3)

            # Apply morphological operations to connect document boundaries
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Dilate to connect broken edges
            edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            height, width = image.shape[:2]

            # Filter out contours that are clearly the full image boundary
            filtered_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                margin = 10  # pixels from edge
                is_full_boundary = x <= margin and y <= margin and x + w >= width - margin and y + h >= height - margin

                # Skip only if it's clearly the full image boundary
                if not is_full_boundary:
                    filtered_contours.append(contour)

            # If no contours after filtering, use original contours but skip the largest one
            if not filtered_contours and len(contours) > 1:
                filtered_contours = contours[1:]  # Skip the largest contour
            elif not filtered_contours:
                filtered_contours = contours[:1]  # Use only the largest if it's the only one

            # Try different epsilon factors for approximation
            epsilon_factors = [0.02, 0.04, 0.06]

            for contour in filtered_contours[:10]:  # Check top 10 contours
                area = cv2.contourArea(contour)

                # Skip contours that are too small
                if area < min_area * 0.1:
                    continue

                for eps_factor in epsilon_factors:
                    epsilon = eps_factor * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    # Look for approximations with 4 points
                    if len(approx) == 4:
                        vertices = approx.reshape(4, 2).astype(np.float32)

                        # Basic validation
                        if self._is_valid_quadrilateral_basic(vertices, min_area):
                            # Additional check: don't return image boundary corners
                            x_coords = vertices[:, 0]
                            y_coords = vertices[:, 1]
                            min_x, max_x = np.min(x_coords), np.max(x_coords)
                            min_y, max_y = np.min(y_coords), np.max(y_coords)

                            # If corners are too close to image boundaries, reject them
                            boundary_margin = 20  # pixels
                            if (
                                min_x <= boundary_margin
                                and min_y <= boundary_margin
                                and max_x >= width - boundary_margin
                                and max_y >= height - boundary_margin
                            ):
                                continue  # This is likely the image boundary, skip it

                            return self._order_corners(vertices)

            return None

        except Exception as e:
            self.logger.warning(f"Advanced scanner failed: {e}")
            return None

    def _is_valid_quadrilateral_basic(self, vertices: np.ndarray, min_area: float) -> bool:
        """Basic validation for quadrilateral corners."""
        if vertices is None or len(vertices) != 4:
            return False

        # Check for degenerate cases
        unique_points = np.unique(vertices.reshape(-1, 2), axis=0)
        if len(unique_points) < 4:
            return False

        # Check area using shoelace formula
        x, y = vertices[:, 0], vertices[:, 1]
        area = 0.5 * abs(x[0] * y[1] + x[1] * y[2] + x[2] * y[3] + x[3] * y[0] - (x[1] * y[0] + x[2] * y[1] + x[3] * y[2] + x[0] * y[3]))
        return area > min_area * 0.05  # Lower threshold for advanced scanner


__all__ = ["DocumentDetector"]
