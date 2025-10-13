"""Document boundary detection utilities."""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Sequence

import cv2
import numpy as np
from pylsd.lsd import lsd

from .contracts import ContractEnforcer
from .external import DOCTR_AVAILABLE


class DocumentDetector:
    """Det                               cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)  # type: ignorecv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)  # type: ignorect document boundaries using configurable strategies."""

    def __init__(
        self,
        logger: logging.Logger,
        min_area_ratio: float,
        use_adaptive: bool,
        use_fallback: bool,
        use_camscanner: bool = False,
        use_doctr_text: bool = False,
    ) -> None:
        """Initialize document detector.

        Args:
            logger: Logger instance for debug/info messages
            min_area_ratio: Minimum area ratio for valid document detection (0.0-1.0)
            use_adaptive: Whether to use adaptive thresholding as fallback
            use_fallback: Whether to use bounding box fallback when other methods fail
            use_camscanner: Whether to use CamScanner-style LSD line detection (advanced method)
            use_doctr_text: Whether to use docTR text detection to infer document boundaries
        """
        self.logger = logger
        self.min_area_ratio = min_area_ratio
        self.use_adaptive = use_adaptive
        self.use_fallback = use_fallback
        self.use_camscanner = use_camscanner
        self.use_doctr_text = use_doctr_text

    def detect(self, image: np.ndarray) -> tuple[np.ndarray | None, str | None]:
        # Validate input contract
        ContractEnforcer.validate_image_input_contract(image, "detector_input")

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape[:2]
        min_area = self.min_area_ratio * float(height * width)

        # Try docTR text-based detection if enabled
        if self.use_doctr_text and DOCTR_AVAILABLE:
            corners = self._detect_document_with_doctr_text(image, min_area)
            if corners is not None:
                return corners, "doctr_text"

        # Try CamScanner method if enabled
        if self.use_camscanner:
            corners = self._detect_document_camscanner(image, min_area)
            if corners is not None:
                return corners, "camscanner"

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

    def _detect_document_with_doctr_text(self, image: np.ndarray, min_area: float) -> np.ndarray | None:
        """Detect document boundaries using docTR text detection results."""
        try:
            from doctr.models import zoo
        except ImportError:
            self.logger.warning("docTR not available for text-based document detection")
            return None

        try:
            # Convert to RGB for docTR
            if image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Create detection predictor
            predictor = zoo.detection_predictor()

            # Run text detection
            result = predictor([image_rgb])
            if not result or "words" not in result[0]:
                return None

            words = result[0]["words"]
            if len(words) == 0:
                return None

            height, width = image.shape[:2]

            # Filter words by confidence (use moderate confidence threshold)
            confident_words = words[words[:, 4] > 0.6]  # Moderate confidence threshold
            if len(confident_words) < 5:  # Need at least 5 confident detections
                return None

            # Extract bounding boxes (x_min, y_min, x_max, y_max, confidence)
            word_boxes = confident_words[:, :4]

            # Convert relative to absolute coordinates
            abs_boxes = word_boxes.copy()
            abs_boxes[:, [0, 2]] *= width  # x coordinates
            abs_boxes[:, [1, 3]] *= height  # y coordinates

            # Calculate text coverage
            x_min_all, y_min_all = abs_boxes[:, [0, 1]].min(axis=0)
            x_max_all, y_max_all = abs_boxes[:, [2, 3]].max(axis=0)
            x_coverage = (x_max_all - x_min_all) / width
            y_coverage = (y_max_all - y_min_all) / height

            # For receipt OCR, text covering most of the image is normal
            # Only reject if text covers > 95% (extremely edge case)
            if x_coverage > 0.95 and y_coverage > 0.95:
                self.logger.debug("docTR text covers entire image, falling back to other methods")
                return None

            # Use the bounding box of all confident text as the document boundary
            # This works better for documents where text covers the main content area
            x_min = x_min_all
            y_min = y_min_all
            x_max = x_max_all
            y_max = y_max_all

            # Add moderate padding (15% of the text region size)
            x_padding = (x_max - x_min) * 0.15
            y_padding = (y_max - y_min) * 0.15

            x_min = max(0, x_min - x_padding)
            x_max = min(width, x_max + x_padding)
            y_min = max(0, y_min - y_padding)
            y_max = min(height, y_max + y_padding)

            # Check if the detected region meets minimum area requirements
            detected_area = (x_max - x_min) * (y_max - y_min)
            if detected_area < min_area:
                return None

            # Return corners in order: top-left, top-right, bottom-right, bottom-left
            corners = np.array(
                [
                    [x_min, y_min],  # top-left
                    [x_max, y_min],  # top-right
                    [x_max, y_max],  # bottom-right
                    [x_min, y_max],  # bottom-left
                ],
                dtype=np.float32,
            )

            return corners

        except Exception as e:
            self.logger.warning("docTR text-based document detection failed: %s", e)
            return None

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

    def _detect_document_camscanner(self, image: np.ndarray, min_area: float) -> np.ndarray | None:
        """Document detection using CamScanner-style LSD line detection and contour approximation.

        This implements the algorithm from the OpenCV Document Scanner that uses:
        1. LSD (Line Segment Detector) to find lines
        2. Separation into horizontal/vertical lines
        3. Connected components analysis
        4. Corner detection at line intersections
        5. Quadrilateral formation and validation
        """
        # Convert to grayscale and apply preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Dilate to remove holes between edge segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Find edges using Canny
        edged = cv2.Canny(dilated, 0, 84)

        # Get corners using LSD-based line detection
        corners = self._get_corners_lsd(edged)

        if len(corners) >= 4:
            # Generate quadrilaterals from corner combinations
            quads = []
            for quad in itertools.combinations(corners, 4):
                points = np.array(quad)
                points = self._order_points(points)
                points = np.array([[p] for p in points], dtype="int32")
                quads.append(points)

            # Get top quadrilaterals by area
            quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]

            # Sort by angle range to remove outliers
            quads = sorted(quads, key=self._angle_range)

            # Check if the best quadrilateral is valid
            if quads and self._is_valid_contour_camscanner(quads[0], image.shape[1], image.shape[0], min_area):
                return quads[0].reshape(4, 2)

        # Fallback: try direct contour detection
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for c in contours:
            approx = cv2.approxPolyDP(c, 80, True)
            if self._is_valid_contour_camscanner(approx, image.shape[1], image.shape[0], min_area):
                return approx.reshape(4, 2)

        return None

    def _get_corners_lsd(self, img: np.ndarray) -> list[tuple[int, int]]:
        """Extract corners using LSD line detection (CamScanner algorithm)."""
        corners = []

        # Use LSD to detect lines
        lines = lsd(img)

        if lines is not None:
            # Separate horizontal and vertical lines
            lines = lines.squeeze().astype(np.int32).tolist()
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)

            for line in lines:
                x1, y1, x2, y2, _ = line
                if abs(x2 - x1) > abs(y2 - y1):
                    # Horizontal line
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                    cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)  # type: ignore
                else:
                    # Vertical line
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                    cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2)  # type: ignore

            # Process horizontal lines
            contours, _ = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)

            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_x = np.amin(contour[:, 0], axis=0) + 2
                max_x = np.amax(contour[:, 0], axis=0) - 2
                left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
                right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
                cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), (1,), 1)
                corners.append((min_x, left_y))
                corners.append((max_x, right_y))

            # Process vertical lines
            contours, _ = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)

            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_y = np.amin(contour[:, 1], axis=0) + 2
                max_y = np.amax(contour[:, 1], axis=0) - 2
                top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
                bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
                cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), (1,), 1)
                corners.append((top_x, min_y))
                corners.append((bottom_x, max_y))

            # Find intersections (corners where lines overlap)
            corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
            corners.extend(zip(corners_x, corners_y, strict=False))

        # Filter corners to remove duplicates/close points
        corners = self._filter_corners(corners)
        return corners

    def _filter_corners(self, corners: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Filter corners to remove duplicates and points that are too close."""

        def distance(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        filtered_corners: list[tuple[int, int]] = []
        for c in corners:
            should_add = True
            for existing in filtered_corners:
                if distance(c, existing) < 10:  # Minimum distance threshold
                    should_add = False
                    break
            if should_add:
                filtered_corners.append(c)

        return filtered_corners

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in top-left, top-right, bottom-right, bottom-left order.

        Uses the same algorithm as the original pyimagesearch transform.order_points.
        """
        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted x-coordinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their y-coordinates
        # so we can grab the top-left and bottom-left points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        # now that we have the top-left coordinate, use it as an anchor to calculate
        # the Euclidean distance between the top-left and right-most points;
        # by the Pythagorean theorem, the point with the largest distance will be
        # our bottom-right point
        from scipy.spatial import distance as dist

        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        # return the coordinates in top-left, top-right, bottom-right, bottom-left order
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _angle_range(self, quad: np.ndarray) -> float:
        """Calculate the range between max and min interior angles of quadrilateral."""
        if quad.shape[0] != 4:
            return 180.0

        tl, tr, br, bl = quad.reshape(4, 2)

        # Calculate angles at each corner
        angles = []
        corners = [(tl, tr, br), (tr, br, bl), (br, bl, tl), (bl, tl, tr)]

        for p1, p2, p3 in corners:
            # Vectors from p2 to p1 and p2 to p3
            v1 = p1 - p2
            v2 = p3 - p2

            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)  # Handle floating point errors
            angle = np.degrees(np.arccos(cos_angle))
            angles.append(angle)

        return np.ptp(angles) if angles else 180.0

    def _is_valid_contour_camscanner(self, cnt: np.ndarray, im_width: int, im_height: int, min_area: float) -> bool:
        """Validate contour using CamScanner criteria."""
        if cnt is None or len(cnt) != 4:
            return False

        area = cv2.contourArea(cnt)
        if area <= im_width * im_height * 0.25:  # MIN_QUAD_AREA_RATIO = 0.25
            return False

        # Check angle range (should be less than 40 degrees variation)
        angle_range = self._angle_range(cnt.reshape(4, 2))
        return angle_range < 40
