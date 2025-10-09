"""Advanced document boundary detection with Office Lens quality."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.spatial import distance as dist


@dataclass
class DetectionHypothesis:
    """Represents a document detection hypothesis with confidence."""

    corners: np.ndarray
    confidence: float
    method: str
    metadata: dict[str, Any] | None = None


@dataclass
class AdvancedDetectionConfig:
    """Configuration for advanced document detection."""

    # Corner detection parameters
    harris_block_size: int = 2
    harris_ksize: int = 3
    harris_k: float = 0.04
    harris_threshold_ratio: float = 0.01

    # Shi-Tomasi corner refinement
    shi_tomasi_max_corners: int = 100
    shi_tomasi_quality_level: float = 0.01
    shi_tomasi_min_distance: int = 10
    shi_tomasi_block_size: int = 3

    # Geometric modeling
    ransac_max_trials: int = 1000
    ransac_residual_threshold: float = 5.0
    min_quadrilateral_area_ratio: float = 0.2
    max_aspect_ratio_deviation: float = 0.5  # Max deviation from 1.0 for aspect ratio

    # Confidence thresholds
    min_corner_confidence: float = 0.7
    min_geometric_confidence: float = 0.8
    min_overall_confidence: float = 0.85

    # Multi-hypothesis parameters
    max_hypotheses: int = 5
    hypothesis_fusion_threshold: float = 0.1  # Distance threshold for hypothesis fusion


class AdvancedDocumentDetector:
    """Advanced document detector implementing Office Lens quality detection.

    This detector implements the three-phase approach from the handover document:
    1. Advanced corner detection (Harris + Shi-Tomasi)
    2. Geometric document modeling (RANSAC quadrilateral fitting)
    3. High-confidence decision making (multi-hypothesis fusion)
    """

    def __init__(
        self,
        config: AdvancedDetectionConfig | None = None,
        logger: logging.Logger | None = None,
    ):
        self.config = config or AdvancedDetectionConfig()
        self.logger = logger or logging.getLogger(__name__)

    def detect_document(
        self,
        image: np.ndarray,
        min_area_ratio: float = 0.18,
    ) -> tuple[np.ndarray | None, str | None, dict[str, Any]]:
        """Detect document boundaries with Office Lens quality.

        Returns:
            tuple: (corners, method, metadata)
                - corners: np.ndarray of shape (4, 2) with document corners, or None
                - method: string describing the detection method used
                - metadata: dictionary with detection details and confidence scores
        """
        if image is None or image.size == 0:
            return None, "invalid_input", {"error": "Invalid input image"}

        height, width = image.shape[:2]
        min_area = min_area_ratio * float(height * width)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Generate multiple detection hypotheses
        hypotheses = self._generate_detection_hypotheses(gray, min_area)

        if not hypotheses:
            return None, "no_hypotheses", {"error": "No valid detection hypotheses generated"}

        # Fuse and rank hypotheses
        best_hypothesis = self._select_best_hypothesis(hypotheses)

        if best_hypothesis is None or best_hypothesis.confidence < self.config.min_overall_confidence:
            return (
                None,
                "low_confidence",
                {
                    "confidence": best_hypothesis.confidence if best_hypothesis else 0.0,
                    "hypotheses_count": len(hypotheses),
                },
            )

        # Validate final detection
        if not self._validate_document_detection(best_hypothesis.corners, width, height, min_area):
            return (
                None,
                "validation_failed",
                {
                    "confidence": best_hypothesis.confidence,
                    "validation_error": "Geometric validation failed",
                },
            )

        metadata = {
            "confidence": best_hypothesis.confidence,
            "method": best_hypothesis.method,
            "hypotheses_count": len(hypotheses),
            "detection_metadata": best_hypothesis.metadata or {},
        }

        return best_hypothesis.corners, f"advanced_{best_hypothesis.method}", metadata

    def _generate_detection_hypotheses(
        self,
        gray: np.ndarray,
        min_area: float,
    ) -> list[DetectionHypothesis]:
        """Generate multiple document detection hypotheses."""
        hypotheses = []

        # Method 1: Harris corner detection + geometric modeling
        harris_hypothesis = self._detect_with_harris_corners(gray, min_area)
        if harris_hypothesis:
            hypotheses.append(harris_hypothesis)

        # Method 2: Shi-Tomasi corner detection + geometric modeling
        shi_tomasi_hypothesis = self._detect_with_shi_tomasi_corners(gray, min_area)
        if shi_tomasi_hypothesis:
            hypotheses.append(shi_tomasi_hypothesis)

        # Method 3: Combined Harris + Shi-Tomasi approach
        combined_hypothesis = self._detect_with_combined_corners(gray, min_area)
        if combined_hypothesis:
            hypotheses.append(combined_hypothesis)

        # Method 4: Contour-based detection with corner refinement
        contour_hypothesis = self._detect_with_contour_refinement(gray, min_area)
        if contour_hypothesis:
            hypotheses.append(contour_hypothesis)

        return hypotheses

    def _detect_with_harris_corners(
        self,
        gray: np.ndarray,
        min_area: float,
    ) -> DetectionHypothesis | None:
        """Detect document using Harris corner detection."""
        try:
            # Apply Harris corner detection
            corners = self._extract_harris_corners(gray)

            if len(corners) < 4:
                return None

            # Fit quadrilateral using RANSAC
            quad_corners, geometric_confidence = self._fit_quadrilateral_ransac(corners)

            if quad_corners is None:
                return None

            # Calculate overall confidence
            corner_confidence = min(1.0, len(corners) / 20.0)  # Normalize by expected corner count
            overall_confidence = (corner_confidence + geometric_confidence) / 2.0

            return DetectionHypothesis(
                corners=quad_corners,
                confidence=overall_confidence,
                method="harris_corners",
                metadata={
                    "corner_count": len(corners),
                    "geometric_confidence": geometric_confidence,
                    "corner_confidence": corner_confidence,
                },
            )

        except Exception as e:
            self.logger.debug(f"Harris corner detection failed: {e}")
            return None

    def _detect_with_shi_tomasi_corners(
        self,
        gray: np.ndarray,
        min_area: float,
    ) -> DetectionHypothesis | None:
        """Detect document using Shi-Tomasi corner detection."""
        try:
            # Apply Shi-Tomasi corner detection
            corners = self._extract_shi_tomasi_corners(gray)

            if len(corners) < 4:
                return None

            # Fit quadrilateral using RANSAC
            quad_corners, geometric_confidence = self._fit_quadrilateral_ransac(corners)

            if quad_corners is None:
                return None

            # Calculate overall confidence
            corner_confidence = min(1.0, len(corners) / 15.0)  # Normalize by expected corner count
            overall_confidence = (corner_confidence + geometric_confidence) / 2.0

            return DetectionHypothesis(
                corners=quad_corners,
                confidence=overall_confidence,
                method="shi_tomasi_corners",
                metadata={
                    "corner_count": len(corners),
                    "geometric_confidence": geometric_confidence,
                    "corner_confidence": corner_confidence,
                },
            )

        except Exception as e:
            self.logger.debug(f"Shi-Tomasi corner detection failed: {e}")
            return None

    def _detect_with_combined_corners(
        self,
        gray: np.ndarray,
        min_area: float,
    ) -> DetectionHypothesis | None:
        """Detect document using combined Harris and Shi-Tomasi corners."""
        try:
            # Extract corners from both methods
            harris_corners = self._extract_harris_corners(gray)
            shi_tomasi_corners = self._extract_shi_tomasi_corners(gray)

            # Combine and deduplicate corners
            all_corners = (
                np.vstack([harris_corners, shi_tomasi_corners])
                if len(harris_corners) > 0 and len(shi_tomasi_corners) > 0
                else harris_corners
                if len(harris_corners) > 0
                else shi_tomasi_corners
            )

            if len(all_corners) < 4:
                return None

            # Remove duplicates based on distance
            unique_corners = self._deduplicate_corners(all_corners, threshold=10.0)

            if len(unique_corners) < 4:
                return None

            # Fit quadrilateral using RANSAC
            quad_corners, geometric_confidence = self._fit_quadrilateral_ransac(unique_corners)

            if quad_corners is None:
                return None

            # Calculate overall confidence
            corner_confidence = min(1.0, len(unique_corners) / 25.0)  # Normalize by expected corner count
            overall_confidence = (corner_confidence + geometric_confidence) / 2.0

            return DetectionHypothesis(
                corners=quad_corners,
                confidence=overall_confidence,
                method="combined_corners",
                metadata={
                    "harris_corners": len(harris_corners),
                    "shi_tomasi_corners": len(shi_tomasi_corners),
                    "unique_corners": len(unique_corners),
                    "geometric_confidence": geometric_confidence,
                    "corner_confidence": corner_confidence,
                },
            )

        except Exception as e:
            self.logger.debug(f"Combined corner detection failed: {e}")
            return None

    def _detect_with_contour_refinement(
        self,
        gray: np.ndarray,
        min_area: float,
    ) -> DetectionHypothesis | None:
        """Detect document using contour detection with corner refinement."""
        try:
            # Find contours using traditional method
            contours = self._find_document_contours(gray, min_area)

            if not contours:
                return None

            # Select best contour
            best_contour = max(contours, key=cv2.contourArea)

            # Approximate polygon
            epsilon = 0.02 * cv2.arcLength(best_contour, True)
            approx = cv2.approxPolyDP(best_contour, epsilon, True)

            if len(approx) != 4:
                return None

            corners = approx.reshape(4, 2).astype(np.float32)

            # Refine corners using sub-pixel accuracy
            refined_corners = self._refine_corners_subpixel(gray, corners)

            # Validate geometric properties
            geometric_confidence = self._calculate_geometric_confidence(refined_corners)

            if geometric_confidence < self.config.min_geometric_confidence:
                return None

            return DetectionHypothesis(
                corners=refined_corners,
                confidence=geometric_confidence,
                method="contour_refinement",
                metadata={
                    "contour_area": cv2.contourArea(best_contour),
                    "approx_vertices": len(approx),
                    "geometric_confidence": geometric_confidence,
                },
            )

        except Exception as e:
            self.logger.debug(f"Contour refinement detection failed: {e}")
            return None

    def _extract_harris_corners(self, gray: np.ndarray) -> np.ndarray:
        """Extract corners using Harris corner detection with adaptive thresholding."""
        # Apply Harris corner detection
        dst = cv2.cornerHarris(
            gray,
            blockSize=self.config.harris_block_size,
            ksize=self.config.harris_ksize,
            k=self.config.harris_k,
        )

        # Dilate to mark corners
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dst = cv2.dilate(dst, kernel)

        # Adaptive thresholding based on local maxima
        threshold = self.config.harris_threshold_ratio * dst.max()

        # Find corners above threshold
        corner_mask = dst > threshold
        corner_coords = np.column_stack(np.where(corner_mask))

        # Convert to (x, y) format and ensure float32
        corners = corner_coords[:, [1, 0]].astype(np.float32)  # Swap to (x, y)

        return corners

    def _extract_shi_tomasi_corners(self, gray: np.ndarray) -> np.ndarray:
        """Extract corners using Shi-Tomasi corner detection."""
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.shi_tomasi_max_corners,
            qualityLevel=self.config.shi_tomasi_quality_level,
            minDistance=self.config.shi_tomasi_min_distance,
            blockSize=self.config.shi_tomasi_block_size,
        )

        if corners is None:
            return np.array([])

        # Reshape to (N, 2) format
        corners = corners.reshape(-1, 2).astype(np.float32)

        return corners

    def _fit_quadrilateral_ransac(
        self,
        corners: np.ndarray,
    ) -> tuple[np.ndarray | None, float]:
        """Fit quadrilateral to corners using RANSAC algorithm."""
        if len(corners) < 4:
            return None, 0.0

        best_quad = None
        best_score = 0.0
        best_inliers = 0

        for _ in range(self.config.ransac_max_trials):
            # Randomly sample 4 points
            if len(corners) == 4:
                sample_indices = [0, 1, 2, 3]
            else:
                sample_indices = list(np.random.choice(len(corners), 4, replace=False))

            sample_points = corners[sample_indices]

            # Try to form a quadrilateral
            quad = self._points_to_quadrilateral(sample_points)
            if quad is None:
                continue

            # Calculate inliers (points close to the quadrilateral edges)
            inliers = self._calculate_quadrilateral_inliers(corners, quad)

            # Score based on number of inliers and geometric validity
            geometric_score = self._calculate_geometric_confidence(quad)
            score = len(inliers) * geometric_score

            if score > best_score:
                best_score = score
                best_quad = quad
                best_inliers = len(inliers)

        if best_quad is None:
            return None, 0.0

        # Normalize confidence by number of corners and geometric quality
        confidence = min(1.0, (best_inliers / len(corners)) * self._calculate_geometric_confidence(best_quad))

        return best_quad, confidence

    def _points_to_quadrilateral(self, points: np.ndarray) -> np.ndarray | None:
        """Convert 4 points to ordered quadrilateral corners."""
        if len(points) != 4:
            return None

        # Order points: top-left, top-right, bottom-right, bottom-left
        return self._order_points(points)

    def _calculate_quadrilateral_inliers(
        self,
        all_points: np.ndarray,
        quad: np.ndarray,
    ) -> list[np.ndarray]:
        """Calculate which points are inliers to the quadrilateral."""
        inliers = []

        # Check distance from each point to nearest quadrilateral edge
        for point in all_points:
            min_distance = float("inf")

            # Check distance to each edge
            for i in range(4):
                edge_start = quad[i]
                edge_end = quad[(i + 1) % 4]
                distance = self._point_to_line_distance(point, edge_start, edge_end)
                min_distance = min(min_distance, distance)

            if min_distance <= self.config.ransac_residual_threshold:
                inliers.append(point)

        return inliers

    def _calculate_geometric_confidence(self, corners: np.ndarray) -> float:
        """Calculate geometric confidence score for quadrilateral."""
        if len(corners) != 4:
            return 0.0

        # Check aspect ratio
        widths = [
            dist.euclidean(corners[0], corners[1]),  # top
            dist.euclidean(corners[2], corners[3]),  # bottom
        ]
        heights = [
            dist.euclidean(corners[0], corners[3]),  # left
            dist.euclidean(corners[1], corners[2]),  # right
        ]

        avg_width = np.mean(widths)
        avg_height = np.mean(heights)

        if avg_width == 0 or avg_height == 0:
            return 0.0

        aspect_ratio = max(avg_width, avg_height) / min(avg_width, avg_height)
        aspect_ratio_deviation = abs(aspect_ratio - 1.0)

        # Check angles (should be close to 90 degrees for rectangles)
        angles = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            p3 = corners[(i + 2) % 4]

            v1 = p1 - p2
            v2 = p3 - p2

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))
            angles.append(angle)

        avg_angle_deviation = np.mean([abs(angle - 90) for angle in angles])

        # Calculate confidence based on geometric properties
        aspect_confidence = max(0.0, 1.0 - aspect_ratio_deviation / self.config.max_aspect_ratio_deviation)
        angle_confidence = max(0.0, 1.0 - avg_angle_deviation / 45.0)  # 45 degrees max deviation

        return float((aspect_confidence + angle_confidence) / 2.0)

    def _select_best_hypothesis(self, hypotheses: list[DetectionHypothesis]) -> DetectionHypothesis | None:
        """Select the best hypothesis using confidence-weighted selection."""
        if not hypotheses:
            return None

        # Sort by confidence
        sorted_hypotheses = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)

        # Take the top hypothesis
        best = sorted_hypotheses[0]

        # Check if we should fuse multiple hypotheses
        if len(sorted_hypotheses) > 1:
            second_best = sorted_hypotheses[1]
            if abs(best.confidence - second_best.confidence) < 0.1:  # Close confidence
                # Check if corners are similar (within fusion threshold)
                avg_distance = np.mean([dist.euclidean(c1, c2) for c1, c2 in zip(best.corners, second_best.corners, strict=True)])
                if avg_distance < self.config.hypothesis_fusion_threshold * max(best.corners.max(), second_best.corners.max()):
                    # Fuse hypotheses by averaging corners
                    fused_corners = (best.corners + second_best.corners) / 2.0
                    fused_confidence = (best.confidence + second_best.confidence) / 2.0

                    best = DetectionHypothesis(
                        corners=fused_corners,
                        confidence=fused_confidence,
                        method="fused_hypotheses",
                        metadata={
                            "fused_methods": [best.method, second_best.method],
                            "original_confidences": [best.confidence, second_best.confidence],
                        },
                    )

        return best

    def _validate_document_detection(
        self,
        corners: np.ndarray,
        width: int,
        height: int,
        min_area: float,
    ) -> bool:
        """Validate final document detection."""
        if corners is None or len(corners) != 4:
            return False

        # Check area
        area = cv2.contourArea(corners.astype(np.int32))
        if area < min_area:
            return False

        # Check bounds
        if (corners < 0).any() or (corners[:, 0] >= width).any() or (corners[:, 1] >= height).any():
            return False

        # Check geometric confidence
        geometric_confidence = self._calculate_geometric_confidence(corners)
        if geometric_confidence < self.config.min_geometric_confidence:
            return False

        return True

    # Utility methods
    def _deduplicate_corners(self, corners: np.ndarray, threshold: float = 10.0) -> np.ndarray:
        """Remove duplicate corners within threshold distance."""
        if len(corners) <= 1:
            return corners

        unique_corners = [corners[0]]

        for corner in corners[1:]:
            is_duplicate = False
            for existing in unique_corners:
                if dist.euclidean(corner, existing) < threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_corners.append(corner)

        return np.array(unique_corners)

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in top-left, top-right, bottom-right, bottom-left order."""
        # Sort by x-coordinate
        x_sorted = pts[np.argsort(pts[:, 0]), :]

        # Split into left and right halves
        left_most = x_sorted[:2, :]
        right_most = x_sorted[2:, :]

        # Sort left points by y-coordinate
        left_most = left_most[np.argsort(left_most[:, 1]), :]
        tl, bl = left_most

        # Find bottom-right and top-right from right points
        D = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
        br, tr = right_most[np.argsort(D)[::-1], :]

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """Calculate distance from point to line segment."""
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)

        if line_len == 0:
            return float(np.linalg.norm(point_vec))

        # Project point onto line
        proj = np.dot(point_vec, line_vec) / (line_len**2)
        proj = np.clip(proj, 0, 1)

        # Find closest point on line segment
        closest = line_start + proj * line_vec

        return float(np.linalg.norm(point - closest))

    def _find_document_contours(self, gray: np.ndarray, min_area: float) -> list[np.ndarray]:
        """Find document-like contours in the image."""
        # Apply preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

        return valid_contours

    def _refine_corners_subpixel(self, gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Refine corner positions using sub-pixel accuracy."""
        try:
            # Define search window
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            refined_corners = cv2.cornerSubPix(gray, corners.copy(), (5, 5), (-1, -1), criteria)
            return refined_corners
        except cv2.error:
            # Fallback to original corners if sub-pixel refinement fails
            return corners
