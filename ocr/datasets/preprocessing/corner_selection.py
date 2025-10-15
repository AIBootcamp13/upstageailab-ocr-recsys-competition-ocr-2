"""
Corner Selection Utility for Document Quadrilateral Extraction.

This module provides utilities to extract ordered quadrilaterals from arbitrary
corner detections, ensuring robust document boundary detection for preprocessing.
"""

from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .advanced_corner_detection import DetectedCorners


class QuadrilateralSelectionConfig(BaseModel):
    """Configuration for quadrilateral selection from corner detections."""

    min_corners_required: int = Field(default=4, ge=4, le=20, description="Minimum corners needed for selection")
    max_corners_to_consider: int = Field(default=50, ge=4, le=200, description="Maximum corners to process for efficiency")
    geometric_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum geometric confidence for quadrilateral")
    area_ratio_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum area ratio relative to image")
    aspect_ratio_min: float = Field(default=0.2, ge=0.0, le=1.0, description="Minimum aspect ratio for valid quadrilateral")
    aspect_ratio_max: float = Field(default=5.0, ge=1.0, le=10.0, description="Maximum aspect ratio for valid quadrilateral")
    enable_fallback_selection: bool = Field(default=True, description="Enable fallback selection when RANSAC fails")

    @field_validator("aspect_ratio_max")
    @classmethod
    def validate_aspect_ratio_bounds(cls, v: float, info) -> float:
        """Ensure aspect ratio bounds are logical."""
        if info.data.get("aspect_ratio_min", 0.2) > v:
            raise ValueError("aspect_ratio_max must be greater than aspect_ratio_min")
        return v


class SelectedQuadrilateral(BaseModel):
    """Result of quadrilateral selection from corner detections."""

    corners: np.ndarray = Field(..., description="Ordered quadrilateral corners (4x2 array)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Selection confidence score")
    method: str = Field(..., description="Selection method used")
    original_corner_count: int = Field(..., ge=0, description="Number of corners in original detection")
    area_ratio: float = Field(..., ge=0.0, le=1.0, description="Area ratio relative to image bounds")
    aspect_ratio: float = Field(..., ge=0.0, description="Aspect ratio of the quadrilateral")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional selection metadata")

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow numpy arrays


class CornerSelectionUtility:
    """
    Utility for extracting ordered quadrilaterals from arbitrary corner detections.

    This class provides robust selection of exactly 4 corners that form a valid
    document quadrilateral, with proper ordering for downstream processing.
    """

    def __init__(self, config: QuadrilateralSelectionConfig | None = None):
        self.config = config or QuadrilateralSelectionConfig()

    def select_quadrilateral(self, detected_corners: DetectedCorners, image_shape: tuple[int, int]) -> SelectedQuadrilateral | None:
        """
        Select an ordered quadrilateral from detected corners.

        Args:
            detected_corners: Detected corners from corner detection
            image_shape: Shape of the original image (height, width)

        Returns:
            SelectedQuadrilateral if successful, None if selection fails
        """
        if detected_corners.corners is None or len(detected_corners.corners) < self.config.min_corners_required:
            return None

        # Limit corners for efficiency
        corners = detected_corners.corners[: self.config.max_corners_to_consider]

        # Try RANSAC-based quadrilateral fitting first
        result = self._select_via_ransac_fitting(corners, image_shape)
        if result is not None:
            return result

        # Fallback to convex hull selection
        result = self._select_via_convex_hull(corners, image_shape)
        if result is not None:
            return result

        # Final fallback to bounding box if enabled
        if self.config.enable_fallback_selection:
            return self._select_via_fallback_bbox(corners, image_shape)

        return None

    def _select_via_ransac_fitting(self, corners: np.ndarray, image_shape: tuple[int, int]) -> SelectedQuadrilateral | None:
        """Select quadrilateral using RANSAC-based geometric fitting."""
        try:
            # Use minimum 4 corners for RANSAC
            if len(corners) < 4:
                return None

            # Fit quadrilateral using RANSAC (simplified implementation)
            best_quadrilateral = None
            best_confidence = 0.0

            # Try multiple RANSAC iterations
            for _ in range(min(50, len(corners) // 2)):
                # Randomly sample 4 points
                sample_indices = np.random.choice(len(corners), 4, replace=False)
                sample_points = corners[sample_indices]

                # Try to form quadrilateral
                ordered_quad = self._order_quadrilateral_points(sample_points)
                if ordered_quad is None:
                    continue

                # Calculate geometric confidence
                confidence = self._calculate_geometric_confidence(ordered_quad, corners)

                if confidence > best_confidence and confidence >= self.config.geometric_confidence_threshold:
                    best_quadrilateral = ordered_quad
                    best_confidence = confidence

            if best_quadrilateral is not None:
                return self._create_result(best_quadrilateral, best_confidence, "ransac_fitting", len(corners), image_shape)

        except Exception:
            # Silently fail and try next method
            pass

        return None

    def _select_via_convex_hull(self, corners: np.ndarray, image_shape: tuple[int, int]) -> SelectedQuadrilateral | None:
        """Select quadrilateral using convex hull approximation."""
        try:
            # Compute convex hull
            hull = cv2.convexHull(corners.astype(np.float32))
            hull_points = hull.reshape(-1, 2)

            if len(hull_points) < 4:
                return None

            # If exactly 4 points, use them directly
            if len(hull_points) == 4:
                ordered_quad = self._order_quadrilateral_points(hull_points)
                if ordered_quad is not None:
                    confidence = self._calculate_geometric_confidence(ordered_quad, corners)
                    if confidence >= self.config.geometric_confidence_threshold:
                        return self._create_result(ordered_quad, confidence, "convex_hull_selection", len(corners), image_shape)

            # If more than 4 points, select the best 4
            elif len(hull_points) > 4:
                ordered_quad = self._select_best_four_from_hull(hull_points, corners)
                if ordered_quad is not None:
                    confidence = self._calculate_geometric_confidence(ordered_quad, corners)
                    if confidence >= self.config.geometric_confidence_threshold:
                        return self._create_result(ordered_quad, confidence, "convex_hull_selection", len(corners), image_shape)

        except Exception:
            # Silently fail and try next method
            pass

        return None

    def _select_via_fallback_bbox(self, corners: np.ndarray, image_shape: tuple[int, int]) -> SelectedQuadrilateral | None:
        """Fallback selection using bounding box of all corners."""
        try:
            if len(corners) < 4:
                return None

            # Calculate bounding box
            x_coords = corners[:, 0]
            y_coords = corners[:, 1]

            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)

            # Create bounding box corners
            bbox_corners = np.array(
                [
                    [x_min, y_min],  # top-left
                    [x_max, y_min],  # top-right
                    [x_max, y_max],  # bottom-right
                    [x_min, y_max],  # bottom-left
                ],
                dtype=np.float32,
            )

            # Calculate confidence based on corner distribution
            confidence = min(0.5, len(corners) / 20.0)  # Lower confidence for fallback

            return self._create_result(bbox_corners, confidence, "fallback_bbox", len(corners), image_shape)

        except Exception:
            return None

    def _order_quadrilateral_points(self, points: np.ndarray) -> np.ndarray | None:
        """Order 4 points into quadrilateral: top-left, top-right, bottom-right, bottom-left."""
        if len(points) != 4:
            return None

        try:
            # Calculate centroid
            center = np.mean(points, axis=0)

            # Calculate angles from center (clockwise)
            angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

            # Sort by angle
            sorted_indices = np.argsort(angles)
            ordered_points = points[sorted_indices]

            return ordered_points

        except Exception:
            return None

    def _select_best_four_from_hull(self, hull_points: np.ndarray, all_corners: np.ndarray) -> np.ndarray | None:
        """Select the best 4 points from convex hull."""
        try:
            # Simple approach: select points that maximize area
            max_area = 0.0
            best_quad = None

            # Try different combinations of 4 points
            for i in range(len(hull_points) - 3):
                for j in range(i + 1, len(hull_points) - 2):
                    for k in range(j + 1, len(hull_points) - 1):
                        for m in range(k + 1, len(hull_points)):
                            candidate_points = np.array([hull_points[i], hull_points[j], hull_points[k], hull_points[m]])
                            ordered_quad = self._order_quadrilateral_points(candidate_points)
                            if ordered_quad is not None:
                                area = self._calculate_quadrilateral_area(ordered_quad)
                                if area > max_area:
                                    max_area = area
                                    best_quad = ordered_quad

            return best_quad

        except Exception:
            return None

    def _calculate_geometric_confidence(self, quadrilateral: np.ndarray, all_corners: np.ndarray) -> float:
        """Calculate how well the quadrilateral fits all detected corners."""
        try:
            if len(all_corners) == 0:
                return 0.0

            # Calculate distances from corners to quadrilateral edges
            total_distance = 0.0
            for corner in all_corners:
                min_distance = float("inf")
                for i in range(4):
                    edge_start = quadrilateral[i]
                    edge_end = quadrilateral[(i + 1) % 4]
                    distance = self._point_to_line_distance(corner, edge_start, edge_end)
                    min_distance = min(min_distance, distance)
                total_distance += min_distance

            # Normalize by number of corners and image size
            avg_distance = total_distance / len(all_corners)
            confidence = max(0.0, 1.0 - avg_distance / 100.0)  # Assume 100px is poor fit

            return min(1.0, confidence)

        except Exception:
            return 0.0

    def _calculate_quadrilateral_area(self, quadrilateral: np.ndarray) -> float:
        """Calculate area of quadrilateral using shoelace formula."""
        try:
            # Use shoelace formula
            x = quadrilateral[:, 0]
            y = quadrilateral[:, 1]
            return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        except Exception:
            return 0.0

    def _point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """Calculate distance from point to line segment."""
        try:
            line_vec = line_end - line_start
            point_vec = point - line_start
            line_len = np.linalg.norm(line_vec)

            if line_len == 0:
                return float(np.linalg.norm(point_vec))

            # Project point onto line
            proj = np.dot(point_vec, line_vec) / (line_len * line_len)
            proj = np.clip(proj, 0, 1)

            # Find closest point on line segment
            closest_point = line_start + proj * line_vec
            return float(np.linalg.norm(point - closest_point))

        except Exception:
            return float("inf")

    def _create_result(
        self, corners: np.ndarray, confidence: float, method: str, original_count: int, image_shape: tuple[int, int]
    ) -> SelectedQuadrilateral:
        """Create validated SelectedQuadrilateral result."""
        # Calculate area ratio
        quad_area = self._calculate_quadrilateral_area(corners)
        image_area = image_shape[0] * image_shape[1]
        area_ratio = quad_area / image_area if image_area > 0 else 0.0

        # Calculate aspect ratio
        widths = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
        heights = [np.linalg.norm(corners[i] - corners[(i + 3) % 4]) for i in range(4)]
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)
        aspect_ratio = float(avg_width / avg_height) if avg_height > 0 else 0.0

        return SelectedQuadrilateral(
            corners=corners,
            confidence=confidence,
            method=method,
            original_corner_count=original_count,
            area_ratio=area_ratio,
            aspect_ratio=aspect_ratio,
            metadata={"config": self.config.model_dump(), "image_shape": image_shape},
        )
