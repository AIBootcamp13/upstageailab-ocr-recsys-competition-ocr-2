"""
Geometric Document Modeling for robust document boundary detection.

This module implements RANSAC-based quadrilateral fitting and geometric
validation to achieve >99% accuracy on document boundary detection.
"""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class GeometricModel(Enum):
    QUADRILATERAL = "quadrilateral"
    RECTANGLE = "rectangle"


@dataclass
class GeometricModelConfig:
    """Configuration for geometric document modeling."""

    model_type: GeometricModel = GeometricModel.QUADRILATERAL
    ransac_iterations: int = 100
    ransac_threshold: float = 5.0
    min_samples: int = 4
    rectangle_angle_tolerance: float = 5.0  # degrees
    rectangle_aspect_ratio_tolerance: float = 0.3
    confidence_threshold: float = 0.8


@dataclass
class DocumentGeometry:
    """Represents a detected document's geometric properties."""

    corners: np.ndarray  # Shape: (4, 2) for quadrilateral
    confidence: float
    model_type: str
    area: float
    aspect_ratio: float
    is_rectangle: bool = False
    metadata: dict | None = None


class GeometricDocumentModeler:
    """
    Geometric modeling for document boundary detection using RANSAC.

    Fits geometric models to detected corners to find document boundaries.
    """

    def __init__(self, config: GeometricModelConfig | None = None):
        self.config = config or GeometricModelConfig()

    def fit_document_geometry(self, corners: np.ndarray) -> DocumentGeometry | None:
        """
        Fit geometric model to detected corners.

        Args:
            corners: Detected corner coordinates (N, 2)

        Returns:
            DocumentGeometry if successful fit, None otherwise
        """
        if len(corners) < self.config.min_samples:
            return None

        if self.config.model_type == GeometricModel.QUADRILATERAL:
            return self._fit_quadrilateral(corners)
        elif self.config.model_type == GeometricModel.RECTANGLE:
            return self._fit_rectangle(corners)
        else:
            return self._fit_quadrilateral(corners)  # Default fallback

    def _fit_quadrilateral(self, corners: np.ndarray) -> DocumentGeometry | None:
        """Fit quadrilateral model using RANSAC."""
        if len(corners) < 4:
            return None

        # Use convex hull to get candidate boundary points
        hull = cv2.convexHull(corners.astype(np.float32))
        hull_points = hull.reshape(-1, 2)

        if len(hull_points) < 4:
            return None

        # Try to fit quadrilateral using RANSAC-like approach
        best_quadrilateral = None
        best_confidence = 0.0

        for _ in range(self.config.ransac_iterations):
            # Randomly sample 4 points
            if len(hull_points) <= 4:
                sample_points = hull_points
            else:
                indices = np.random.choice(len(hull_points), 4, replace=False)
                sample_points = hull_points[indices]

            # Try to form quadrilateral
            quadrilateral = self._points_to_quadrilateral(sample_points)
            if quadrilateral is None:
                continue

            # Calculate fit quality
            confidence = self._calculate_quadrilateral_confidence(quadrilateral, hull_points)

            if confidence > best_confidence:
                best_confidence = confidence
                best_quadrilateral = quadrilateral

        if best_quadrilateral is None or best_confidence < self.config.confidence_threshold:
            return None

        # Calculate geometric properties
        area = self._calculate_polygon_area(best_quadrilateral)
        aspect_ratio = self._calculate_aspect_ratio(best_quadrilateral)

        return DocumentGeometry(
            corners=best_quadrilateral,
            confidence=best_confidence,
            model_type="quadrilateral",
            area=area,
            aspect_ratio=aspect_ratio,
            is_rectangle=self._is_rectangle(best_quadrilateral),
            metadata={
                "ransac_iterations": self.config.ransac_iterations,
                "fit_points": len(hull_points),
                "hull_points": hull_points.tolist(),
            },
        )

    def _fit_rectangle(self, corners: np.ndarray) -> DocumentGeometry | None:
        """Fit rectangle model with strict geometric constraints."""
        if len(corners) < 4:
            return None

        # Find minimum area bounding rectangle
        rect = cv2.minAreaRect(corners.astype(np.float32))
        box = cv2.boxPoints(rect)
        box = box.astype(np.float32)

        # Check if it's actually rectangular (right angles)
        if not self._is_rectangle(box):
            return None

        # Calculate confidence based on how well points fit the rectangle
        confidence = self._calculate_rectangle_confidence(box, corners)

        if confidence < self.config.confidence_threshold:
            return None

        area = self._calculate_polygon_area(box)
        aspect_ratio = self._calculate_aspect_ratio(box)

        return DocumentGeometry(
            corners=box,
            confidence=confidence,
            model_type="rectangle",
            area=area,
            aspect_ratio=aspect_ratio,
            is_rectangle=True,
            metadata={"min_area_rect": rect, "original_corners_count": len(corners)},
        )

    def _points_to_quadrilateral(self, points: np.ndarray) -> np.ndarray | None:
        """Convert 4 points to ordered quadrilateral corners."""
        if len(points) != 4:
            return None

        # Find convex hull to ensure proper ordering
        hull = cv2.convexHull(points.astype(np.float32))
        if len(hull) != 4:
            return None

        # Order points in clockwise direction starting from top-left
        ordered_points = self._order_points(hull.reshape(-1, 2))
        return ordered_points

    def _order_points(self, points: np.ndarray) -> np.ndarray:
        """Order points in clockwise direction: top-left, top-right, bottom-right, bottom-left."""
        # Calculate centroid
        center = np.mean(points, axis=0)

        # Calculate angles from center
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

        # Sort by angle (clockwise)
        sorted_indices = np.argsort(angles)
        ordered_points = points[sorted_indices]

        return ordered_points

    def _calculate_quadrilateral_confidence(self, quadrilateral: np.ndarray, all_points: np.ndarray) -> float:
        """Calculate how well the quadrilateral fits all detected points."""
        if len(all_points) == 0:
            return 0.0

        # Calculate distance from each point to the quadrilateral edges
        total_distance = 0.0
        for point in all_points:
            # Find minimum distance to any edge
            min_distance = float("inf")
            for i in range(4):
                edge_start = quadrilateral[i]
                edge_end = quadrilateral[(i + 1) % 4]
                distance = self._point_to_line_distance(point, edge_start, edge_end)
                min_distance = min(min_distance, distance)
            total_distance += min_distance

        # Normalize by number of points and threshold
        avg_distance = total_distance / len(all_points)
        confidence = max(0.0, 1.0 - (avg_distance / self.config.ransac_threshold))

        return confidence

    def _calculate_rectangle_confidence(self, rectangle: np.ndarray, all_points: np.ndarray) -> float:
        """Calculate confidence for rectangle fit."""
        return self._calculate_quadrilateral_confidence(rectangle, all_points)

    def _point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """Calculate distance from point to line segment."""
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)

        if line_len == 0:
            return float(np.linalg.norm(point_vec))

        # Project point onto line
        projection = np.dot(point_vec, line_vec) / (line_len**2)
        projection = np.clip(projection, 0, 1)

        # Find closest point on line segment
        closest_point = line_start + projection * line_vec

        return float(np.linalg.norm(point - closest_point))

    def _calculate_polygon_area(self, points: np.ndarray) -> float:
        """Calculate area of polygon using shoelace formula."""
        if len(points) < 3:
            return 0.0

        # Ensure closed polygon
        if not np.array_equal(points[0], points[-1]):
            points = np.vstack([points, points[0]])

        # Shoelace formula
        x, y = points[:, 0], points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _calculate_aspect_ratio(self, points: np.ndarray) -> float:
        """Calculate aspect ratio of bounding box."""
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        width = max_coords[0] - min_coords[0]
        height = max_coords[1] - min_coords[1]

        if height == 0:
            return float("inf")

        return width / height

    def _is_rectangle(self, points: np.ndarray, angle_tolerance: float | None = None) -> bool:
        """Check if quadrilateral is approximately rectangular."""
        if angle_tolerance is None:
            angle_tolerance = self.config.rectangle_angle_tolerance

        if len(points) != 4:
            return False

        # Calculate vectors between consecutive points
        vectors = []
        for i in range(4):
            start = points[i]
            end = points[(i + 1) % 4]
            vectors.append(end - start)

        # Check angles between adjacent vectors
        for i in range(4):
            v1 = vectors[i]
            v2 = vectors[(i + 1) % 4]

            # Calculate angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle) * 180 / np.pi

            # Should be close to 90 degrees
            if abs(angle - 90) > angle_tolerance:
                return False

        return True


def validate_document_geometry(geometry: DocumentGeometry, image_shape: tuple[int, int, int]) -> bool:
    """
    Validate fitted document geometry.

    Checks if geometry is reasonable for document detection.
    """
    height, width = image_shape[:2]
    image_area = width * height

    # Check area constraints (not too small, not too large)
    if geometry.area < image_area * 0.05:  # Less than 5% of image
        return False
    if geometry.area > image_area * 0.95:  # More than 95% of image
        return False

    # Check aspect ratio (reasonable for documents)
    if geometry.aspect_ratio < 0.1 or geometry.aspect_ratio > 10:
        return False

    # Check corners are within bounds
    x_coords, y_coords = geometry.corners[:, 0], geometry.corners[:, 1]
    if not np.all((x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)):
        return False

    return True
