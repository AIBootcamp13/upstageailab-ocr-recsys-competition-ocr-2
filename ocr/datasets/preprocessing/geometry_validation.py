"""
Geometry Validation Utility for Document Quadrilateral Validation.

This module provides utilities to validate quadrilateral geometry before
perspective and orientation correction, ensuring robust processing with
graceful fallbacks.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class GeometryValidationConfig(BaseModel):
    """Configuration for geometry validation checks."""

    min_area_pixels: int = Field(default=1000, ge=100, description="Minimum quadrilateral area in pixels")
    max_area_ratio: float = Field(default=0.9, ge=0.1, le=1.0, description="Maximum area ratio relative to image")
    min_aspect_ratio: float = Field(default=0.1, ge=0.01, le=1.0, description="Minimum aspect ratio (width/height)")
    max_aspect_ratio: float = Field(default=10.0, ge=1.0, le=50.0, description="Maximum aspect ratio (width/height)")
    max_skew_angle: float = Field(default=45.0, ge=0.0, le=90.0, description="Maximum skew angle in degrees")
    min_side_length: int = Field(default=20, ge=5, description="Minimum side length in pixels")
    enable_bounds_check: bool = Field(default=True, description="Check if corners are within image bounds")
    enable_convexity_check: bool = Field(default=True, description="Check if quadrilateral is convex")
    enable_self_intersection_check: bool = Field(default=True, description="Check for self-intersecting quadrilaterals")

    @field_validator("max_aspect_ratio")
    @classmethod
    def validate_aspect_ratio_bounds(cls, v: float, info) -> float:
        """Ensure aspect ratio bounds are logical."""
        if info.data.get("min_aspect_ratio", 0.1) > v:
            raise ValueError("max_aspect_ratio must be greater than min_aspect_ratio")
        return v


class GeometryValidationResult(BaseModel):
    """Result of geometry validation with detailed feedback."""

    is_valid: bool = Field(description="Whether the geometry passes validation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Validation confidence score")
    issues: list[str] = Field(default_factory=list, description="List of validation issues found")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Geometric metrics computed")
    fallback_recommendation: str | None = Field(default=None, description="Recommended fallback action")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional validation metadata")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class GeometryValidationUtility:
    """
    Utility for validating quadrilateral geometry before processing.

    This class provides comprehensive validation of document quadrilaterals
    with detailed feedback and fallback recommendations.
    """

    def __init__(self, config: GeometryValidationConfig | None = None):
        self.config = config or GeometryValidationConfig()

    def validate_quadrilateral(self, corners: np.ndarray, image_shape: tuple[int, int]) -> GeometryValidationResult:
        """
        Validate quadrilateral geometry for processing.

        Args:
            corners: Quadrilateral corners as (4, 2) array
            image_shape: Shape of the original image (height, width)

        Returns:
            GeometryValidationResult with validation details
        """
        if not isinstance(corners, np.ndarray) or corners.shape != (4, 2):
            return GeometryValidationResult(
                is_valid=False,
                confidence=0.0,
                issues=["Invalid corner format - must be (4, 2) numpy array"],
                fallback_recommendation="skip_processing",
            )

        issues = []
        metrics = {}
        confidence = 1.0

        # Basic geometric metrics
        try:
            area = self._calculate_quadrilateral_area(corners)
            metrics["area"] = area

            if area < self.config.min_area_pixels:
                issues.append(f"Area too small: {area} < {self.config.min_area_pixels} pixels")
                confidence *= 0.5

            # Area ratio check
            image_area = image_shape[0] * image_shape[1]
            area_ratio = area / image_area if image_area > 0 else 0.0
            metrics["area_ratio"] = area_ratio

            if area_ratio > self.config.max_area_ratio:
                issues.append(f"Area ratio too large: {area_ratio:.3f} > {self.config.max_area_ratio}")
                confidence *= 0.7

        except Exception as e:
            issues.append(f"Area calculation failed: {str(e)}")
            confidence = 0.0

        # Aspect ratio validation
        try:
            aspect_ratio = self._calculate_aspect_ratio(corners)
            metrics["aspect_ratio"] = aspect_ratio

            if aspect_ratio < self.config.min_aspect_ratio:
                issues.append(f"Aspect ratio too small: {aspect_ratio:.3f} < {self.config.min_aspect_ratio}")
                confidence *= 0.8
            elif aspect_ratio > self.config.max_aspect_ratio:
                issues.append(f"Aspect ratio too large: {aspect_ratio:.3f} > {self.config.max_aspect_ratio}")
                confidence *= 0.8

        except Exception as e:
            issues.append(f"Aspect ratio calculation failed: {str(e)}")
            confidence *= 0.9

        # Side length validation
        try:
            side_lengths = self._calculate_side_lengths(corners)
            metrics["side_lengths"] = side_lengths.tolist()
            min_side = float(np.min(side_lengths))

            if min_side < self.config.min_side_length:
                issues.append(f"Side too short: {min_side:.1f} < {self.config.min_side_length} pixels")
                confidence *= 0.9

        except Exception as e:
            issues.append(f"Side length calculation failed: {str(e)}")
            confidence *= 0.9

        # Skew angle validation
        try:
            skew_angle = self._calculate_max_skew_angle(corners)
            metrics["max_skew_angle"] = skew_angle

            if skew_angle > self.config.max_skew_angle:
                issues.append(f"Skew angle too large: {skew_angle:.1f}° > {self.config.max_skew_angle}°")
                confidence *= 0.8

        except Exception as e:
            issues.append(f"Skew angle calculation failed: {str(e)}")
            confidence *= 0.9

        # Bounds check
        if self.config.enable_bounds_check:
            try:
                if not self._check_bounds(corners, image_shape):
                    issues.append("Corners extend outside image bounds")
                    confidence *= 0.7
            except Exception as e:
                issues.append(f"Bounds check failed: {str(e)}")
                confidence *= 0.9

        # Convexity check
        if self.config.enable_convexity_check:
            try:
                if not self._check_convexity(corners):
                    issues.append("Quadrilateral is not convex")
                    confidence *= 0.6
            except Exception as e:
                issues.append(f"Convexity check failed: {str(e)}")
                confidence *= 0.9

        # Self-intersection check
        if self.config.enable_self_intersection_check:
            try:
                if self._check_self_intersection(corners):
                    issues.append("Quadrilateral has self-intersections")
                    confidence = 0.0
            except Exception as e:
                issues.append(f"Self-intersection check failed: {str(e)}")
                confidence *= 0.8

        # Determine fallback recommendation
        fallback = self._determine_fallback_recommendation(issues, confidence)

        return GeometryValidationResult(
            is_valid=len(issues) == 0,
            confidence=confidence,
            issues=issues,
            metrics=metrics,
            fallback_recommendation=fallback,
            metadata={"config": self.config.model_dump(), "image_shape": image_shape, "corner_count": len(corners)},
        )

    def _calculate_quadrilateral_area(self, corners: np.ndarray) -> float:
        """Calculate area of quadrilateral using shoelace formula."""
        # Ensure we have exactly 4 points
        if len(corners) != 4:
            raise ValueError("Must have exactly 4 corners")

        # Shoelace formula
        x = corners[:, 0]
        y = corners[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _calculate_aspect_ratio(self, corners: np.ndarray) -> float:
        """Calculate aspect ratio (width/height) of quadrilateral."""
        # Calculate widths and heights of all sides
        widths = []
        heights = []

        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            dx = abs(p2[0] - p1[0])
            dy = abs(p2[1] - p1[1])
            widths.append(dx)
            heights.append(dy)

        avg_width = np.mean(widths)
        avg_height = np.mean(heights)

        return float(avg_width / avg_height) if avg_height > 0 else 0.0

    def _calculate_side_lengths(self, corners: np.ndarray) -> np.ndarray:
        """Calculate lengths of all four sides."""
        lengths = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            length = np.linalg.norm(p2 - p1)
            lengths.append(length)
        return np.array(lengths)

    def _calculate_max_skew_angle(self, corners: np.ndarray) -> float:
        """Calculate maximum skew angle from horizontal/vertical."""
        max_angle = 0.0

        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]

            # Calculate angle of this edge
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            if dx != 0:
                angle = abs(np.arctan(dy / dx)) * 180 / np.pi
                # Normalize to 0-45 degrees (angle from horizontal)
                angle = min(angle, 90 - angle)
                max_angle = max(max_angle, angle)

        return max_angle

    def _check_bounds(self, corners: np.ndarray, image_shape: tuple[int, int]) -> bool:
        """Check if all corners are within image bounds."""
        height, width = image_shape

        # Add small margin for floating point precision
        margin = 1.0

        x_coords = corners[:, 0]
        y_coords = corners[:, 1]

        return bool(
            np.all(x_coords >= -margin)
            and np.all(x_coords <= width + margin)
            and np.all(y_coords >= -margin)
            and np.all(y_coords <= height + margin)
        )

    def _check_convexity(self, corners: np.ndarray) -> bool:
        """Check if quadrilateral is convex."""
        # For a quadrilateral, check that all internal angles are < 180 degrees
        # This is a simplified check - a more robust implementation would use convex hull

        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Check if all turns are in the same direction (all left or all right)
        turns = []
        for i in range(4):
            o = corners[i]
            a = corners[(i + 1) % 4]
            b = corners[(i + 2) % 4]
            turns.append(cross_product(o, a, b) > 0)

        # All turns should be in the same direction for convexity
        return all(turns) or not any(turns)

    def _check_self_intersection(self, corners: np.ndarray) -> bool:
        """Check if quadrilateral has self-intersections."""

        # Check if any edges intersect
        def lines_intersect(p1, p2, p3, p4):
            def ccw(a, b, c):
                return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

        # Check all pairs of non-adjacent edges
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

        for i in range(len(edges)):
            for j in range(i + 2, len(edges)):  # Skip adjacent edges
                if i == 0 and j == len(edges) - 1:  # Skip wrap-around adjacent edges
                    continue

                e1 = edges[i]
                e2 = edges[j]

                if lines_intersect(corners[e1[0]], corners[e1[1]], corners[e2[0]], corners[e2[1]]):
                    return True

        return False

    def _determine_fallback_recommendation(self, issues: list[str], confidence: float) -> str | None:
        """Determine appropriate fallback action based on issues."""
        if not issues:
            return None

        # Critical issues that should skip processing
        critical_issues = ["Invalid corner format", "Area calculation failed", "Quadrilateral has self-intersections"]

        if any(any(critical in issue for critical in critical_issues) for issue in issues):
            return "skip_processing"

        # Moderate issues that might use alternative processing
        if confidence < 0.5:
            return "use_alternative_method"

        # Minor issues that can proceed with caution
        return "proceed_with_caution"
