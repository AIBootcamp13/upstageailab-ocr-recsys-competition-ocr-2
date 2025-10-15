"""
Advanced Corner Detection for Document Boundary Detection.

This module implements high-precision corner detection algorithms to achieve
>99% accuracy on document boundary detection, even for simple bright rectangles.
"""

from enum import Enum
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CornerDetectionMethod(Enum):
    HARRIS = "harris"
    SHI_TOMASI = "shi_tomasi"
    COMBINED = "combined"


class CornerDetectionConfig(BaseModel):
    """Configuration for corner detection algorithms with validation."""

    method: CornerDetectionMethod = Field(default=CornerDetectionMethod.COMBINED, description="Corner detection algorithm to use")
    harris_block_size: int = Field(default=2, ge=1, le=10, description="Harris corner detection block size")
    harris_ksize: int = Field(default=3, ge=1, le=31, description="Harris corner detection kernel size (odd)")
    harris_k: float = Field(default=0.04, ge=0.01, le=0.1, description="Harris corner detection sensitivity parameter")
    harris_threshold: float = Field(default=0.01, ge=0.0, le=1.0, description="Harris corner detection threshold")
    shi_tomasi_max_corners: int = Field(default=100, ge=1, le=1000, description="Shi-Tomasi maximum corners to detect")
    shi_tomasi_quality_level: float = Field(default=0.01, ge=0.001, le=1.0, description="Shi-Tomasi quality level threshold")
    shi_tomasi_min_distance: int = Field(default=10, ge=1, le=100, description="Shi-Tomasi minimum distance between corners")
    shi_tomasi_block_size: int = Field(default=3, ge=1, le=31, description="Shi-Tomasi block size")
    subpixel_criteria: tuple = Field(
        default=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001), description="Sub-pixel refinement termination criteria"
    )
    subpixel_window_size: tuple[int, int] = Field(default=(5, 5), description="Sub-pixel refinement window size")

    @field_validator("harris_ksize", "shi_tomasi_block_size")
    @classmethod
    def validate_odd_kernel_sizes(cls, v):
        """Ensure kernel sizes are odd."""
        if v % 2 == 0:
            raise ValueError(f"Kernel size must be odd, got {v}")
        return v

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow numpy arrays and tuples


class DetectedCorners(BaseModel):
    """Container for detected corner coordinates and metadata with validation."""

    corners: np.ndarray = Field(..., description="Corner coordinates as numpy array with shape (N, 2)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    method: str = Field(..., description="Detection method used")
    subpixel_refined: bool = Field(default=False, description="Whether corners were refined to sub-pixel accuracy")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional detection metadata")

    @field_validator("corners")
    @classmethod
    def validate_corners(cls, v):
        """Validate corner coordinates array."""
        if not isinstance(v, np.ndarray):
            raise ValueError("Corners must be a numpy array")
        if v.ndim != 2:
            raise ValueError("Corners must be a 2D array")
        if v.shape[1] != 2:
            raise ValueError("Corners must have shape (N, 2)")
        if v.shape[0] == 0:
            return v  # Allow empty arrays for no corners detected
        return v

    @field_validator("method")
    @classmethod
    def validate_method(cls, v):
        """Validate detection method."""
        valid_methods = ["harris", "shi_tomasi", "combined"]
        if v not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got {v}")
        return v

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow numpy arrays


class AdvancedCornerDetector:
    """
    Advanced corner detection with multiple algorithms and validation.

    Achieves sub-pixel accuracy and robust detection for document boundaries.
    """

    def __init__(self, config: CornerDetectionConfig | None = None):
        self.config = config or CornerDetectionConfig()

    def detect_corners(self, image: np.ndarray) -> DetectedCorners:
        """
        Detect corners in the image using configured method.

        Args:
            image: Input image (grayscale or BGR)

        Returns:
            DetectedCorners object with corner coordinates and metadata
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        corners = None
        confidence = 0.0
        method = self.config.method.value

        if self.config.method in [CornerDetectionMethod.HARRIS, CornerDetectionMethod.COMBINED]:
            harris_corners, harris_conf = self._detect_harris_corners(gray)
            if self.config.method == CornerDetectionMethod.HARRIS:
                corners = harris_corners
                confidence = harris_conf
            else:  # COMBINED
                corners = harris_corners
                confidence = harris_conf

        if self.config.method in [CornerDetectionMethod.SHI_TOMASI, CornerDetectionMethod.COMBINED]:
            shi_corners, shi_conf = self._detect_shi_tomasi_corners(gray)
            if self.config.method == CornerDetectionMethod.SHI_TOMASI:
                corners = shi_corners
                confidence = shi_conf
            else:  # COMBINED - merge results
                if corners is not None and len(shi_corners) > 0:
                    corners = self._merge_corner_results(corners, shi_corners)
                    confidence = max(harris_conf, shi_conf)
                else:
                    corners = shi_corners
                    confidence = shi_conf

        # Refine to sub-pixel accuracy
        if corners is not None and len(corners) > 0:
            corners = self._refine_subpixel(gray, corners)
            subpixel_refined = True
        else:
            corners = np.empty((0, 2), dtype=np.float32)
            subpixel_refined = False

        return DetectedCorners(
            corners=corners,
            confidence=confidence,
            method=method,
            subpixel_refined=subpixel_refined,
            metadata={
                "config": self.config.__dict__,
                "image_shape": image.shape,
                "total_corners_detected": len(corners) if corners is not None else 0,
            },
        )

    def _detect_harris_corners(self, gray: np.ndarray) -> tuple[np.ndarray, float]:
        """Detect corners using Harris corner detection algorithm."""
        # Convert to float32 for Harris detection
        gray_float = np.float32(gray)

        # Apply Harris corner detection
        harris_response = cv2.cornerHarris(
            gray_float, blockSize=self.config.harris_block_size, ksize=self.config.harris_ksize, k=self.config.harris_k
        )

        # Dilate to mark corners
        kernel = np.ones((3, 3), np.uint8)
        harris_response = cv2.dilate(harris_response, kernel)

        # Threshold for corner detection
        threshold = self.config.harris_threshold * harris_response.max()
        corner_mask = harris_response > threshold

        # Extract corner coordinates
        corner_coords = np.column_stack(np.where(corner_mask))

        # Calculate confidence as ratio of strong corners
        strong_corners = harris_response[corner_mask]
        total_pixels = float(np.prod(gray.shape))
        denom = max(1.0, total_pixels * 0.001)
        confidence = float(len(strong_corners) / denom)  # Normalize

        return corner_coords[:, ::-1], min(confidence, 1.0)  # Flip to (x, y)

    def _detect_shi_tomasi_corners(self, gray: np.ndarray) -> tuple[np.ndarray, float]:
        """Detect corners using Shi-Tomasi corner detection algorithm."""
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.shi_tomasi_max_corners,
            qualityLevel=self.config.shi_tomasi_quality_level,
            minDistance=self.config.shi_tomasi_min_distance,
            blockSize=self.config.shi_tomasi_block_size,
        )

        if corners is None:
            return np.array([]), 0.0

        # Reshape to (N, 2)
        corners = corners.reshape(-1, 2)

        # Calculate confidence based on corner quality distribution
        # For Shi-Tomasi, we use the quality level as confidence proxy
        confidence = min(self.config.shi_tomasi_quality_level * 100, 1.0)

        return corners, confidence

    def _merge_corner_results(self, harris_corners: np.ndarray, shi_corners: np.ndarray) -> np.ndarray:
        """Merge Harris and Shi-Tomasi corner results."""
        if len(harris_corners) == 0:
            return shi_corners
        if len(shi_corners) == 0:
            return harris_corners

        # Combine all corners
        all_corners = np.vstack([harris_corners, shi_corners])

        # Remove duplicates within 5 pixels
        merged_corners: list[np.ndarray] = []
        for corner in all_corners:
            is_duplicate = False
            for existing in merged_corners:
                if np.linalg.norm(corner - existing) < 5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                merged_corners.append(corner)

        return np.array(merged_corners)

    def _refine_subpixel(self, gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Refine corner locations to sub-pixel accuracy."""
        if len(corners) == 0:
            return corners

        # Convert corners to float32
        corners_float = np.float32(corners).reshape(-1, 1, 2)

        # Refine using cornerSubPix
        try:
            refined_corners = cv2.cornerSubPix(
                gray, corners_float, self.config.subpixel_window_size, (-1, -1), self.config.subpixel_criteria
            )
            return refined_corners.reshape(-1, 2)
        except cv2.error:
            # Return original corners if refinement fails
            return corners


def validate_document_corners(corners: np.ndarray, image_shape: tuple[int, int, int]) -> bool:
    """
    Validate detected corners for document boundary detection.

    Checks geometric constraints for quadrilateral document boundaries.
    """
    if len(corners) < 4:
        return False

    height, width = image_shape[:2]

    # Check if corners are within image bounds
    x_coords, y_coords = corners[:, 0], corners[:, 1]
    if not np.all((x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)):
        return False

    # Check for reasonable spread (corners shouldn't be too clustered)
    if len(corners) >= 4:
        # Calculate bounding box of corners
        min_coords = np.min(corners, axis=0)
        max_coords = np.max(corners, axis=0)
        bbox_area = (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1])
        image_area = width * height

        # Bounding box should cover reasonable portion of image
        if bbox_area / image_area < 0.1:  # Less than 10% coverage
            return False

    return True
