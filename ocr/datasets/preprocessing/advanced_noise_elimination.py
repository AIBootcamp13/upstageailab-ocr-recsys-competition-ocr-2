"""
Advanced Noise Elimination for Document Preprocessing.

This module implements Office Lens quality noise elimination with adaptive
background subtraction, shadow detection and removal, text region preservation,
and content-aware morphological operations.
"""

from enum import Enum
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class NoiseReductionMethod(Enum):
    """Available noise reduction methods."""

    ADAPTIVE_BACKGROUND = "adaptive_background"
    SHADOW_REMOVAL = "shadow_removal"
    MORPHOLOGICAL = "morphological"
    COMBINED = "combined"


class NoiseEliminationConfig(BaseModel):
    """Configuration for advanced noise elimination.

    Uses Pydantic BaseModel for validation and type safety.
    """

    method: NoiseReductionMethod = Field(default=NoiseReductionMethod.COMBINED, description="Noise reduction method to use")

    # Adaptive background subtraction parameters
    adaptive_block_size: int = Field(default=15, ge=3, description="Block size for adaptive thresholding (must be odd)")
    adaptive_c: float = Field(default=10.0, ge=0.0, description="Constant subtracted from weighted mean")

    # Shadow detection and removal parameters
    shadow_detection_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Threshold for shadow detection")
    shadow_removal_strength: float = Field(default=0.8, ge=0.0, le=1.0, description="Strength of shadow removal")

    # Text preservation parameters
    preserve_text_regions: bool = Field(default=True, description="Whether to preserve text regions during cleaning")
    text_region_dilation: int = Field(default=3, ge=1, description="Dilation kernel size for text region protection")

    # Morphological operations parameters
    morph_kernel_size: int = Field(default=3, ge=1, description="Kernel size for morphological operations")
    morph_iterations: int = Field(default=1, ge=1, description="Number of morphological operation iterations")

    # Content awareness parameters
    content_aware: bool = Field(default=True, description="Enable content-aware processing")
    edge_preservation_strength: float = Field(default=0.8, ge=0.0, le=1.0, description="Strength of edge preservation")

    @field_validator("adaptive_block_size")
    @classmethod
    def validate_block_size_odd(cls, v):
        """Ensure block size is odd."""
        if v % 2 == 0:
            raise ValueError("adaptive_block_size must be odd")
        return v

    model_config = ConfigDict(arbitrary_types_allowed=True)


class NoiseEliminationResult(BaseModel):
    """Result of noise elimination processing.

    Uses Pydantic BaseModel for validation and type safety.
    """

    cleaned_image: Any = Field(..., description="Noise-eliminated image (numpy array)")
    noise_mask: Any | None = Field(default=None, description="Binary mask of detected noise regions")
    shadow_mask: Any | None = Field(default=None, description="Binary mask of detected shadow regions")
    text_mask: Any | None = Field(default=None, description="Binary mask of detected text regions")
    effectiveness_score: float = Field(..., ge=0.0, le=1.0, description="Estimated noise elimination effectiveness (0-1)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional processing metadata")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AdvancedNoiseEliminator:
    """
    Advanced noise elimination for document preprocessing.

    Implements Office Lens quality noise reduction with:
    - Adaptive background subtraction
    - Shadow detection and removal
    - Text region preservation
    - Content-aware morphological operations
    """

    def __init__(self, config: NoiseEliminationConfig | None = None):
        """Initialize noise eliminator with configuration.

        Args:
            config: Configuration for noise elimination
        """
        self.config = config or NoiseEliminationConfig()

    def eliminate_noise(self, image: np.ndarray) -> NoiseEliminationResult:
        """
        Eliminate noise from document image.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            NoiseEliminationResult with cleaned image and metadata
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            is_color = True
        else:
            gray = image.copy()
            is_color = False

        # Initialize masks
        noise_mask = None
        shadow_mask = None
        text_mask = None

        # Apply selected noise reduction method
        if self.config.method == NoiseReductionMethod.ADAPTIVE_BACKGROUND:
            cleaned, noise_mask = self._adaptive_background_subtraction(gray)

        elif self.config.method == NoiseReductionMethod.SHADOW_REMOVAL:
            cleaned, shadow_mask = self._shadow_detection_and_removal(gray)

        elif self.config.method == NoiseReductionMethod.MORPHOLOGICAL:
            cleaned = self._morphological_cleaning(gray)

        else:  # COMBINED
            # Detect text regions first if preservation is enabled
            if self.config.preserve_text_regions:
                text_mask = self._detect_text_regions(gray)

            # Apply adaptive background subtraction
            cleaned, noise_mask = self._adaptive_background_subtraction(gray)

            # Apply shadow removal
            cleaned, shadow_mask = self._shadow_detection_and_removal(cleaned)

            # Apply morphological cleaning with text preservation
            if text_mask is not None:
                cleaned = self._morphological_cleaning_with_preservation(cleaned, text_mask)
            else:
                cleaned = self._morphological_cleaning(cleaned)

        # Convert back to color if needed
        if is_color:
            cleaned = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

        # Calculate effectiveness score
        effectiveness_score = self._calculate_effectiveness(
            image if not is_color else gray, cleaned if not is_color else cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        )

        # Build metadata
        metadata = {
            "method": self.config.method.value,
            "was_color": is_color,
            "original_shape": image.shape,
            "config": self.config.model_dump(),
        }

        return NoiseEliminationResult(
            cleaned_image=cleaned,
            noise_mask=noise_mask,
            shadow_mask=shadow_mask,
            text_mask=text_mask,
            effectiveness_score=effectiveness_score,
            metadata=metadata,
        )

    def _adaptive_background_subtraction(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply adaptive background subtraction.

        Args:
            gray: Grayscale input image

        Returns:
            Tuple of (cleaned image, noise mask)
        """
        # Apply adaptive thresholding to identify foreground
        cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.config.adaptive_block_size, self.config.adaptive_c
        )

        # Estimate background
        kernel_size = max(31, self.config.adaptive_block_size * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # Subtract background
        diff = cv2.absdiff(gray, background)

        # Threshold the difference to get noise mask
        _, noise_mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Clean the image by removing noise
        cleaned = gray.copy()
        cleaned[noise_mask > 0] = background[noise_mask > 0]

        return cleaned, noise_mask

    def _shadow_detection_and_removal(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect and remove shadows from image.

        Args:
            gray: Grayscale input image

        Returns:
            Tuple of (cleaned image, shadow mask)
        """
        # Detect shadows using illumination analysis
        # Apply morphological opening to estimate illumination
        kernel_size = 25
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Estimate illumination using morphological opening
        illumination = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        # Calculate illumination ratio
        illumination_float = illumination.astype(np.float32) + 1e-6
        ratio = gray.astype(np.float32) / illumination_float

        # Detect shadows (regions with low ratio)
        threshold = self.config.shadow_detection_threshold
        shadow_mask = (ratio < threshold).astype(np.uint8) * 255

        # Remove small shadow regions (noise)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel_small)

        # Remove shadows by correcting illumination
        cleaned = gray.copy().astype(np.float32)

        if np.any(shadow_mask > 0):
            # Calculate correction factor
            shadow_regions = cleaned[shadow_mask > 0]
            non_shadow_regions = cleaned[shadow_mask == 0]

            if len(non_shadow_regions) > 0:
                target_mean = np.mean(non_shadow_regions)
                shadow_mean = np.mean(shadow_regions)

                if shadow_mean > 0:
                    correction = target_mean / shadow_mean
                    correction = min(correction, 2.0)  # Limit correction

                    # Apply correction with strength parameter
                    strength = self.config.shadow_removal_strength
                    cleaned[shadow_mask > 0] *= 1 + (correction - 1) * strength

        cleaned = np.clip(cleaned, 0, 255).astype(np.uint8)

        return cleaned, shadow_mask

    def _detect_text_regions(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect text regions for preservation during cleaning.

        Args:
            gray: Grayscale input image

        Returns:
            Binary mask of text regions
        """
        # Apply edge detection to find text
        edges = cv2.Canny(gray, 50, 150)

        # Dilate to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.config.text_region_dilation, self.config.text_region_dilation))
        text_mask = cv2.dilate(edges, kernel, iterations=2)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_mask, connectivity=8)

        # Filter components by aspect ratio and size (typical for text)
        filtered_mask = np.zeros_like(text_mask)

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            # Text regions typically have certain aspect ratios
            if h > 0:
                aspect_ratio = w / h

                # Keep regions that look like text
                if 0.1 <= aspect_ratio <= 20 and area > 50:
                    filtered_mask[labels == i] = 255

        return filtered_mask

    def _morphological_cleaning(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations for noise cleaning.

        Args:
            gray: Grayscale input image

        Returns:
            Cleaned image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.config.morph_kernel_size, self.config.morph_kernel_size))

        # Apply opening (erosion followed by dilation) to remove noise
        cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=self.config.morph_iterations)

        # Apply closing (dilation followed by erosion) to fill small holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=self.config.morph_iterations)

        return cleaned

    def _morphological_cleaning_with_preservation(self, gray: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological cleaning while preserving text regions.

        Args:
            gray: Grayscale input image
            text_mask: Binary mask of text regions to preserve

        Returns:
            Cleaned image with preserved text
        """
        # Clean the image
        cleaned = self._morphological_cleaning(gray)

        # Restore text regions from original
        cleaned[text_mask > 0] = gray[text_mask > 0]

        return cleaned

    def _calculate_effectiveness(self, original: np.ndarray, cleaned: np.ndarray) -> float:
        """
        Calculate noise elimination effectiveness score.

        Args:
            original: Original grayscale image
            cleaned: Cleaned grayscale image

        Returns:
            Effectiveness score (0-1)
        """
        # Calculate noise reduction using variance
        np.var(original.astype(np.float32))
        np.var(cleaned.astype(np.float32))

        # Calculate smoothness improvement
        original_laplacian = cv2.Laplacian(original, cv2.CV_64F)
        cleaned_laplacian = cv2.Laplacian(cleaned, cv2.CV_64F)

        original_noise = np.var(original_laplacian)
        cleaned_noise = np.var(cleaned_laplacian)

        # Combine metrics
        if original_noise > 0:
            noise_reduction = max(0, 1 - (cleaned_noise / original_noise))
        else:
            noise_reduction = 0.0

        # Calculate contrast preservation
        original_contrast = np.std(original.astype(np.float32))
        cleaned_contrast = np.std(cleaned.astype(np.float32))

        if original_contrast > 0:
            contrast_preservation = min(1.0, cleaned_contrast / original_contrast)
        else:
            contrast_preservation = 1.0

        # Weighted combination
        effectiveness = 0.6 * noise_reduction + 0.4 * contrast_preservation

        return max(0.0, min(1.0, effectiveness))


def validate_noise_elimination_result(result: NoiseEliminationResult, min_effectiveness: float = 0.5) -> bool:
    """
    Validate noise elimination result.

    Args:
        result: Noise elimination result to validate
        min_effectiveness: Minimum required effectiveness score

    Returns:
        True if result is valid and meets effectiveness threshold
    """
    # Check that cleaned image exists
    if result.cleaned_image is None:
        return False

    # Check effectiveness threshold
    if result.effectiveness_score < min_effectiveness:
        return False

    # Check image shape consistency
    if result.noise_mask is not None:
        if result.cleaned_image.shape[:2] != result.noise_mask.shape[:2]:
            return False

    return True
