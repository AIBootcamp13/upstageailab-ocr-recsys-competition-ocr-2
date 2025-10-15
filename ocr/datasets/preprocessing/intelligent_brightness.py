"""
Intelligent Brightness Adjustment for Document Preprocessing.

This module implements content-aware brightness correction, local contrast enhancement,
histogram equalization with document constraints, and adaptive gamma correction to achieve
Office Lens quality document preprocessing.

Data Contracts:
- Uses Pydantic V2 BaseModel for all data structures
- Follows preprocessing-data-contracts.md standards
- Provides type-safe validation for all inputs and outputs
"""

import logging
from enum import Enum
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


class BrightnessMethod(str, Enum):
    """Available brightness adjustment methods."""

    ADAPTIVE_HISTOGRAM = "adaptive_histogram"
    GAMMA_CORRECTION = "gamma_correction"
    CLAHE = "clahe"
    CONTENT_AWARE = "content_aware"
    AUTO = "auto"


class BrightnessConfig(BaseModel):
    """Configuration for intelligent brightness adjustment.

    Attributes:
        method: Brightness adjustment method to use
        clahe_clip_limit: Clipping limit for CLAHE (1.0-5.0)
        clahe_tile_size: Tile grid size for CLAHE
        gamma_value: Gamma correction value (0.5-2.0, default 1.0 is no change)
        auto_gamma: Automatically estimate optimal gamma value
        preserve_text_regions: Preserve text regions during adjustment
        target_mean_brightness: Target mean brightness (0-255)
        brightness_threshold_low: Low brightness threshold for correction
        brightness_threshold_high: High brightness threshold for correction
        local_window_size: Window size for local contrast enhancement
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    method: BrightnessMethod = Field(default=BrightnessMethod.AUTO, description="Brightness adjustment method")
    clahe_clip_limit: float = Field(default=2.0, ge=1.0, le=5.0, description="CLAHE clipping limit")
    clahe_tile_size: tuple[int, int] = Field(default=(8, 8), description="CLAHE tile grid size")
    gamma_value: float = Field(default=1.0, ge=0.5, le=2.0, description="Gamma correction value")
    auto_gamma: bool = Field(default=True, description="Automatically estimate optimal gamma")
    preserve_text_regions: bool = Field(default=True, description="Preserve text regions during adjustment")
    target_mean_brightness: float = Field(default=180.0, ge=0.0, le=255.0, description="Target mean brightness")
    brightness_threshold_low: float = Field(default=100.0, ge=0.0, le=255.0, description="Low brightness threshold")
    brightness_threshold_high: float = Field(default=200.0, ge=0.0, le=255.0, description="High brightness threshold")
    local_window_size: int = Field(default=64, ge=8, le=256, description="Local contrast window size")

    @field_validator("clahe_tile_size")
    @classmethod
    def validate_tile_size(cls, v: tuple[int, int]) -> tuple[int, int]:
        """Validate CLAHE tile size."""
        if len(v) != 2:
            raise ValueError("Tile size must be a tuple of 2 integers")
        if v[0] < 1 or v[1] < 1:
            raise ValueError("Tile size dimensions must be >= 1")
        if v[0] > 32 or v[1] > 32:
            raise ValueError("Tile size dimensions must be <= 32")
        return v

    @field_validator("brightness_threshold_high")
    @classmethod
    def validate_thresholds(cls, v: float, info) -> float:
        """Validate brightness thresholds."""
        if "brightness_threshold_low" in info.data:
            if v <= info.data["brightness_threshold_low"]:
                raise ValueError("brightness_threshold_high must be > brightness_threshold_low")
        return v


class BrightnessQuality(BaseModel):
    """Quality metrics for brightness adjustment.

    Attributes:
        contrast_score: Contrast quality score (0-1)
        brightness_uniformity: Brightness uniformity score (0-1)
        histogram_spread: Histogram spread score (0-1)
        text_preservation_score: Text preservation score (0-1)
        overall_quality: Overall quality score (0-1)
    """

    contrast_score: float = Field(ge=0.0, le=1.0)
    brightness_uniformity: float = Field(ge=0.0, le=1.0)
    histogram_spread: float = Field(ge=0.0, le=1.0)
    text_preservation_score: float = Field(ge=0.0, le=1.0)
    overall_quality: float = Field(ge=0.0, le=1.0)


class BrightnessResult(BaseModel):
    """Result of brightness adjustment operation.

    Attributes:
        adjusted_image: Brightness-adjusted image
        method_used: Method that was used
        gamma_value: Gamma value that was applied
        quality_metrics: Quality assessment metrics
        processing_time_ms: Processing time in milliseconds
        metadata: Additional processing metadata
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    adjusted_image: np.ndarray = Field(description="Brightness-adjusted image")
    method_used: BrightnessMethod = Field(description="Method used")
    gamma_value: float | None = Field(default=None, description="Applied gamma value")
    quality_metrics: BrightnessQuality = Field(description="Quality metrics")
    processing_time_ms: float = Field(ge=0.0, description="Processing time")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("adjusted_image")
    @classmethod
    def validate_image(cls, v: np.ndarray) -> np.ndarray:
        """Validate adjusted image."""
        if not isinstance(v, np.ndarray):
            raise ValueError("adjusted_image must be a numpy array")
        if v.size == 0:
            raise ValueError("adjusted_image cannot be empty")
        if v.dtype != np.uint8:
            raise ValueError("adjusted_image must be uint8 type")
        return v


class IntelligentBrightnessAdjuster:
    """
    Intelligent brightness adjustment for document preprocessing.

    Implements multiple brightness correction methods with automatic method selection,
    content-aware processing, and quality assessment.
    """

    def __init__(self, config: BrightnessConfig | None = None):
        """Initialize brightness adjuster.

        Args:
            config: Brightness adjustment configuration
        """
        self.config = config or BrightnessConfig()
        logger.info(f"Initialized IntelligentBrightnessAdjuster with method: {self.config.method}")

    def adjust_brightness(self, image: np.ndarray, config: BrightnessConfig | None = None) -> BrightnessResult:
        """Adjust image brightness using configured method.

        Args:
            image: Input image (grayscale or BGR)
            config: Optional override configuration

        Returns:
            BrightnessResult with adjusted image and metrics
        """
        import time

        start_time = time.time()

        cfg = config or self.config

        # Validate input
        if image.size == 0:
            raise ValueError("Input image is empty")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            is_color = True
        else:
            gray = image.copy()
            is_color = False

        # Select method
        method = cfg.method
        if method == BrightnessMethod.AUTO:
            method = self._select_best_method(gray, cfg)

        # Apply brightness adjustment
        if method == BrightnessMethod.CLAHE:
            adjusted = self._apply_clahe(gray, cfg)
        elif method == BrightnessMethod.GAMMA_CORRECTION:
            adjusted = self._apply_gamma_correction(gray, cfg)
        elif method == BrightnessMethod.ADAPTIVE_HISTOGRAM:
            adjusted = self._apply_adaptive_histogram(gray, cfg)
        elif method == BrightnessMethod.CONTENT_AWARE:
            adjusted = self._apply_content_aware(gray, cfg)
        else:
            adjusted = gray.copy()

        # Convert back to color if needed
        if is_color:
            adjusted = cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR)

        # Calculate quality metrics
        quality = self._calculate_quality(gray, adjusted if not is_color else cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY))

        processing_time = (time.time() - start_time) * 1000

        # Determine gamma value used
        gamma_used = None
        if method == BrightnessMethod.GAMMA_CORRECTION:
            gamma_used = cfg.gamma_value if not cfg.auto_gamma else self._estimate_gamma(gray)

        return BrightnessResult(
            adjusted_image=adjusted,
            method_used=method,
            gamma_value=gamma_used,
            quality_metrics=quality,
            processing_time_ms=processing_time,
            metadata={
                "input_shape": image.shape,
                "input_dtype": str(image.dtype),
                "mean_brightness_before": float(np.mean(gray)),
                "mean_brightness_after": float(np.mean(adjusted if not is_color else cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY))),
                "method_selected": method.value,
            },
        )

    def _select_best_method(self, gray: np.ndarray, config: BrightnessConfig) -> BrightnessMethod:
        """Automatically select the best brightness adjustment method.

        Args:
            gray: Grayscale input image
            config: Configuration

        Returns:
            Selected brightness method
        """
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        # Very low brightness -> gamma correction
        if mean_brightness < config.brightness_threshold_low:
            logger.debug(f"Selected GAMMA_CORRECTION (mean brightness: {mean_brightness:.1f})")
            return BrightnessMethod.GAMMA_CORRECTION

        # Very high brightness -> inverse gamma or CLAHE
        if mean_brightness > config.brightness_threshold_high:
            logger.debug(f"Selected CLAHE (mean brightness: {mean_brightness:.1f})")
            return BrightnessMethod.CLAHE

        # Low contrast -> CLAHE
        if std_brightness < 30:
            logger.debug(f"Selected CLAHE (low contrast, std: {std_brightness:.1f})")
            return BrightnessMethod.CLAHE

        # Uneven lighting -> content-aware
        local_variance = self._calculate_local_variance(gray, config.local_window_size)
        if np.std(local_variance) > 500:
            logger.debug(f"Selected CONTENT_AWARE (uneven lighting, variance std: {np.std(local_variance):.1f})")
            return BrightnessMethod.CONTENT_AWARE

        # Default to adaptive histogram
        logger.debug("Selected ADAPTIVE_HISTOGRAM (default)")
        return BrightnessMethod.ADAPTIVE_HISTOGRAM

    def _apply_clahe(self, gray: np.ndarray, config: BrightnessConfig) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            gray: Grayscale input image
            config: Configuration

        Returns:
            CLAHE-adjusted image
        """
        clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit, tileGridSize=config.clahe_tile_size)
        return clahe.apply(gray)

    def _apply_gamma_correction(self, gray: np.ndarray, config: BrightnessConfig) -> np.ndarray:
        """Apply gamma correction.

        Args:
            gray: Grayscale input image
            config: Configuration

        Returns:
            Gamma-corrected image
        """
        if config.auto_gamma:
            gamma = self._estimate_gamma(gray)
        else:
            gamma = config.gamma_value

        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)

        return cv2.LUT(gray, table)

    def _estimate_gamma(self, gray: np.ndarray) -> float:
        """Estimate optimal gamma value for the image.

        Args:
            gray: Grayscale input image

        Returns:
            Estimated gamma value

        Note:
            Gamma correction formula: output = (input/255)^(1/gamma) * 255
            - gamma < 1.0: darkens the image (inv_gamma > 1.0, higher exponent)
            - gamma > 1.0: brightens the image (inv_gamma < 1.0, lower exponent)
            - gamma = 1.0: no change
        """
        mean_brightness = np.mean(gray)
        target_brightness = 180.0  # Target for document images

        # Calculate gamma that would move mean to target
        # Formula: target/255 = (mean/255)^(1/gamma)
        # Solving: gamma = log(mean/255) / log(target/255)

        if mean_brightness < 10:
            return 2.0  # Very dark -> brighten significantly (gamma > 1.0)
        if mean_brightness > 245:
            return 0.5  # Very bright -> darken (gamma < 1.0)

        # Calculate optimal gamma
        gamma = np.log(mean_brightness / 255.0) / np.log(target_brightness / 255.0)

        # Clamp to reasonable range
        gamma = np.clip(gamma, 0.5, 2.0)

        return float(gamma)

    def _apply_adaptive_histogram(self, gray: np.ndarray, config: BrightnessConfig) -> np.ndarray:
        """Apply adaptive histogram equalization.

        Args:
            gray: Grayscale input image
            config: Configuration

        Returns:
            Adaptively equalized image
        """
        # Simple histogram equalization
        equalized = cv2.equalizeHist(gray)

        # Blend with original based on contrast
        std_original = np.std(gray)
        if std_original < 30:
            # Low contrast -> more equalization
            alpha = 0.8
        else:
            # Good contrast -> subtle equalization
            alpha = 0.3

        result = cv2.addWeighted(gray, 1 - alpha, equalized, alpha, 0)
        return result.astype(np.uint8)

    def _apply_content_aware(self, gray: np.ndarray, config: BrightnessConfig) -> np.ndarray:
        """Apply content-aware brightness adjustment.

        Args:
            gray: Grayscale input image
            config: Configuration

        Returns:
            Content-aware adjusted image
        """
        # Estimate local illumination
        kernel_size = config.local_window_size
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Gaussian blur to estimate illumination
        illumination = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # Subtract illumination and rescale
        # Avoid division by zero
        illumination = np.clip(illumination, 10, 255)

        # Normalize by local illumination
        normalized = (gray.astype(np.float32) / illumination.astype(np.float32)) * 180.0
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        # Apply mild CLAHE to enhance local contrast
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        result = clahe.apply(normalized)

        return result

    def _calculate_local_variance(self, gray: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate local variance map.

        Args:
            gray: Grayscale input image
            window_size: Window size for local variance calculation

        Returns:
            Local variance map
        """
        # Use sliding window to calculate local variance
        kernel = np.ones((window_size, window_size)) / (window_size**2)

        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        mean_sq = cv2.filter2D((gray.astype(np.float32) ** 2), -1, kernel)

        variance = mean_sq - (mean**2)
        variance = np.clip(variance, 0, None)

        return variance

    def _calculate_quality(self, original: np.ndarray, adjusted: np.ndarray) -> BrightnessQuality:
        """Calculate quality metrics for brightness adjustment.

        Args:
            original: Original grayscale image
            adjusted: Adjusted grayscale image

        Returns:
            Quality metrics
        """
        # Contrast score (higher std = better contrast)
        std_adjusted = np.std(adjusted)
        contrast_score = min(std_adjusted / 80.0, 1.0)  # Normalize by target std

        # Brightness uniformity (lower variance of local means = more uniform)
        local_means = self._calculate_local_variance(adjusted, 32)
        uniformity_score = max(0.0, 1.0 - np.std(local_means) / 1000.0)

        # Histogram spread (how well distributed the histogram is)
        hist = cv2.calcHist([adjusted], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        # Entropy as measure of spread
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        histogram_score = min(entropy / 8.0, 1.0)  # Normalize by max entropy

        # Text preservation (edge strength)
        edges_original = cv2.Canny(original, 50, 150)
        edges_adjusted = cv2.Canny(adjusted, 50, 150)
        edge_ratio = np.sum(edges_adjusted) / (np.sum(edges_original) + 1)
        text_preservation = min(edge_ratio, 1.0)

        # Overall quality (weighted average)
        overall = 0.3 * contrast_score + 0.2 * uniformity_score + 0.3 * histogram_score + 0.2 * text_preservation

        return BrightnessQuality(
            contrast_score=float(contrast_score),
            brightness_uniformity=float(uniformity_score),
            histogram_spread=float(histogram_score),
            text_preservation_score=float(text_preservation),
            overall_quality=float(overall),
        )


def create_brightness_adjuster(config: BrightnessConfig | None = None) -> IntelligentBrightnessAdjuster:
    """Factory function to create brightness adjuster.

    Args:
        config: Optional configuration

    Returns:
        Configured IntelligentBrightnessAdjuster instance
    """
    return IntelligentBrightnessAdjuster(config)
