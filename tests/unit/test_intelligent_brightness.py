"""
Tests for Intelligent Brightness Adjustment module.
"""

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from ocr.datasets.preprocessing.intelligent_brightness import (
    BrightnessConfig,
    BrightnessMethod,
    BrightnessQuality,
    BrightnessResult,
    IntelligentBrightnessAdjuster,
    create_brightness_adjuster,
)


class TestBrightnessConfig:
    """Test BrightnessConfig data model."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = BrightnessConfig()

        assert config.method == BrightnessMethod.AUTO
        assert config.clahe_clip_limit == 2.0
        assert config.clahe_tile_size == (8, 8)
        assert config.gamma_value == 1.0
        assert config.auto_gamma is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = BrightnessConfig(method=BrightnessMethod.CLAHE, clahe_clip_limit=3.0, gamma_value=1.5)

        assert config.method == BrightnessMethod.CLAHE
        assert config.clahe_clip_limit == 3.0
        assert config.gamma_value == 1.5

    def test_validation_clahe_clip_limit(self):
        """Test validation of CLAHE clip limit."""
        # Valid range
        config = BrightnessConfig(clahe_clip_limit=2.0)
        assert config.clahe_clip_limit == 2.0

        # Too low
        with pytest.raises(ValidationError):
            BrightnessConfig(clahe_clip_limit=0.5)

        # Too high
        with pytest.raises(ValidationError):
            BrightnessConfig(clahe_clip_limit=6.0)

    def test_validation_gamma_value(self):
        """Test validation of gamma value."""
        # Valid range
        config = BrightnessConfig(gamma_value=1.5)
        assert config.gamma_value == 1.5

        # Too low
        with pytest.raises(ValidationError):
            BrightnessConfig(gamma_value=0.3)

        # Too high
        with pytest.raises(ValidationError):
            BrightnessConfig(gamma_value=3.0)

    def test_validation_tile_size(self):
        """Test validation of tile size."""
        # Valid
        config = BrightnessConfig(clahe_tile_size=(8, 8))
        assert config.clahe_tile_size == (8, 8)

        # Invalid - not tuple of 2
        with pytest.raises(ValidationError):
            BrightnessConfig(clahe_tile_size=(8,))

        # Invalid - negative values
        with pytest.raises(ValidationError):
            BrightnessConfig(clahe_tile_size=(0, 8))

        # Invalid - too large
        with pytest.raises(ValidationError):
            BrightnessConfig(clahe_tile_size=(64, 64))

    def test_validation_brightness_thresholds(self):
        """Test validation of brightness thresholds."""
        # Valid
        config = BrightnessConfig(brightness_threshold_low=100.0, brightness_threshold_high=200.0)
        assert config.brightness_threshold_low == 100.0
        assert config.brightness_threshold_high == 200.0

        # Invalid - high <= low
        with pytest.raises(ValidationError):
            BrightnessConfig(brightness_threshold_low=150.0, brightness_threshold_high=150.0)


class TestBrightnessResult:
    """Test BrightnessResult data model."""

    def test_valid_result(self):
        """Test valid brightness result creation."""
        adjusted = np.ones((100, 100), dtype=np.uint8) * 180
        quality = BrightnessQuality(
            contrast_score=0.8, brightness_uniformity=0.7, histogram_spread=0.75, text_preservation_score=0.85, overall_quality=0.78
        )

        result = BrightnessResult(
            adjusted_image=adjusted,
            method_used=BrightnessMethod.CLAHE,
            gamma_value=None,
            quality_metrics=quality,
            processing_time_ms=50.0,
            metadata={"test": "value"},
        )

        assert result.adjusted_image.shape == (100, 100)
        assert result.method_used == BrightnessMethod.CLAHE
        assert result.quality_metrics.overall_quality == 0.78

    def test_invalid_image_type(self):
        """Test validation of image type."""
        quality = BrightnessQuality(
            contrast_score=0.8, brightness_uniformity=0.7, histogram_spread=0.75, text_preservation_score=0.85, overall_quality=0.78
        )

        # Not a numpy array
        with pytest.raises(ValidationError):
            BrightnessResult(adjusted_image=[1, 2, 3], method_used=BrightnessMethod.CLAHE, quality_metrics=quality, processing_time_ms=50.0)

    def test_invalid_image_empty(self):
        """Test validation of empty image."""
        quality = BrightnessQuality(
            contrast_score=0.8, brightness_uniformity=0.7, histogram_spread=0.75, text_preservation_score=0.85, overall_quality=0.78
        )

        with pytest.raises(ValidationError):
            BrightnessResult(
                adjusted_image=np.array([]), method_used=BrightnessMethod.CLAHE, quality_metrics=quality, processing_time_ms=50.0
            )

    def test_invalid_image_dtype(self):
        """Test validation of image dtype."""
        quality = BrightnessQuality(
            contrast_score=0.8, brightness_uniformity=0.7, histogram_spread=0.75, text_preservation_score=0.85, overall_quality=0.78
        )

        # Wrong dtype
        with pytest.raises(ValidationError):
            BrightnessResult(
                adjusted_image=np.ones((100, 100), dtype=np.float32),
                method_used=BrightnessMethod.CLAHE,
                quality_metrics=quality,
                processing_time_ms=50.0,
            )


class TestIntelligentBrightnessAdjuster:
    """Test IntelligentBrightnessAdjuster class."""

    @pytest.fixture
    def adjuster(self):
        """Create adjuster instance."""
        return IntelligentBrightnessAdjuster()

    @pytest.fixture
    def dark_image(self):
        """Create a dark test image."""
        return np.ones((200, 300), dtype=np.uint8) * 50

    @pytest.fixture
    def bright_image(self):
        """Create a bright test image."""
        return np.ones((200, 300), dtype=np.uint8) * 220

    @pytest.fixture
    def normal_image(self):
        """Create a normal brightness test image."""
        img = np.ones((200, 300), dtype=np.uint8) * 150
        # Add some variation
        cv2.rectangle(img, (50, 50), (250, 150), 200, -1)
        return img

    @pytest.fixture
    def low_contrast_image(self):
        """Create a low contrast test image."""
        img = np.ones((200, 300), dtype=np.uint8) * 128
        # Very subtle variation
        cv2.rectangle(img, (50, 50), (250, 150), 135, -1)
        return img

    def test_adjuster_initialization(self):
        """Test adjuster initialization."""
        adjuster = IntelligentBrightnessAdjuster()
        assert adjuster.config.method == BrightnessMethod.AUTO

        config = BrightnessConfig(method=BrightnessMethod.CLAHE)
        adjuster = IntelligentBrightnessAdjuster(config)
        assert adjuster.config.method == BrightnessMethod.CLAHE

    def test_adjust_brightness_basic(self, adjuster, normal_image):
        """Test basic brightness adjustment."""
        result = adjuster.adjust_brightness(normal_image)

        assert isinstance(result, BrightnessResult)
        assert result.adjusted_image.shape == normal_image.shape
        assert result.adjusted_image.dtype == np.uint8
        assert result.processing_time_ms > 0
        assert 0 <= result.quality_metrics.overall_quality <= 1

    def test_clahe_method(self, adjuster, low_contrast_image):
        """Test CLAHE method specifically."""
        config = BrightnessConfig(method=BrightnessMethod.CLAHE)
        result = adjuster.adjust_brightness(low_contrast_image, config)

        assert result.method_used == BrightnessMethod.CLAHE
        # CLAHE should improve contrast
        original_std = np.std(low_contrast_image)
        adjusted_std = np.std(result.adjusted_image)
        assert adjusted_std > original_std

    def test_gamma_correction_dark(self, adjuster, dark_image):
        """Test gamma correction on dark image."""
        config = BrightnessConfig(method=BrightnessMethod.GAMMA_CORRECTION, auto_gamma=True)
        result = adjuster.adjust_brightness(dark_image, config)

        assert result.method_used == BrightnessMethod.GAMMA_CORRECTION
        assert result.gamma_value is not None
        # Should brighten the image
        assert np.mean(result.adjusted_image) > np.mean(dark_image)

    def test_gamma_correction_manual(self, adjuster, dark_image):
        """Test manual gamma correction."""
        # Test with gamma > 1.0 for brightening dark images
        config = BrightnessConfig(
            method=BrightnessMethod.GAMMA_CORRECTION,
            auto_gamma=False,
            gamma_value=1.5,  # gamma > 1.0 brightens
        )
        result = adjuster.adjust_brightness(dark_image, config)

        assert result.method_used == BrightnessMethod.GAMMA_CORRECTION
        assert result.gamma_value == 1.5
        # Should brighten the dark image
        assert np.mean(result.adjusted_image) > np.mean(dark_image)

        # Test with gamma < 1.0 for darkening bright images
        bright_image = np.ones((200, 300), dtype=np.uint8) * 220
        config2 = BrightnessConfig(
            method=BrightnessMethod.GAMMA_CORRECTION,
            auto_gamma=False,
            gamma_value=0.5,  # gamma < 1.0 darkens
        )
        result2 = adjuster.adjust_brightness(bright_image, config2)
        # Should darken the bright image
        assert np.mean(result2.adjusted_image) < np.mean(bright_image)

    def test_adaptive_histogram(self, adjuster, normal_image):
        """Test adaptive histogram equalization."""
        config = BrightnessConfig(method=BrightnessMethod.ADAPTIVE_HISTOGRAM)
        result = adjuster.adjust_brightness(normal_image, config)

        assert result.method_used == BrightnessMethod.ADAPTIVE_HISTOGRAM
        assert result.adjusted_image.shape == normal_image.shape

    def test_content_aware(self, adjuster, normal_image):
        """Test content-aware brightness adjustment."""
        config = BrightnessConfig(method=BrightnessMethod.CONTENT_AWARE)
        result = adjuster.adjust_brightness(normal_image, config)

        assert result.method_used == BrightnessMethod.CONTENT_AWARE
        assert result.adjusted_image.shape == normal_image.shape

    def test_auto_method_selection_dark(self, adjuster, dark_image):
        """Test automatic method selection for dark image."""
        config = BrightnessConfig(method=BrightnessMethod.AUTO, brightness_threshold_low=100.0)
        result = adjuster.adjust_brightness(dark_image, config)

        # Should select gamma correction for dark image
        assert result.method_used == BrightnessMethod.GAMMA_CORRECTION

    def test_auto_method_selection_bright(self, adjuster, bright_image):
        """Test automatic method selection for bright image."""
        config = BrightnessConfig(method=BrightnessMethod.AUTO, brightness_threshold_high=200.0)
        result = adjuster.adjust_brightness(bright_image, config)

        # Should select CLAHE for bright image
        assert result.method_used == BrightnessMethod.CLAHE

    def test_auto_method_selection_low_contrast(self, adjuster, low_contrast_image):
        """Test automatic method selection for low contrast image."""
        result = adjuster.adjust_brightness(low_contrast_image)

        # Should select CLAHE for low contrast
        assert result.method_used == BrightnessMethod.CLAHE

    def test_color_image_handling(self, adjuster):
        """Test handling of color images."""
        color_image = np.ones((200, 300, 3), dtype=np.uint8) * 100
        result = adjuster.adjust_brightness(color_image)

        assert result.adjusted_image.shape == color_image.shape
        assert len(result.adjusted_image.shape) == 3

    def test_quality_metrics(self, adjuster, normal_image):
        """Test quality metrics calculation."""
        result = adjuster.adjust_brightness(normal_image)

        metrics = result.quality_metrics
        assert 0 <= metrics.contrast_score <= 1
        assert 0 <= metrics.brightness_uniformity <= 1
        assert 0 <= metrics.histogram_spread <= 1
        assert 0 <= metrics.text_preservation_score <= 1
        assert 0 <= metrics.overall_quality <= 1

    def test_metadata(self, adjuster, normal_image):
        """Test result metadata."""
        result = adjuster.adjust_brightness(normal_image)

        assert "input_shape" in result.metadata
        assert "mean_brightness_before" in result.metadata
        assert "mean_brightness_after" in result.metadata
        assert "method_selected" in result.metadata

    def test_gamma_estimation(self, adjuster):
        """Test gamma estimation."""
        # Very dark image
        dark = np.ones((100, 100), dtype=np.uint8) * 20
        gamma_dark = adjuster._estimate_gamma(dark)
        assert gamma_dark > 1.0  # Should brighten (gamma > 1.0)

        # Very bright image
        bright = np.ones((100, 100), dtype=np.uint8) * 240
        gamma_bright = adjuster._estimate_gamma(bright)
        assert gamma_bright < 1.0  # Should darken (gamma < 1.0)

        # Normal image
        normal = np.ones((100, 100), dtype=np.uint8) * 180
        gamma_normal = adjuster._estimate_gamma(normal)
        assert 0.9 <= gamma_normal <= 1.1  # Should be close to 1.0

    def test_local_variance_calculation(self, adjuster):
        """Test local variance calculation."""
        img = np.random.randint(0, 256, (200, 300), dtype=np.uint8)
        variance = adjuster._calculate_local_variance(img, 32)

        assert variance.shape == img.shape
        assert variance.dtype == np.float32
        assert np.all(variance >= 0)

    def test_empty_image_error(self, adjuster):
        """Test error handling for empty image."""
        empty = np.array([])
        with pytest.raises(ValueError, match="Input image is empty"):
            adjuster.adjust_brightness(empty)

    def test_processing_time(self, adjuster, normal_image):
        """Test that processing time is recorded."""
        result = adjuster.adjust_brightness(normal_image)
        assert result.processing_time_ms > 0
        assert result.processing_time_ms < 10000  # Should be fast


class TestFactoryFunction:
    """Test factory function."""

    def test_create_brightness_adjuster_default(self):
        """Test factory function with default config."""
        adjuster = create_brightness_adjuster()
        assert isinstance(adjuster, IntelligentBrightnessAdjuster)
        assert adjuster.config.method == BrightnessMethod.AUTO

    def test_create_brightness_adjuster_custom(self):
        """Test factory function with custom config."""
        config = BrightnessConfig(method=BrightnessMethod.CLAHE)
        adjuster = create_brightness_adjuster(config)
        assert isinstance(adjuster, IntelligentBrightnessAdjuster)
        assert adjuster.config.method == BrightnessMethod.CLAHE


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline_dark_document(self):
        """Test full pipeline on dark document."""
        # Create dark document with text
        img = np.ones((400, 600), dtype=np.uint8) * 60
        cv2.putText(img, "Dark Document", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, 200, 3)

        adjuster = create_brightness_adjuster()
        result = adjuster.adjust_brightness(img)

        # Should improve brightness
        assert np.mean(result.adjusted_image) > np.mean(img)
        # Should maintain reasonable quality
        assert result.quality_metrics.overall_quality > 0.3

    def test_full_pipeline_uneven_lighting(self):
        """Test full pipeline on image with uneven lighting."""
        # Create image with gradient lighting
        img = np.zeros((400, 600), dtype=np.uint8)
        for i in range(400):
            img[i, :] = int(50 + (i / 400) * 150)

        adjuster = create_brightness_adjuster()
        result = adjuster.adjust_brightness(img)

        # Should select content-aware or similar method
        assert result.method_used in [BrightnessMethod.CONTENT_AWARE, BrightnessMethod.CLAHE, BrightnessMethod.ADAPTIVE_HISTOGRAM]

    def test_full_pipeline_low_contrast(self):
        """Test full pipeline on low contrast document."""
        # Create low contrast document
        img = np.ones((400, 600), dtype=np.uint8) * 128
        cv2.rectangle(img, (100, 100), (500, 300), 140, -1)

        adjuster = create_brightness_adjuster()
        result = adjuster.adjust_brightness(img)

        # Should improve contrast
        assert np.std(result.adjusted_image) > np.std(img)
        # Likely selected CLAHE
        assert result.method_used in [BrightnessMethod.CLAHE, BrightnessMethod.ADAPTIVE_HISTOGRAM]
