"""Enhanced Preprocessing Pipeline with Office Lens Quality Features.

This module integrates Phase 1 (advanced document detection) and Phase 2
(advanced enhancement) features into a configurable, modular pipeline.

Features:
- Advanced noise elimination with shadow removal
- Document flattening for crumpled paper
- Intelligent brightness adjustment
- Quality-based processing decisions
- Performance monitoring and logging
- Backward compatible with existing pipeline
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .advanced_noise_elimination import AdvancedNoiseEliminator, NoiseEliminationConfig, NoiseReductionMethod
from .config import DocumentPreprocessorConfig, EnhancementMethod
from .contracts import validate_image_input_with_fallback
from .document_flattening import DocumentFlattener, FlatteningConfig, FlatteningMethod
from .intelligent_brightness import BrightnessConfig, BrightnessMethod, IntelligentBrightnessAdjuster
from .pipeline import DocumentPreprocessor


class EnhancementStage(str, Enum):
    """Stages of enhancement pipeline."""

    NOISE_ELIMINATION = "noise_elimination"
    DOCUMENT_FLATTENING = "document_flattening"
    BRIGHTNESS_ADJUSTMENT = "brightness_adjustment"
    BASIC_ENHANCEMENT = "basic_enhancement"


class QualityThresholds(BaseModel):
    """Quality thresholds for processing decisions."""

    model_config = ConfigDict(strict=False)

    min_noise_elimination_effectiveness: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum noise elimination effectiveness")
    min_flattening_quality: float = Field(default=0.4, ge=0.0, le=1.0, description="Minimum flattening quality")
    min_brightness_quality: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum brightness quality")
    skip_flattening_below_curvature: float = Field(default=0.05, ge=0.0, le=1.0, description="Skip flattening if curvature below this")


class EnhancedPipelineConfig(BaseModel):
    """Configuration for enhanced preprocessing pipeline.

    Extends the base pipeline config with advanced enhancement features.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Base pipeline config
    base_config: DocumentPreprocessorConfig = Field(default_factory=DocumentPreprocessorConfig)

    # Phase 2 enhancement features
    enable_advanced_noise_elimination: bool = Field(default=False, description="Enable advanced noise elimination")
    enable_document_flattening: bool = Field(default=False, description="Enable document flattening")
    enable_intelligent_brightness: bool = Field(default=False, description="Enable intelligent brightness adjustment")

    # Enhancement configurations
    noise_elimination_config: NoiseEliminationConfig | None = Field(default=None, description="Noise elimination configuration")
    flattening_config: FlatteningConfig | None = Field(default=None, description="Document flattening configuration")
    brightness_config: BrightnessConfig | None = Field(default=None, description="Brightness adjustment configuration")

    # Quality-based processing
    enable_quality_checks: bool = Field(default=True, description="Enable quality-based processing decisions")
    quality_thresholds: QualityThresholds = Field(default_factory=QualityThresholds, description="Quality thresholds")

    # Performance monitoring
    enable_performance_logging: bool = Field(default=True, description="Enable performance monitoring and logging")
    log_stage_timing: bool = Field(default=True, description="Log timing for each processing stage")

    # Enhancement chain configuration
    enhancement_chain: list[EnhancementStage] = Field(
        default_factory=lambda: [
            EnhancementStage.NOISE_ELIMINATION,
            EnhancementStage.DOCUMENT_FLATTENING,
            EnhancementStage.BRIGHTNESS_ADJUSTMENT,
            EnhancementStage.BASIC_ENHANCEMENT,
        ],
        description="Order of enhancement stages",
    )


class ProcessingMetrics(BaseModel):
    """Performance metrics for pipeline processing."""

    model_config = ConfigDict(strict=False)

    total_time_ms: float = Field(ge=0.0)
    stage_times_ms: dict[str, float] = Field(default_factory=dict)
    stages_executed: list[str] = Field(default_factory=list)
    stages_skipped: list[str] = Field(default_factory=list)
    quality_scores: dict[str, float] = Field(default_factory=dict)


class EnhancedPreprocessingResult(BaseModel):
    """Result from enhanced preprocessing pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: np.ndarray = Field(description="Processed image")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    metrics: ProcessingMetrics = Field(description="Performance metrics")
    quality_assessment: dict[str, Any] = Field(default_factory=dict, description="Quality assessment results")


class EnhancedDocumentPreprocessor:
    """
    Enhanced document preprocessing pipeline with Office Lens quality features.

    Integrates Phase 1 (advanced detection) and Phase 2 (advanced enhancement)
    into a modular, configurable pipeline with quality-based decisions and
    performance monitoring.
    """

    noise_eliminator: AdvancedNoiseEliminator | None
    document_flattener: DocumentFlattener | None
    brightness_adjuster: IntelligentBrightnessAdjuster | None

    def __init__(self, config: EnhancedPipelineConfig | None = None):
        """Initialize enhanced preprocessor.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or EnhancedPipelineConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize base preprocessor
        self.base_preprocessor = DocumentPreprocessor(**self.config.base_config.to_dict())

        # Initialize Phase 2 enhancement components
        if self.config.enable_advanced_noise_elimination:
            self.noise_eliminator = AdvancedNoiseEliminator(self.config.noise_elimination_config or NoiseEliminationConfig())
        else:
            self.noise_eliminator = None

        if self.config.enable_document_flattening:
            self.document_flattener = DocumentFlattener(self.config.flattening_config or FlatteningConfig())
        else:
            self.document_flattener = None

        if self.config.enable_intelligent_brightness:
            self.brightness_adjuster = IntelligentBrightnessAdjuster(self.config.brightness_config or BrightnessConfig())
        else:
            self.brightness_adjuster = None

        if self.config.enable_performance_logging:
            self.logger.info(
                f"EnhancedDocumentPreprocessor initialized - "
                f"noise={self.config.enable_advanced_noise_elimination}, "
                f"flattening={self.config.enable_document_flattening}, "
                f"brightness={self.config.enable_intelligent_brightness}"
            )

    @validate_image_input_with_fallback
    def __call__(self, image: np.ndarray) -> dict[str, Any]:
        """Process image through enhanced pipeline.

        Args:
            image: Input image

        Returns:
            Dictionary with processed image, metadata, and metrics
        """
        start_time = time.time()
        stage_times: dict[str, float] = {}
        stages_executed: list[str] = []
        stages_skipped: list[str] = []
        quality_scores: dict[str, float] = {}

        # Run base preprocessing pipeline first (document detection, perspective correction, etc.)
        stage_start = time.time()
        base_result = self.base_preprocessor(image)
        stage_times["base_preprocessing"] = (time.time() - stage_start) * 1000
        stages_executed.append("base_preprocessing")

        current_image = base_result["image"]
        metadata = base_result.get("metadata", {})

        # Get document corners from base processing for flattening
        corners = metadata.get("document_corners")

        # Execute enhancement chain in configured order
        for stage in self.config.enhancement_chain:
            stage_start = time.time()

            if stage == EnhancementStage.NOISE_ELIMINATION and self.config.enable_advanced_noise_elimination:
                current_image, quality_score = self._apply_noise_elimination(current_image)
                quality_scores["noise_elimination"] = quality_score
                stages_executed.append("noise_elimination")

            elif stage == EnhancementStage.DOCUMENT_FLATTENING and self.config.enable_document_flattening:
                current_image, quality_score = self._apply_document_flattening(current_image, corners)
                quality_scores["document_flattening"] = quality_score
                stages_executed.append("document_flattening")

            elif stage == EnhancementStage.BRIGHTNESS_ADJUSTMENT and self.config.enable_intelligent_brightness:
                current_image, quality_score = self._apply_brightness_adjustment(current_image)
                quality_scores["brightness_adjustment"] = quality_score
                stages_executed.append("brightness_adjustment")

            elif stage == EnhancementStage.BASIC_ENHANCEMENT:
                # Basic enhancement is already done in base pipeline
                stages_skipped.append("basic_enhancement_redundant")
                continue

            else:
                stages_skipped.append(stage.value)
                continue

            stage_times[stage.value] = (time.time() - stage_start) * 1000

            if self.config.log_stage_timing:
                self.logger.debug(f"Stage {stage.value} completed in {stage_times[stage.value]:.2f}ms")

        total_time = (time.time() - start_time) * 1000

        # Build metrics
        metrics = ProcessingMetrics(
            total_time_ms=total_time,
            stage_times_ms=stage_times,
            stages_executed=stages_executed,
            stages_skipped=stages_skipped,
            quality_scores=quality_scores,
        )

        # Update metadata with enhancement info
        metadata["enhancement_stages"] = stages_executed
        metadata["enhancement_quality_scores"] = quality_scores

        if self.config.enable_performance_logging:
            self.logger.info(
                f"Enhanced preprocessing completed in {total_time:.2f}ms - stages: {len(stages_executed)}, quality_avg: {sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0:.2f}"
            )

        return {
            "image": current_image,
            "metadata": metadata,
            "metrics": metrics.model_dump(),
            "quality_assessment": quality_scores,
        }

    def _apply_noise_elimination(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Apply advanced noise elimination.

        Args:
            image: Input image

        Returns:
            Tuple of (cleaned image, quality score)
        """
        if self.noise_eliminator is None:
            return image, 0.0

        try:
            result = self.noise_eliminator.eliminate_noise(image)

            # Check quality threshold
            if self.config.enable_quality_checks:
                if result.effectiveness_score < self.config.quality_thresholds.min_noise_elimination_effectiveness:
                    self.logger.warning(
                        f"Noise elimination quality ({result.effectiveness_score:.2f}) "
                        f"below threshold ({self.config.quality_thresholds.min_noise_elimination_effectiveness:.2f}), using original"
                    )
                    return image, result.effectiveness_score

            return result.cleaned_image, result.effectiveness_score

        except Exception as e:
            self.logger.error(f"Noise elimination failed: {e}", exc_info=True)
            return image, 0.0

    def _apply_document_flattening(self, image: np.ndarray, corners: np.ndarray | None) -> tuple[np.ndarray, float]:
        """Apply document flattening.

        Args:
            image: Input image
            corners: Optional document corners

        Returns:
            Tuple of (flattened image, quality score)
        """
        if self.document_flattener is None:
            return image, 0.0

        try:
            result = self.document_flattener.flatten_document(image, corners)

            # Check if flattening was skipped (already flat)
            if "skipped_reason" in result.metadata:
                return image, 1.0  # Perfect quality - no flattening needed

            # Check quality threshold
            quality_score = result.quality_metrics.overall_quality if result.quality_metrics else 0.5

            if self.config.enable_quality_checks:
                if quality_score < self.config.quality_thresholds.min_flattening_quality:
                    self.logger.warning(
                        f"Flattening quality ({quality_score:.2f}) "
                        f"below threshold ({self.config.quality_thresholds.min_flattening_quality:.2f}), using original"
                    )
                    return image, quality_score

            return result.flattened_image, quality_score

        except Exception as e:
            self.logger.error(f"Document flattening failed: {e}", exc_info=True)
            return image, 0.0

    def _apply_brightness_adjustment(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Apply intelligent brightness adjustment.

        Args:
            image: Input image

        Returns:
            Tuple of (adjusted image, quality score)
        """
        if self.brightness_adjuster is None:
            return image, 0.0

        try:
            result = self.brightness_adjuster.adjust_brightness(image)

            quality_score = result.quality_metrics.overall_quality

            # Check quality threshold
            if self.config.enable_quality_checks:
                if quality_score < self.config.quality_thresholds.min_brightness_quality:
                    self.logger.warning(
                        f"Brightness adjustment quality ({quality_score:.2f}) "
                        f"below threshold ({self.config.quality_thresholds.min_brightness_quality:.2f}), using original"
                    )
                    return image, quality_score

            return result.adjusted_image, quality_score

        except Exception as e:
            self.logger.error(f"Brightness adjustment failed: {e}", exc_info=True)
            return image, 0.0


def create_office_lens_preprocessor(
    enable_noise_elimination: bool = True,
    enable_flattening: bool = True,
    enable_brightness: bool = True,
    noise_method: NoiseReductionMethod = NoiseReductionMethod.COMBINED,
    flattening_method: FlatteningMethod = FlatteningMethod.ADAPTIVE,
    brightness_method: BrightnessMethod = BrightnessMethod.AUTO,
) -> EnhancedDocumentPreprocessor:
    """Factory function to create Office Lens quality preprocessor.

    Args:
        enable_noise_elimination: Enable advanced noise elimination
        enable_flattening: Enable document flattening
        enable_brightness: Enable intelligent brightness adjustment
        noise_method: Noise elimination method
        flattening_method: Flattening method
        brightness_method: Brightness adjustment method

    Returns:
        Configured EnhancedDocumentPreprocessor
    """
    # Create base config with Office Lens enhancement
    base_config = DocumentPreprocessorConfig(
        enable_document_detection=True,
        enable_perspective_correction=True,
        enable_enhancement=True,
        enhancement_method=EnhancementMethod.OFFICE_LENS,  # Use Office Lens enhancement in base pipeline
        enable_final_resize=True,
        target_size=(640, 640),
    )

    # Create enhancement configs
    noise_config = NoiseEliminationConfig(method=noise_method) if enable_noise_elimination else None

    flattening_config = FlatteningConfig(method=flattening_method) if enable_flattening else None

    brightness_config = BrightnessConfig(method=brightness_method) if enable_brightness else None

    # Create pipeline config
    pipeline_config = EnhancedPipelineConfig(
        base_config=base_config,
        enable_advanced_noise_elimination=enable_noise_elimination,
        enable_document_flattening=enable_flattening,
        enable_intelligent_brightness=enable_brightness,
        noise_elimination_config=noise_config,
        flattening_config=flattening_config,
        brightness_config=brightness_config,
        enable_quality_checks=True,
        enable_performance_logging=True,
    )

    return EnhancedDocumentPreprocessor(pipeline_config)


def create_fast_preprocessor(target_size: tuple[int, int] = (640, 640)) -> EnhancedDocumentPreprocessor:
    """Factory function to create fast preprocessor (basic features only).

    Args:
        target_size: Target output size

    Returns:
        Configured EnhancedDocumentPreprocessor with basic features
    """
    base_config = DocumentPreprocessorConfig(
        enable_document_detection=True,
        enable_perspective_correction=True,
        enable_enhancement=True,
        enhancement_method=EnhancementMethod.CONSERVATIVE,
        enable_final_resize=True,
        target_size=target_size,
    )

    pipeline_config = EnhancedPipelineConfig(
        base_config=base_config,
        enable_advanced_noise_elimination=False,
        enable_document_flattening=False,
        enable_intelligent_brightness=False,
        enable_quality_checks=False,
        enable_performance_logging=False,
    )

    return EnhancedDocumentPreprocessor(pipeline_config)


__all__ = [
    "EnhancedDocumentPreprocessor",
    "EnhancedPipelineConfig",
    "EnhancementStage",
    "QualityThresholds",
    "ProcessingMetrics",
    "EnhancedPreprocessingResult",
    "create_office_lens_preprocessor",
    "create_fast_preprocessor",
]
