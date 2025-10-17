"""
Enhanced preprocessing pipeline with selective stage execution for Streamlit Preprocessing Viewer.

This module extends the basic pipeline to support selective execution of individual stages,
performance monitoring, and GPU acceleration capabilities.
"""

import logging
import time
from typing import Any

import cv2
import numpy as np

# Import preprocessing classes
from ocr.datasets.preprocessing.advanced_corner_detection import AdvancedCornerDetector
from ocr.datasets.preprocessing.advanced_noise_elimination import AdvancedNoiseEliminator
from ocr.datasets.preprocessing.corner_selection import CornerSelectionUtility
from ocr.datasets.preprocessing.detector import DocumentDetector
from ocr.datasets.preprocessing.document_flattening import DocumentFlattener
from ocr.datasets.preprocessing.enhancement import ImageEnhancer
from ocr.datasets.preprocessing.geometry_validation import GeometryValidationUtility
from ocr.datasets.preprocessing.intelligent_brightness import IntelligentBrightnessAdjuster
from ocr.datasets.preprocessing.orientation import OrientationCorrector
from ocr.datasets.preprocessing.perspective import PerspectiveCorrector
from ocr.datasets.preprocessing.telemetry import PreprocessingTelemetry

# Import viewer contracts for validation
from ..viewer_contracts import validate_roi_request, validate_viewer_config


class PipelinePerformanceMonitor:
    """Monitor performance metrics for pipeline stages."""

    def __init__(self):
        self.stage_times: dict[str, float] = {}
        self.stage_memory: dict[str, int] = {}
        self.total_time: float = 0.0

    def start_stage(self, stage_name: str):
        """Start timing a pipeline stage."""
        self.stage_times[stage_name] = time.time()

    def end_stage(self, stage_name: str):
        """End timing a pipeline stage."""
        if stage_name in self.stage_times:
            duration = time.time() - self.stage_times[stage_name]
            self.stage_times[stage_name] = duration
            return duration
        return 0.0

    def get_stage_metrics(self) -> dict[str, Any]:
        """Get performance metrics for all stages."""
        return {"stage_times": self.stage_times.copy(), "total_time": sum(self.stage_times.values()), "stage_count": len(self.stage_times)}


class EnhancedPreprocessingPipeline:
    """
    Enhanced pipeline orchestrator with selective stage execution and performance monitoring.

    This class extends the basic pipeline to support:
    - Selective execution of individual pipeline stages
    - Performance monitoring and metrics collection
    - GPU acceleration hooks
    - Intermediate result caching
    """

    def __init__(self, logger: logging.Logger | None = None, use_gpu: bool = False):
        self.logger = logger or logging.getLogger(__name__)
        self.use_gpu = use_gpu
        self.performance_monitor = PipelinePerformanceMonitor()

        # Initialize preprocessing components
        self._init_components()

        # Cache for intermediate results
        self._result_cache: dict[str, Any] = {}

    def _init_components(self):
        """Initialize all preprocessing components."""
        self.corner_detector = AdvancedCornerDetector()
        self.corner_selector = CornerSelectionUtility()
        self.geometry_validator = GeometryValidationUtility()
        self.telemetry = PreprocessingTelemetry(self.logger)
        self.document_flattener = DocumentFlattener()
        self.noise_eliminator = AdvancedNoiseEliminator()
        self.brightness_adjuster = IntelligentBrightnessAdjuster()
        self.image_enhancer = ImageEnhancer()

        # Initialize correctors with proper parameters
        def ensure_doctr(feature: str) -> bool:
            return True  # For viewer, assume DOCTR is available

        self.perspective_corrector = PerspectiveCorrector(
            logger=self.logger, ensure_doctr=ensure_doctr, use_doctr_geometry=False, doctr_assume_horizontal=False
        )

        # Create a mock detector for orientation corrector
        class MockDetector(DocumentDetector):
            def __init__(self, logger):
                # Initialize with minimal required parameters
                super().__init__(
                    logger=logger,
                    min_area_ratio=0.0,
                    use_adaptive=False,
                    use_fallback=False,
                )

            def detect(self, image):
                return None, "mock"

        self.orientation_corrector = OrientationCorrector(
            logger=self.logger,
            ensure_doctr=ensure_doctr,
            detector=MockDetector(self.logger),
            angle_threshold=2.0,
            expand_canvas=True,
            preserve_origin_shape=False,
        )

    def execute_stage(
        self, stage_name: str, image: np.ndarray, config: dict[str, Any], context: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Execute a single pipeline stage.

        Args:
            stage_name: Name of the stage to execute
            image: Input image for the stage
            config: Configuration dictionary
            context: Optional context from previous stages

        Returns:
            Tuple of (processed_image, stage_results)
        """
        self.performance_monitor.start_stage(stage_name)

        try:
            method_name = f"_execute_{stage_name}"
            if hasattr(self, method_name):
                result_image, stage_results = getattr(self, method_name)(image, config, context or {})
            else:
                # Fallback for unknown stages
                result_image, stage_results = image, {}

            duration = self.performance_monitor.end_stage(stage_name)
            stage_results["_execution_time"] = duration

            return result_image, stage_results

        except Exception as e:
            self.logger.error(f"Error executing stage {stage_name}: {e}")
            duration = self.performance_monitor.end_stage(stage_name)
            return image, {"error": str(e), "_execution_time": duration}

    def execute_stages_selective(self, image: np.ndarray, config: dict[str, Any], stages: list[str]) -> dict[str, Any]:
        """
        Execute selected pipeline stages and capture intermediate results.

        Args:
            image: Input image
            config: Preprocessing configuration
            stages: List of stage names to execute

        Returns:
            Dictionary mapping stage names to results
        """
        results: dict[str, Any] = {"original": image.copy()}
        current_image = image.copy()
        context: dict[str, Any] = {}

        # Reset performance monitor
        self.performance_monitor = PipelinePerformanceMonitor()

        for stage_name in stages:
            if stage_name == "original":
                continue

            current_image, stage_results = self.execute_stage(stage_name, current_image, config, context)

            # Update results and context
            results.update(stage_results)
            results[stage_name] = current_image.copy()
            context.update(stage_results)

        # Add performance metrics
        results["_performance_metrics"] = self.performance_monitor.get_stage_metrics()

        return results

    def get_available_stages(self) -> list[str]:
        """Get list of all available pipeline stages."""
        return [
            "original",
            "roi_selected",
            "roi_cropped",
            "grayscale",
            "color_inverted",
            "corners_detected",
            "flattened",
            "perspective_corrected",
            "orientation_corrected",
            "noise_eliminated",
            "brightness_adjusted",
            "enhanced",
            "final",
        ]

    def get_stage_dependencies(self, stage_name: str) -> list[str]:
        """Get list of stages that must be executed before the given stage."""
        dependencies = {
            "roi_cropped": ["roi_selected"],
            "corners_detected": ["roi_cropped"],
            "flattened": ["corners_detected"],
            "perspective_corrected": ["corners_detected"],
            "orientation_corrected": ["corners_detected"],
            "noise_eliminated": ["perspective_corrected", "orientation_corrected"],
            "brightness_adjusted": ["noise_eliminated"],
            "enhanced": ["brightness_adjusted"],
            "final": ["enhanced"],
        }
        return dependencies.get(stage_name, [])

    # Individual stage execution methods

    def _execute_roi_selected(
        self, image: np.ndarray, config: dict[str, Any], context: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Execute ROI selection stage."""
        roi = context.get("roi")
        if roi is None:
            return image, {}

        x, y, w, h = roi
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[y : y + h, x : x + w] = 255
        highlighted = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
        return highlighted, {"roi": roi}

    def _execute_roi_cropped(self, image: np.ndarray, config: dict[str, Any], context: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Execute ROI cropping stage."""
        roi = context.get("roi")
        if roi is None:
            return image, {}

        x, y, w, h = roi
        cropped = image[y : y + h, x : x + w].copy()
        return cropped, {"roi_cropped": True}

    def _execute_grayscale(self, image: np.ndarray, config: dict[str, Any], context: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Execute grayscale conversion stage."""
        if not config.get("convert_to_grayscale", False):
            return image, {}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bgr_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return bgr_gray, {"grayscale_applied": True}

    def _execute_color_inverted(
        self, image: np.ndarray, config: dict[str, Any], context: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Execute color inversion stage."""
        if not config.get("color_inversion", False):
            return image, {}

        inverted = cv2.bitwise_not(image)
        return inverted, {"color_inverted": True}

    def _execute_corners_detected(
        self, image: np.ndarray, config: dict[str, Any], context: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Execute corner detection stage."""
        if not config.get("enable_document_detection", True):
            return image, {}

        detected = self.corner_detector.detect_corners(image)
        results = {}

        if detected.corners is not None:
            # Select ordered quadrilateral from detected corners
            image_height, image_width = image.shape[:2]
            selected_quad = self.corner_selector.select_quadrilateral(detected, (image_height, image_width))

            if selected_quad is not None:
                # Draw detected corners and quadrilateral
                vis_image = image.copy()
                for corner in detected.corners:
                    cv2.circle(vis_image, tuple(map(int, corner)), 3, (0, 255, 0), -1)
                quad_corners = selected_quad.corners.astype(int)
                for i in range(4):
                    cv2.line(vis_image, tuple(quad_corners[i]), tuple(quad_corners[(i + 1) % 4]), (255, 0, 0), 2)

                results.update(
                    {
                        "detected_corners": detected.corners,
                        "selected_quadrilateral": selected_quad.corners,
                        "quadrilateral_confidence": f"{selected_quad.confidence:.3f}",
                    }
                )
                return vis_image, results
            else:
                # Just show detected corners
                vis_image = image.copy()
                for corner in detected.corners:
                    cv2.circle(vis_image, tuple(map(int, corner)), 5, (0, 255, 0), -1)
                results["detected_corners"] = detected.corners
                return vis_image, results

        return image, results

    def _execute_flattened(self, image: np.ndarray, config: dict[str, Any], context: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Execute document flattening stage."""
        if not config.get("enable_document_flattening", True):
            return image, {}

        corners = context.get("selected_quadrilateral") or context.get("detected_corners")
        if not isinstance(corners, np.ndarray):
            return image, {}

        try:
            flattened_result = self.document_flattener.flatten_document(image, corners)
            if (
                flattened_result is not None
                and hasattr(flattened_result, "flattened_image")
                and flattened_result.flattened_image is not None
            ):
                return flattened_result.flattened_image, {"flattened": True}
        except Exception as e:
            self.logger.warning(f"Document flattening failed: {e}")

        return image, {"flattened": False}

    def _execute_perspective_corrected(
        self, image: np.ndarray, config: dict[str, Any], context: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Execute perspective correction stage."""
        if not config.get("enable_perspective_correction", True):
            return image, {}

        corners = context.get("selected_quadrilateral") or context.get("detected_corners")
        if not isinstance(corners, np.ndarray):
            return image, {}

        # Validate geometry
        image_height, image_width = image.shape[:2]
        validation_result = self.geometry_validator.validate_quadrilateral(corners, (image_height, image_width))

        if not (
            validation_result.is_valid or validation_result.fallback_recommendation in ["proceed_with_caution", "use_alternative_method"]
        ):
            return image, {
                "geometry_validation_valid": "false",
                "geometry_validation_confidence": f"{validation_result.confidence:.3f}",
                "geometry_validation_issues": "; ".join(validation_result.issues) if validation_result.issues else "none",
            }

        try:
            corrected, _, _ = self.perspective_corrector.correct(image, corners)
            return corrected, {
                "geometry_validation_valid": "true" if validation_result.is_valid else "false",
                "geometry_validation_confidence": f"{validation_result.confidence:.3f}",
                "geometry_validation_issues": "; ".join(validation_result.issues) if validation_result.issues else "none",
            }
        except Exception as e:
            self.logger.warning(f"Perspective correction failed: {e}")
            return image, {
                "geometry_validation_valid": "false",
                "geometry_validation_confidence": "0.000",
                "geometry_validation_issues": f"Correction failed: {str(e)}",
            }

    def _execute_orientation_corrected(
        self, image: np.ndarray, config: dict[str, Any], context: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Execute orientation correction stage."""
        if not config.get("enable_orientation_correction", False):
            return image, {}

        corners = context.get("selected_quadrilateral") or context.get("detected_corners")
        if not isinstance(corners, np.ndarray):
            return image, {}

        try:
            corrected, _, _ = self.orientation_corrector.correct(image, corners)
            return corrected, {"orientation_corrected": True}
        except Exception as e:
            self.logger.warning(f"Orientation correction failed: {e}")
            return image, {"orientation_corrected": False, "orientation_error": str(e)}

    def _execute_noise_eliminated(
        self, image: np.ndarray, config: dict[str, Any], context: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Execute noise elimination stage."""
        if not config.get("enable_noise_elimination", True):
            return image, {}

        try:
            noise_result = self.noise_eliminator.eliminate_noise(image)
            if hasattr(noise_result, "cleaned_image") and noise_result.cleaned_image is not None:
                return noise_result.cleaned_image, {"noise_eliminated": True}
        except Exception as e:
            self.logger.warning(f"Noise elimination failed: {e}")

        return image, {"noise_eliminated": False}

    def _execute_brightness_adjusted(
        self, image: np.ndarray, config: dict[str, Any], context: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Execute brightness adjustment stage."""
        if not config.get("enable_brightness_adjustment", True):
            return image, {}

        try:
            brightness_result = self.brightness_adjuster.adjust_brightness(image)
            if hasattr(brightness_result, "adjusted_image") and brightness_result.adjusted_image is not None:
                return brightness_result.adjusted_image, {"brightness_adjusted": True}
        except Exception as e:
            self.logger.warning(f"Brightness adjustment failed: {e}")

        return image, {"brightness_adjusted": False}

    def _execute_enhanced(self, image: np.ndarray, config: dict[str, Any], context: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Execute enhancement stage."""
        if not config.get("enable_enhancement", True):
            return image, {}

        method = config.get("enhancement_method", "conservative")
        try:
            enhanced, applied_enhancements = self.image_enhancer.enhance(image, method)
            return enhanced, {"enhanced": True, "enhancement_method": method, "applied_enhancements": applied_enhancements}
        except Exception as e:
            self.logger.warning(f"Enhancement failed: {e}")
            return image, {"enhanced": False, "enhancement_error": str(e)}

    def _execute_final(self, image: np.ndarray, config: dict[str, Any], context: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Execute final stage (no-op, just marks completion)."""
        return image, {"pipeline_completed": True}

    # Backward compatibility methods

    @validate_viewer_config
    @validate_roi_request
    def process_with_intermediates(
        self, image: np.ndarray, config: dict[str, Any], roi: tuple[int, int, int, int] | None = None
    ) -> dict[str, np.ndarray | str]:
        """
        Backward-compatible method that processes all stages.

        This maintains compatibility with existing code while using the new selective execution engine.
        """
        # Set up ROI context if provided
        context = {}
        if roi is not None:
            context["roi"] = roi

        # Execute all available stages
        all_stages = self.get_available_stages()
        results = self.execute_stages_selective(image, config, all_stages)

        # Convert results to expected format
        formatted_results = {}
        for key, value in results.items():
            if key != "_performance_metrics":
                formatted_results[key] = value

        return formatted_results

    def get_telemetry_summary(self) -> dict[str, Any]:
        """Get summary of telemetry events from the last processing run."""
        return self.telemetry.get_events_summary()

    def get_telemetry_events(self) -> list[Any]:
        """Get all telemetry events from the last processing run."""
        return self.telemetry.events

    def get_stage_description(self, stage: str) -> str:
        """Get human-readable description for a pipeline stage."""
        descriptions = {
            "original": "Original input image",
            "roi_selected": "Region of interest highlighted",
            "roi_cropped": "Cropped to selected region",
            "grayscale": "Converted to grayscale",
            "color_inverted": "Colors inverted",
            "corners_detected": "Document corners detected",
            "flattened": "Document geometrically flattened",
            "perspective_corrected": "Perspective distortion corrected",
            "orientation_corrected": "Orientation corrected",
            "noise_eliminated": "Noise eliminated",
            "brightness_adjusted": "Brightness intelligently adjusted",
            "enhanced": "Image enhanced",
            "final": "Final processed image",
        }
        return descriptions.get(stage, stage)
