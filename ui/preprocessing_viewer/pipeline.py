"""
Viewer-specific preprocessing pipeline orchestrator for Streamlit Preprocessing Viewer.

This module provides a standalone pipeline that imports and calls individual preprocessing
functions sequentially, capturing intermediate results for step-by-step visualization.
"""

import logging
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
from ocr.datasets.preprocessing.geometry_validation import GeometryValidationResult, GeometryValidationUtility
from ocr.datasets.preprocessing.intelligent_brightness import IntelligentBrightnessAdjuster
from ocr.datasets.preprocessing.orientation import OrientationCorrector
from ocr.datasets.preprocessing.perspective import PerspectiveCorrector
from ocr.datasets.preprocessing.telemetry import PreprocessingTelemetry

# Import viewer contracts for validation
from .viewer_contracts import validate_roi_request, validate_viewer_config


class PreprocessingViewerPipeline:
    """
    Standalone pipeline orchestrator for the preprocessing viewer.

    This class imports and calls individual preprocessing functions sequentially,
    capturing intermediate results for visualization and comparison.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

        # Initialize preprocessing components
        self.corner_detector = AdvancedCornerDetector()
        self.corner_selector = CornerSelectionUtility()
        self.geometry_validator = GeometryValidationUtility()
        self.telemetry = PreprocessingTelemetry(logger)
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

    @validate_viewer_config
    @validate_roi_request
    def process_with_intermediates(
        self, image: np.ndarray, config: dict[str, Any], roi: tuple[int, int, int, int] | None = None
    ) -> dict[str, np.ndarray | str]:
        """
        Process image through pipeline and capture all intermediate results.

        Args:
            image: Input image as numpy array
            config: Preprocessing configuration dictionary
            roi: Optional ROI as (x, y, w, h) tuple

        Returns:
            Dictionary mapping stage names to processed images
        """
        self.logger.info(f"Starting preprocessing pipeline - image shape: {image.shape}, config keys: {list(config.keys())}")
        results: dict[str, np.ndarray | str] = {"original": image.copy()}
        current_image = image.copy()
        validation_result: GeometryValidationResult | None = None

        # Apply ROI if specified
        if roi is not None:
            x, y, w, h = roi
            roi_image = current_image[y : y + h, x : x + w].copy()
            # Create mask for full image
            mask = np.zeros_like(current_image, dtype=np.uint8)
            mask[y : y + h, x : x + w] = 255
            results["roi_selected"] = cv2.addWeighted(current_image, 0.7, mask, 0.3, 0)
            current_image = roi_image
            results["roi_cropped"] = current_image.copy()

        try:
            # Stage 1: Color preprocessing
            if config.get("enable_color_preprocessing", True):
                if config.get("convert_to_grayscale", False):
                    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                    current_image = cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR)
                    results["grayscale"] = current_image.copy()

                if config.get("color_inversion", False):
                    current_image = cv2.bitwise_not(current_image)
                    results["color_inverted"] = current_image.copy()

            # Stage 2: Document detection
            if config.get("enable_document_detection", True):
                detected = self.corner_detector.detect_corners(current_image)
                if detected.corners is not None:
                    # Select ordered quadrilateral from detected corners
                    image_height, image_width = current_image.shape[:2]
                    selected_quad = self.corner_selector.select_quadrilateral(detected, (image_height, image_width))
                    if selected_quad is not None:
                        # Log successful corner selection
                        self.telemetry.log_processing_decision(
                            stage="corner_selection",
                            decision="quadrilateral_selected",
                            success=True,
                            confidence=selected_quad.confidence,
                            issues=[],
                            corner_count=len(detected.corners),
                            selected_corners=selected_quad.corners.tolist(),
                        )

                        # Draw detected corners
                        vis_image = current_image.copy()
                        for corner in detected.corners:
                            cv2.circle(vis_image, tuple(map(int, corner)), 3, (0, 255, 0), -1)
                        # Draw selected quadrilateral
                        quad_corners = selected_quad.corners.astype(int)
                        for i in range(4):
                            cv2.line(vis_image, tuple(quad_corners[i]), tuple(quad_corners[(i + 1) % 4]), (255, 0, 0), 2)
                        results["corners_detected"] = vis_image
                        results["detected_corners"] = detected.corners
                        results["selected_quadrilateral"] = selected_quad.corners
                        results["quadrilateral_confidence"] = f"{selected_quad.confidence:.3f}"
                    else:
                        # Log failed corner selection
                        self.telemetry.log_processing_decision(
                            stage="corner_selection",
                            decision="quadrilateral_selection_failed",
                            success=False,
                            issues=["Could not select valid quadrilateral from detected corners"],
                            corner_count=len(detected.corners),
                        )

                        # Fallback: just show detected corners
                        vis_image = current_image.copy()
                        for corner in detected.corners:
                            cv2.circle(vis_image, tuple(map(int, corner)), 5, (0, 255, 0), -1)
                        results["corners_detected"] = vis_image
                        results["detected_corners"] = detected.corners
                else:
                    # Log corner detection failure
                    self.telemetry.log_processing_decision(
                        stage="corner_detection",
                        decision="no_corners_detected",
                        success=False,
                        issues=["Corner detection returned no corners"],
                    )
                    results["corners_detected"] = current_image.copy()

            # Stage 3: Document flattening (WARNING: CPU-intensive, can take 3-15s)
            corners_for_processing = results.get("selected_quadrilateral") or results.get("detected_corners")
            if config.get("enable_document_flattening", False) and isinstance(corners_for_processing, np.ndarray):
                self.logger.info("Starting document flattening (may take 3-15 seconds)...")
                try:
                    flattened_result = self.document_flattener.flatten_document(current_image, corners_for_processing)
                    if (
                        flattened_result is not None
                        and hasattr(flattened_result, "flattened_image")
                        and flattened_result.flattened_image is not None
                    ):
                        current_image = flattened_result.flattened_image
                        results["flattened"] = current_image.copy()
                        self.logger.info("Document flattening completed successfully")
                except Exception as e:
                    self.logger.warning(f"Document flattening failed: {e}")
                    # Skip flattening if it fails

            # Stage 4: Perspective correction
            if config.get("enable_perspective_correction", True) and isinstance(corners_for_processing, np.ndarray):
                # Validate geometry before perspective correction
                image_height, image_width = current_image.shape[:2]
                validation_result = self.geometry_validator.validate_quadrilateral(corners_for_processing, (image_height, image_width))

                # Log geometry validation
                self.telemetry.log_geometry_validation(
                    stage="perspective_correction",
                    validation_result=validation_result,
                    corners=corners_for_processing,
                    image_shape=(image_height, image_width),
                )

                if validation_result.is_valid or validation_result.fallback_recommendation in [
                    "proceed_with_caution",
                    "use_alternative_method",
                ]:
                    try:
                        corrected, _, _ = self.perspective_corrector.correct(current_image, corners_for_processing)
                        current_image = corrected
                        results["perspective_corrected"] = current_image.copy()
                        results["geometry_validation_valid"] = "true" if validation_result.is_valid else "false"
                        results["geometry_validation_confidence"] = f"{validation_result.confidence:.3f}"
                        results["geometry_validation_issues"] = "; ".join(validation_result.issues) if validation_result.issues else "none"
                        results["geometry_validation_fallback"] = validation_result.fallback_recommendation or "none"

                        # Log successful correction
                        self.telemetry.log_correction_attempt(stage="perspective_correction", correction_type="perspective", success=True)
                    except Exception as e:
                        self.logger.warning(f"Perspective correction failed: {e}")
                        results["geometry_validation_valid"] = "false"
                        results["geometry_validation_confidence"] = "0.000"
                        results["geometry_validation_issues"] = f"Correction failed: {str(e)}"
                        results["geometry_validation_fallback"] = "skip_processing"

                        # Log failed correction
                        self.telemetry.log_correction_attempt(
                            stage="perspective_correction", correction_type="perspective", success=False, error_message=str(e)
                        )
                else:
                    self.logger.info(f"Skipping perspective correction due to geometry validation: {validation_result.issues}")
                    results["geometry_validation_valid"] = "false"
                    results["geometry_validation_confidence"] = f"{validation_result.confidence:.3f}"
                    results["geometry_validation_issues"] = "; ".join(validation_result.issues) if validation_result.issues else "none"
                    results["geometry_validation_fallback"] = validation_result.fallback_recommendation or "none"

                    # Log fallback action
                    self.telemetry.log_fallback_action(
                        stage="perspective_correction",
                        fallback_type=validation_result.fallback_recommendation or "skip_processing",
                        reason="Geometry validation failed",
                    )

            # Stage 5: Orientation correction
            if config.get("enable_orientation_correction", False) and isinstance(corners_for_processing, np.ndarray):
                # Re-validate geometry if not already validated in perspective stage
                if "geometry_validation_valid" not in results:
                    image_height, image_width = current_image.shape[:2]
                    validation_result = self.geometry_validator.validate_quadrilateral(corners_for_processing, (image_height, image_width))

                    # Log geometry validation for orientation
                    self.telemetry.log_geometry_validation(
                        stage="orientation_correction",
                        validation_result=validation_result,
                        corners=corners_for_processing,
                        image_shape=(image_height, image_width),
                    )
                else:
                    # Use existing validation result
                    validation_result = None  # We'll check the stored result

                # Check if we should proceed with orientation correction
                should_proceed = False
                if "geometry_validation_valid" in results:
                    is_valid = results.get("geometry_validation_valid") == "true"
                    fallback = results.get("geometry_validation_fallback", "none")
                    should_proceed = is_valid or fallback in ["proceed_with_caution", "use_alternative_method"]
                elif validation_result:
                    should_proceed = validation_result.is_valid or validation_result.fallback_recommendation in [
                        "proceed_with_caution",
                        "use_alternative_method",
                    ]

                if should_proceed:
                    try:
                        corrected, _, _ = self.orientation_corrector.correct(current_image, corners_for_processing)
                        current_image = corrected
                        results["orientation_corrected"] = current_image.copy()

                        # Log successful correction
                        self.telemetry.log_correction_attempt(stage="orientation_correction", correction_type="orientation", success=True)
                    except Exception as e:
                        self.logger.warning(f"Orientation correction failed: {e}")
                        # Don't overwrite existing geometry validation
                        if "geometry_validation_valid" not in results:
                            results["geometry_validation_valid"] = "false"
                            results["geometry_validation_confidence"] = "0.000"
                            results["geometry_validation_issues"] = f"Orientation correction failed: {str(e)}"
                            results["geometry_validation_fallback"] = "skip_processing"

                        # Log failed correction
                        self.telemetry.log_correction_attempt(
                            stage="orientation_correction", correction_type="orientation", success=False, error_message=str(e)
                        )
                else:
                    self.logger.info("Skipping orientation correction due to geometry validation")

                    # Log fallback action
                    fallback_reason = "Geometry validation failed"
                    if "geometry_validation_issues" in results:
                        fallback_reason = f"Previous validation: {results['geometry_validation_issues']}"

                    self.telemetry.log_fallback_action(
                        stage="orientation_correction", fallback_type="skip_processing", reason=fallback_reason
                    )

            # Stage 6: Noise elimination
            if config.get("enable_noise_elimination", True):
                self.logger.info("Starting noise elimination...")
                try:
                    noise_result = self.noise_eliminator.eliminate_noise(current_image)
                    if hasattr(noise_result, "cleaned_image") and noise_result.cleaned_image is not None:
                        current_image = noise_result.cleaned_image
                        results["noise_eliminated"] = current_image.copy()
                        self.logger.info("Noise elimination completed successfully")
                except Exception as e:
                    self.logger.warning(f"Noise elimination failed: {e}")
                    # Skip noise elimination if it fails

            # Stage 7: Brightness adjustment
            if config.get("enable_brightness_adjustment", True):
                self.logger.info("Starting brightness adjustment...")
                try:
                    brightness_result = self.brightness_adjuster.adjust_brightness(current_image)
                    if hasattr(brightness_result, "adjusted_image") and brightness_result.adjusted_image is not None:
                        current_image = brightness_result.adjusted_image
                        results["brightness_adjusted"] = current_image.copy()
                        self.logger.info("Brightness adjustment completed successfully")
                except Exception as e:
                    self.logger.warning(f"Brightness adjustment failed: {e}")
                    # Skip brightness adjustment if it fails

            # Stage 8: Enhancement
            if config.get("enable_enhancement", True):
                method = config.get("enhancement_method", "conservative")
                try:
                    enhanced, _ = self.image_enhancer.enhance(current_image, method)
                    current_image = enhanced
                    results["enhanced"] = current_image.copy()
                except Exception:
                    # Skip enhancement if it fails
                    pass

            # Final result
            results["final"] = current_image.copy()
            self.logger.info(f"Preprocessing pipeline completed successfully - {len(results)} stages executed")

        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {e}", exc_info=True)
            results["error"] = str(e)
            results["final"] = image.copy()  # Fallback to original

        return results

    def get_telemetry_summary(self) -> dict[str, Any]:
        """Get summary of telemetry events from the last processing run."""
        return self.telemetry.get_events_summary()

    def get_telemetry_events(self) -> list[Any]:
        """Get all telemetry events from the last processing run."""
        return self.telemetry.events

    def get_available_stages(self) -> list[str]:
        """Get list of all possible pipeline stages."""
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
