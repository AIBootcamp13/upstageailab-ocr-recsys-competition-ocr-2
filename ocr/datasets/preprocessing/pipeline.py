"""High-level document preprocessing pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import DocumentPreprocessorConfig
from .contracts import validate_image_input_with_fallback, validate_preprocessing_result_with_fallback
from .detector import DocumentDetector
from .enhancement import ImageEnhancer
from .external import ALBUMENTATIONS_AVAILABLE, DOCTR_AVAILABLE, A, ImageOnlyTransform
from .metadata import DocumentMetadata, PreprocessingState
from .orientation import OrientationCorrector
from .padding import PaddingCleanup
from .perspective import PerspectiveCorrector
from .resize import FinalResizer


class DocumentPreprocessor:
    """Microsoft Lens-style document preprocessing pipeline composed of modular steps."""

    def __init__(
        self,
        enable_document_detection: bool = True,
        enable_perspective_correction: bool = True,
        enable_enhancement: bool = True,
        enhancement_method: str = "conservative",
        target_size: tuple[int, int] | None = (640, 640),
        enable_final_resize: bool = True,
        enable_orientation_correction: bool = False,
        orientation_angle_threshold: float = 2.0,
        orientation_expand_canvas: bool = True,
        orientation_preserve_original_shape: bool = False,
        use_doctr_geometry: bool = False,
        doctr_assume_horizontal: bool = False,
        enable_padding_cleanup: bool = False,
        document_detection_min_area_ratio: float = 0.18,
        document_detection_use_adaptive: bool = True,
        document_detection_use_fallback_box: bool = True,
        document_detection_use_camscanner: bool = False,
        document_detection_use_doctr_text: bool = False,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        config = DocumentPreprocessorConfig(
            enable_document_detection=enable_document_detection,
            enable_perspective_correction=enable_perspective_correction,
            enable_enhancement=enable_enhancement,
            enhancement_method=enhancement_method,  # type: ignore
            target_size=target_size,
            enable_final_resize=enable_final_resize,
            enable_orientation_correction=enable_orientation_correction,
            orientation_angle_threshold=orientation_angle_threshold,
            orientation_expand_canvas=orientation_expand_canvas,
            orientation_preserve_original_shape=orientation_preserve_original_shape,
            use_doctr_geometry=use_doctr_geometry,
            doctr_assume_horizontal=doctr_assume_horizontal,
            enable_padding_cleanup=enable_padding_cleanup,
            document_detection_min_area_ratio=document_detection_min_area_ratio,
            document_detection_use_adaptive=document_detection_use_adaptive,
            document_detection_use_fallback_box=document_detection_use_fallback_box,
            document_detection_use_camscanner=document_detection_use_camscanner,
            document_detection_use_doctr_text=document_detection_use_doctr_text,
        )

        if config.enhancement_method not in {"conservative", "office_lens"}:
            raise ValueError(f"enhancement_method must be 'conservative' or 'office_lens', got '{config.enhancement_method}'")

        self.config = config
        self.doctr_available = DOCTR_AVAILABLE
        self._warned_features: set[str] = set()

        # Legacy attribute surface for backwards compatibility with configs/tests
        self.enable_document_detection = self.config.enable_document_detection
        self.enable_perspective_correction = self.config.enable_perspective_correction
        self.enable_enhancement = self.config.enable_enhancement
        self.enhancement_method = self.config.enhancement_method
        self.target_size = self.config.target_size
        self.enable_final_resize = self.config.enable_final_resize
        self.enable_orientation_correction = self.config.enable_orientation_correction
        self.orientation_angle_threshold = self.config.orientation_angle_threshold
        self.orientation_expand_canvas = self.config.orientation_expand_canvas
        self.orientation_preserve_original_shape = self.config.orientation_preserve_original_shape
        self.use_doctr_geometry = self.config.use_doctr_geometry
        self.doctr_assume_horizontal = self.config.doctr_assume_horizontal
        self.enable_padding_cleanup = self.config.enable_padding_cleanup
        self.document_detection_min_area_ratio = self.config.document_detection_min_area_ratio
        self.document_detection_use_adaptive = self.config.document_detection_use_adaptive
        self.document_detection_use_fallback_box = self.config.document_detection_use_fallback_box
        self.document_detection_use_camscanner = self.config.document_detection_use_camscanner
        self.document_detection_use_doctr_text = self.config.document_detection_use_doctr_text

        self.detector = DocumentDetector(
            logger=self.logger,
            min_area_ratio=self.config.document_detection_min_area_ratio,
            use_adaptive=self.config.document_detection_use_adaptive,
            use_fallback=self.config.document_detection_use_fallback_box,
            use_camscanner=self.config.document_detection_use_camscanner,
            use_doctr_text=self.config.document_detection_use_doctr_text,
        )
        self.orientation_corrector = OrientationCorrector(
            logger=self.logger,
            ensure_doctr=self._ensure_doctr,
            detector=self.detector,
            angle_threshold=self.config.orientation_angle_threshold,
            expand_canvas=self.config.orientation_expand_canvas,
            preserve_origin_shape=self.config.orientation_preserve_original_shape,
        )
        self.perspective_corrector = PerspectiveCorrector(
            logger=self.logger,
            ensure_doctr=self._ensure_doctr,
            use_doctr_geometry=self.config.use_doctr_geometry,
            doctr_assume_horizontal=self.config.doctr_assume_horizontal,
        )
        self.padding_cleanup = PaddingCleanup(self._ensure_doctr)
        self.image_enhancer = ImageEnhancer()
        self.final_resizer = FinalResizer()

    @validate_image_input_with_fallback
    @validate_preprocessing_result_with_fallback
    def __call__(self, image: np.ndarray) -> dict[str, np.ndarray | dict]:
        if not isinstance(image, np.ndarray) or image.size == 0 or len(image.shape) < 2:
            self.logger.warning("Invalid input image, using fallback processing")
            width, height = self.config.target_size if self.config.target_size is not None else (256, 256)
            fallback_image = np.full((height, width, 3), 128, dtype=np.uint8)
            metadata = DocumentMetadata(original_shape=getattr(image, "shape", ()))
            metadata.processing_steps.append("fallback")
            metadata.error = "Invalid input image"
            metadata.final_shape = tuple(int(dim) for dim in fallback_image.shape)
            return {"image": fallback_image, "metadata": metadata.to_dict()}

        state = PreprocessingState(
            image=image.copy(),
            metadata=DocumentMetadata(original_shape=image.shape),
        )

        try:
            if self.config.enable_document_detection:
                corners, method = self._detect_document_boundaries_with_method(state.image)
                state.corners = corners
                state.metadata.document_detection_method = method
                if corners is not None:
                    state.metadata.document_corners = corners
                    state.metadata.processing_steps.append("document_detection")
                    # Store intermediate image for debugging
                    state.metadata.image_after_document_detection = state.image.copy()
                else:
                    self.logger.warning("Document boundaries not detected; geometric corrections skipped")
            else:
                state.metadata.document_detection_method = "disabled"

            if (
                self.config.enable_orientation_correction
                and state.corners is not None
                and not self.config.document_detection_use_camscanner
            ):
                corrected_image, corrected_corners, orientation_meta = self.orientation_corrector.correct(
                    state.image,
                    state.corners,
                )
                state.image = corrected_image
                if corrected_corners is not None:
                    state.corners = corrected_corners
                    state.metadata.document_corners = corrected_corners
                    if orientation_meta and orientation_meta.get("redetection_method"):
                        state.metadata.document_detection_method = orientation_meta["redetection_method"]
                if orientation_meta is not None:
                    state.metadata.orientation = orientation_meta
                    state.metadata.processing_steps.append("orientation_correction")
                    # Store intermediate image for debugging
                    state.metadata.image_after_orientation_correction = state.image.copy()

            if self.config.enable_perspective_correction and state.corners is not None:
                corrected, matrix, method = self.perspective_corrector.correct(state.image, state.corners)
                state.image = corrected
                state.metadata.perspective_matrix = matrix
                state.metadata.perspective_method = method
                state.metadata.processing_steps.append("perspective_correction")
                # Store intermediate image for debugging
                state.metadata.image_after_perspective_correction = state.image.copy()

            if self.config.enable_padding_cleanup:
                cleaned = self.padding_cleanup.cleanup(state.image)
                if cleaned is not None:
                    state.image = cleaned
                    state.metadata.processing_steps.append("padding_cleanup")

            if self.config.enable_enhancement:
                enhanced, applied = self.image_enhancer.enhance(state.image, self.config.enhancement_method)
                state.image = enhanced
                state.metadata.enhancement_applied.extend(applied)
                state.metadata.processing_steps.append("image_enhancement")
                # Store intermediate image for debugging
                state.metadata.image_after_enhancement = state.image.copy()

            if self.config.enable_final_resize and self.config.target_size is not None:
                state.image = self.final_resizer.resize(state.image, self.config.target_size)
                state.metadata.processing_steps.append("resize_to_target")

            state.update_final_shape()
            return {"image": state.image, "metadata": state.metadata.to_dict()}

        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.error("Preprocessing failed: %s", exc, exc_info=True)
            fallback_image = image.copy()
            if self.config.enable_final_resize and self.config.target_size is not None:
                fallback_image = self.final_resizer.resize(fallback_image, self.config.target_size)
            state.metadata.error = str(exc)
            state.metadata.processing_steps = ["fallback_resize"]
            state.metadata.final_shape = tuple(int(dim) for dim in fallback_image.shape)
            return {"image": fallback_image, "metadata": state.metadata.to_dict()}

    def _ensure_doctr(self, feature: str) -> bool:
        if self.doctr_available:
            return True
        if feature not in self._warned_features:
            self.logger.warning(
                "python-doctr is required for %s but is not installed; skipping this step.",
                feature,
            )
            self._warned_features.add(feature)
        return False

    # ------------------------------------------------------------------
    # Backwards compatibility helpers (legacy private API)
    # ------------------------------------------------------------------

    def _detect_document_boundaries_with_method(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray | None, str | None]:
        """Detect document boundaries and return both corners and method."""

        return self.detector.detect(image)

    def _detect_document_boundaries(self, image: np.ndarray) -> np.ndarray | None:
        """Compatibility wrapper returning only corners."""

        corners, _ = self._detect_document_boundaries_with_method(image)
        return corners

    def _enhance_image(self, image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Compatibility wrapper for conservative enhancement."""

        return self.image_enhancer._enhance_image(image)

    def _enhance_image_office_lens(self, image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Compatibility wrapper for Office Lens-style enhancement."""

        return self.image_enhancer._enhance_image_office_lens(image)

    def _resize_to_target(self, image: np.ndarray) -> np.ndarray:
        """Compatibility wrapper for resizing helper (uses configured target size)."""

        if self.config.target_size is None:
            return image
        return self.final_resizer.resize(image, self.config.target_size)

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Compatibility wrapper for corner ordering utility."""

        return self.detector._order_corners(corners)


if TYPE_CHECKING or (ALBUMENTATIONS_AVAILABLE and A is not None and ImageOnlyTransform is not None):
    assert A is not None  # For mypy
    assert ImageOnlyTransform is not None  # For mypy

    class LensStylePreprocessorAlbumentations(A.DualTransform):  # type: ignore[misc,name-defined]
        """Albumentations-compatible wrapper for the document preprocessor.

        BUG FIX (BUG-2025-003): Properly inherits from A.DualTransform to handle both
        images and keypoints. Previous implementation only transformed images, causing
        coordinate space mismatches with polygons.

        This transform applies geometric preprocessing (document detection, perspective
        correction) to both images and keypoints simultaneously to maintain coordinate
        space consistency.
        """

        def __init__(self, preprocessor: DocumentPreprocessor, always_apply: bool = False, p: float = 1.0):
            super().__init__(always_apply=always_apply, p=p)
            self.preprocessor = preprocessor
            # Store preprocessing result for keypoint transformation
            self._last_preprocessing_result: dict[str, Any] | None = None

        def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:  # type: ignore[override]
            """Apply document preprocessing to the image.

            Args:
                img: Input image as numpy array
                **params: Additional parameters from Albumentations

            Returns:
                Processed image as numpy array
            """
            result = self.preprocessor(img)
            # Store the full result for keypoint transformation
            self._last_preprocessing_result = result
            # Return just the processed image; Albumentations handles dict wrapping
            processed_image = result["image"]
            assert isinstance(processed_image, np.ndarray), "Preprocessor must return numpy array"
            return processed_image

        def apply_to_keypoints(self, keypoints: list[tuple[float, ...]], **params: Any) -> list[tuple[float, ...]]:  # type: ignore[override]
            """Apply geometric transformations to keypoints using preprocessing matrices.

            Args:
                keypoints: List of keypoints as (x, y, ...) tuples
                **params: Additional parameters from Albumentations

            Returns:
                Transformed keypoints as list of tuples
            """
            # Use the stored preprocessing result
            result = self._last_preprocessing_result
            if result is None:
                # No preprocessing result available, return keypoints unchanged
                return keypoints

            # Check if geometric transformations were applied
            metadata = result.get("metadata", {})
            processing_steps = metadata.get("processing_steps", [])
            has_geometric_transform = any(step in processing_steps for step in ["document_detection", "perspective_correction"])

            if not has_geometric_transform:
                # No geometric transformations applied, return keypoints unchanged
                return keypoints

            # Get the perspective matrix
            perspective_matrix = metadata.get("perspective_matrix")
            if perspective_matrix is None or not isinstance(perspective_matrix, list | np.ndarray):
                # No transformation matrix available, return keypoints unchanged
                return keypoints

            perspective_matrix = np.array(perspective_matrix)
            if perspective_matrix.shape != (3, 3):
                # Invalid matrix shape, return keypoints unchanged
                return keypoints

            # Transform keypoints using the perspective matrix
            transformed_keypoints = []
            for kp in keypoints:
                x, y = kp[0], kp[1]

                # Convert to homogeneous coordinates
                homogeneous_point = np.array([x, y, 1.0])

                # Apply transformation
                transformed_point = perspective_matrix @ homogeneous_point

                # Convert back to cartesian coordinates
                transformed_x = transformed_point[0] / transformed_point[2]
                transformed_y = transformed_point[1] / transformed_point[2]

                # Keep additional keypoint parameters (angle, scale, etc.)
                transformed_keypoints.append((transformed_x, transformed_y) + tuple(kp[2:]))

            return transformed_keypoints

        def get_transform_init_args_names(self) -> tuple[str, ...]:
            return ("preprocessor",)

else:
    # Fallback when Albumentations is not available
    class LensStylePreprocessorAlbumentations:  # type: ignore[no-redef]
        """Fallback wrapper when Albumentations is not available."""

        def __init__(self, preprocessor: DocumentPreprocessor):
            self.preprocessor = preprocessor

        def __call__(self, image: np.ndarray, **kwargs: Any) -> dict[str, Any]:
            """Process image and return result dict."""
            return self.preprocessor(image)

        def get_transform_init_args_names(self) -> tuple[()]:
            return ()


__all__ = ["DocumentPreprocessor", "LensStylePreprocessorAlbumentations"]
