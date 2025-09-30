"""
Microsoft Lens-style Image Preprocessing Module for OCR

This module implements document preprocessing techniques inspired by Microsoft Lens,
focusing on perspective correction and image enhancement for optimal OCR performance.
"""

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    # Type declarations for optional doctr imports
    estimate_page_angle: Callable[[np.ndarray], float] | None
    extract_rcrops: Callable[[np.ndarray, np.ndarray], list[np.ndarray]] | None
    doctr_remove_image_padding: Callable[[np.ndarray], np.ndarray] | None
    doctr_rotate_image: Callable[[np.ndarray, float, bool, bool], np.ndarray] | None

# Optional doctr imports
estimate_page_angle = None
extract_rcrops = None
doctr_remove_image_padding = None
doctr_rotate_image = None

try:
    import albumentations as A

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None

try:
    from doctr.utils.geometry import (
        estimate_page_angle as _estimate_page_angle,
    )
    from doctr.utils.geometry import (
        extract_rcrops as _extract_rcrops,
    )
    from doctr.utils.geometry import (
        remove_image_padding as _doctr_remove_image_padding,
    )
    from doctr.utils.geometry import (
        rotate_image as _doctr_rotate_image,
    )

    estimate_page_angle = _estimate_page_angle
    extract_rcrops = _extract_rcrops
    doctr_remove_image_padding = _doctr_remove_image_padding
    doctr_rotate_image = _doctr_rotate_image

    DOCTR_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency guard
    DOCTR_AVAILABLE = False
    estimate_page_angle = None
    extract_rcrops = None
    doctr_remove_image_padding = None
    doctr_rotate_image = None


class DocumentPreprocessor:
    """
    Microsoft Lens-style document preprocessing pipeline.

    Implements:
    1. Document boundary detection
    2. Perspective correction
    3. Image enhancement (contrast, sharpening, noise reduction)
    4. Text-specific enhancement
    """

    def __init__(
        self,
        enable_document_detection: bool = True,
        enable_perspective_correction: bool = True,
        enable_enhancement: bool = True,
        enable_text_enhancement: bool = False,  # Disabled by default as it can be destructive
        enhancement_method: str = "conservative",  # "conservative" or "office_lens"
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
    ):
        """
        Initialize the document preprocessor.

        Args:
            enable_document_detection: Enable automatic document boundary detection
            enable_perspective_correction: Enable perspective correction
            enable_enhancement: Enable general image enhancement
            enable_text_enhancement: Enable text-specific enhancement
            enhancement_method: Enhancement method to use ("conservative" or "office_lens")
            target_size: Target size for processed images
        """
        self.enable_document_detection = enable_document_detection
        self.enable_perspective_correction = enable_perspective_correction
        self.enable_enhancement = enable_enhancement
        self.enable_text_enhancement = enable_text_enhancement
        self.enhancement_method = enhancement_method
        self.target_size = tuple(target_size) if target_size is not None else None
        self.enable_final_resize = enable_final_resize
        self.enable_orientation_correction = enable_orientation_correction
        self.orientation_angle_threshold = float(orientation_angle_threshold)
        self.orientation_expand_canvas = orientation_expand_canvas
        self.orientation_preserve_original_shape = orientation_preserve_original_shape
        self.use_doctr_geometry = use_doctr_geometry
        self.doctr_assume_horizontal = doctr_assume_horizontal
        self.enable_padding_cleanup = enable_padding_cleanup
        self.doctr_available = DOCTR_AVAILABLE
        self._warned_features: set[str] = set()
        self.document_detection_min_area_ratio = float(max(0.0, min(document_detection_min_area_ratio, 1.0)))
        self.document_detection_use_adaptive = document_detection_use_adaptive
        self.document_detection_use_fallback_box = document_detection_use_fallback_box
        self._last_detection_method: str | None = None

        # Validate enhancement method
        if enhancement_method not in ["conservative", "office_lens"]:
            raise ValueError(f"enhancement_method must be 'conservative' or 'office_lens', got '{enhancement_method}'")

        # Configure logging
        self.logger = logging.getLogger(__name__)

    def __call__(self, image: np.ndarray) -> dict[str, np.ndarray | dict]:
        """
        Process an image through the preprocessing pipeline.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            Dict containing:
            - 'image': Processed image
            - 'metadata': Processing metadata (corners, transforms, etc.)
        """
        # suggestion(bug_risk): The fallback for invalid images always returns a gray image, which may mask upstream issues.
        # Consider raising an exception or returning an explicit error instead to prevent silent failures and make upstream issues more visible.
        # Validate input image
        if not isinstance(image, np.ndarray) or image.size == 0 or len(image.shape) < 2:
            self.logger.warning("Invalid input image, using fallback processing")
            # Create a minimal valid image for fallback
            fallback_width, fallback_height = self.target_size if self.target_size is not None else (256, 256)
            fallback_image = np.full((fallback_height, fallback_width, 3), 128, dtype=np.uint8)
            return {
                "image": fallback_image,
                "metadata": {
                    "original_shape": getattr(image, "shape", "invalid"),
                    "processing_steps": ["fallback"],
                    "error": "Invalid input image",
                },
            }

        original_image = image.copy()
        metadata: dict[str, Any] = {
            "original_shape": image.shape,
            "processing_steps": [],
            "document_corners": None,
            "perspective_matrix": None,
            "enhancement_applied": [],
        }

        try:
            corners: np.ndarray | None = None

            # Step 1: Document Detection
            if self.enable_document_detection:
                corners = self._detect_document_boundaries(image)
                if corners is not None:
                    metadata["document_corners"] = corners
                    metadata["document_detection_method"] = self._last_detection_method
                    metadata["processing_steps"].append("document_detection")
                else:
                    self.logger.warning("Document boundaries not detected; geometric corrections skipped")
                    metadata["document_detection_method"] = self._last_detection_method or "unavailable"
            else:
                metadata["document_corners"] = None

            # Step 2: Optional orientation correction (doctr geometry)
            if self.enable_orientation_correction and metadata["document_corners"] is not None:
                image, corners, orientation_meta = self._apply_orientation_correction(image, metadata["document_corners"])
                if corners is not None:
                    metadata["document_corners"] = corners
                if orientation_meta is not None:
                    metadata["orientation"] = orientation_meta
                    metadata["processing_steps"].append("orientation_correction")

            # Step 3: Perspective Correction
            if self.enable_perspective_correction and metadata["document_corners"] is not None:
                image, perspective_matrix, method = self._correct_perspective(image, metadata["document_corners"])
                metadata["perspective_matrix"] = perspective_matrix
                metadata["perspective_method"] = method
                metadata["processing_steps"].append("perspective_correction")

            # Step 4: Optional padding cleanup
            if self.enable_padding_cleanup:
                cleaned = self._apply_padding_cleanup(image)
                if cleaned is not None:
                    image = cleaned
                    metadata["processing_steps"].append("padding_cleanup")

            # Step 5: General Image Enhancement
            if self.enable_enhancement:
                if self.enhancement_method == "office_lens":
                    image, enhancements = self._enhance_image_office_lens(image)
                else:  # conservative
                    image, enhancements = self._enhance_image(image)
                metadata["enhancement_applied"].extend(enhancements)
                metadata["processing_steps"].append("image_enhancement")

            # Step 6: Text-Specific Enhancement
            if self.enable_text_enhancement:
                image = self._enhance_text_regions(image)
                metadata["processing_steps"].append("text_enhancement")

            # Step 7: Resize to target size (optional)
            if self.enable_final_resize and self.target_size is not None:
                image = self._resize_to_target(image)
                metadata["processing_steps"].append("resize_to_target")

            metadata["final_shape"] = image.shape

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            # Return original image if processing fails
            if self.enable_final_resize and self.target_size is not None:
                image = self._resize_to_target(original_image)
            else:
                image = original_image
            metadata["error"] = str(e)
            metadata["processing_steps"] = ["fallback_resize"]
            metadata["final_shape"] = image.shape

        return {"image": image, "metadata": metadata}

    def _ensure_doctr(self, feature: str) -> bool:
        if self.doctr_available:
            return True
        if feature not in self._warned_features:
            self.logger.warning("python-doctr is required for %s but is not installed; skipping this step.", feature)
            self._warned_features.add(feature)
        return False

    def _apply_orientation_correction(
        self,
        image: np.ndarray,
        corners: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any] | None]:
        if not self._ensure_doctr("orientation_correction"):
            return image, corners, None
        if estimate_page_angle is None or doctr_rotate_image is None:
            return image, corners, None
        if corners.shape != (4, 2):
            return image, corners, None

        height, width = image.shape[:2]
        rel_corners = corners.astype(np.float32).copy()
        rel_corners[:, 0] /= max(width, 1)
        rel_corners[:, 1] /= max(height, 1)

        angle = float(estimate_page_angle(np.expand_dims(rel_corners, axis=0)))
        if not np.isfinite(angle):
            return image, corners, None

        metadata: dict[str, Any] = {
            "original_angle": angle,
            "angle_correction": 0.0,
        }

        if abs(angle) < self.orientation_angle_threshold:
            return image, corners, metadata

        rotated = doctr_rotate_image(
            image,
            -angle,
            expand=self.orientation_expand_canvas,
            preserve_origin_shape=self.orientation_preserve_original_shape,
        )  # type: ignore

        new_corners = self._detect_document_boundaries(rotated)
        if new_corners is None:
            self.logger.debug(
                "Orientation correction applied (angle=%.2f°) but redetection failed; keeping original corners.",
                angle,
            )
            metadata["angle_correction"] = -angle
            metadata["redetection_success"] = False
            metadata["processed_shape"] = tuple(int(dim) for dim in rotated.shape)
            return rotated, corners, metadata

        metadata["angle_correction"] = -angle
        metadata["redetection_success"] = True
        metadata["processed_shape"] = tuple(int(dim) for dim in rotated.shape)
        return rotated, new_corners, metadata

    def _apply_padding_cleanup(self, image: np.ndarray) -> np.ndarray | None:
        if not self._ensure_doctr("padding_cleanup"):
            return None
        if doctr_remove_image_padding is None:
            return None
        cleaned = doctr_remove_image_padding(image)
        return cleaned if cleaned.size != 0 and cleaned.shape != image.shape else None

    def _normalize_corners(self, corners: np.ndarray, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        norm_corners = corners.astype(np.float32).copy()
        norm_corners[:, 0] /= max(width, 1)
        norm_corners[:, 1] /= max(height, 1)
        return norm_corners

    def _compute_perspective_targets(
        self,
        corners: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
        tl, tr, br, bl = corners.astype(np.float32)

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(round(width_a)), int(round(width_b)), 1)

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(round(height_a)), int(round(height_b)), 1)

        dst_points = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype=np.float32,
        )

        src_points = np.array([tl, tr, br, bl], dtype=np.float32)
        return src_points, (max_width, max_height), dst_points

    def _doctr_perspective_correction(self, image: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._ensure_doctr("doctr_geometry"):
            raise RuntimeError("docTR geometry helpers unavailable")
        if extract_rcrops is None:
            raise RuntimeError("docTR extract_rcrops not available")

        norm_corners = self._normalize_corners(corners, image)
        crops = extract_rcrops(
            image,
            np.expand_dims(norm_corners, axis=0),
            assume_horizontal=self.doctr_assume_horizontal,
        )  # type: ignore
        if not crops:
            raise RuntimeError("docTR extract_rcrops returned no crops")

        corrected = crops[0]
        src_points, (max_width, max_height), dst_points = self._compute_perspective_targets(corners)
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        if corrected.shape[0] != max_height or corrected.shape[1] != max_width:
            corrected = cv2.resize(corrected, (max_width, max_height), interpolation=cv2.INTER_LINEAR)

        return corrected, perspective_matrix

    def _correct_perspective(
        self,
        image: np.ndarray,
        corners: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if self.use_doctr_geometry:
            try:
                corrected, matrix = self._doctr_perspective_correction(image, corners)
                return corrected, matrix, "doctr_rcrop"
            except Exception as error:  # pragma: no cover - best-effort fallback
                self.logger.warning("docTR perspective correction failed (%s); falling back to OpenCV.", error)

        corrected, matrix = self._opencv_perspective_correction(image, corners)
        return corrected, matrix, "opencv"

    def _detect_document_boundaries(self, image: np.ndarray) -> np.ndarray | None:
        """
        Detect document boundaries using layered fallbacks so docTR geometry can run.

        Args:
            image: Input image

        Returns:
            Corner coordinates of detected document (4x2 array) or None
        """
        self._last_detection_method = None

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape[:2]
        min_area = self.document_detection_min_area_ratio * float(height * width)

        corners = self._detect_document_from_edges(gray, min_area)
        if corners is not None:
            self._last_detection_method = "canny_contour"
            return corners

        if self.document_detection_use_adaptive:
            corners = self._detect_document_with_adaptive(gray, min_area)
            if corners is not None:
                self._last_detection_method = "adaptive_threshold"
                return corners

        if self.document_detection_use_fallback_box:
            corners = self._fallback_document_bounding_box(gray, min_area)
            if corners is not None:
                self._last_detection_method = "bounding_box"
                return corners

        self._last_detection_method = "failed"
        return None

    def _detect_document_from_edges(self, gray: np.ndarray, min_area: float) -> np.ndarray | None:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        return self._extract_corners_from_binary(dilated, min_area)

    def _detect_document_with_adaptive(self, gray: np.ndarray, min_area: float) -> np.ndarray | None:
        # Invert adaptive threshold so document regions become foreground
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            15,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
        return self._extract_corners_from_binary(closed, min_area)

    def _extract_corners_from_binary(self, binary: np.ndarray, min_area: float) -> np.ndarray | None:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = self._select_document_contour(contours, min_area)
        if contour is None:
            return None

        corners = self._approximate_corners(contour)
        if corners is not None:
            return self._order_corners(corners)
        return None

    def _select_document_contour(self, contours: Sequence[np.ndarray], min_area: float) -> np.ndarray | None:
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= max(min_area, 1.0)]
        if not valid_contours:
            return None
        return max(valid_contours, key=cv2.contourArea)

    def _approximate_corners(self, contour: np.ndarray) -> np.ndarray | None:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None

        approximations = [0.02, 0.03, 0.04, 0.05]
        for epsilon_factor in approximations:
            approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)
            if approx.shape[0] == 4:
                return approx.reshape(4, 2)

        hull = cv2.convexHull(contour)
        if hull is not None and hull.shape[0] >= 4:
            hull_perimeter = cv2.arcLength(hull, True)
            if hull_perimeter > 0:
                for epsilon_factor in approximations:
                    approx = cv2.approxPolyDP(hull, epsilon_factor * hull_perimeter, True)
                    if approx.shape[0] == 4:
                        return approx.reshape(4, 2)

        return None

    def _fallback_document_bounding_box(self, gray: np.ndarray, min_area: float) -> np.ndarray | None:
        # Use Otsu threshold to find foreground content; invert to treat text as foreground
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        coords = cv2.findNonZero(cleaned)
        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)
        if w <= 10 or h <= 10:
            return None

        if float(w * h) < max(min_area * 0.6, 1.0):
            return None

        corners = np.array(
            [
                [x, y],
                [x + w - 1, y],
                [x + w - 1, y + h - 1],
                [x, y + h - 1],
            ],
            dtype=np.float32,
        )
        return corners

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners in consistent order [top-left, top-right, bottom-right, bottom-left]
        using coordinate sums and differences for reliable assignment.

        Args:
            corners: 4 corner points

        Returns:
            Ordered corners [top-left, top-right, bottom-right, bottom-left]
        """
        # Calculate sum and difference of coordinates for each corner
        sums = corners.sum(axis=1)
        diffs = np.diff(corners, axis=1).flatten()

        # Order: [top-left, top-right, bottom-right, bottom-left]
        # Top-left: smallest sum
        # Bottom-right: largest sum
        # Top-right: smallest difference
        # Bottom-left: largest difference

        ordered_corners = np.zeros((4, 2), dtype=np.float32)

        # Top-left: smallest sum
        ordered_corners[0] = corners[np.argmin(sums)]
        # Bottom-right: largest sum
        ordered_corners[2] = corners[np.argmax(sums)]
        # Top-right: smallest difference (most rightward of remaining)
        remaining = [i for i in range(4) if i not in [np.argmin(sums), np.argmax(sums)]]
        ordered_corners[1] = corners[remaining[np.argmin(diffs[remaining])]]
        # Bottom-left: largest difference (most leftward of remaining)
        ordered_corners[3] = corners[remaining[np.argmax(diffs[remaining])]]

        return ordered_corners

    def _opencv_perspective_correction(self, image: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Correct perspective distortion using OpenCV warpPerspective based on provided corners.

        Args:
            image: Input image
            corners: Document corner coordinates [tl, tr, br, bl]

        Returns:
            Tuple of (corrected_image, perspective_matrix)
        """
        src_points, (max_width, max_height), dst_points = self._compute_perspective_targets(corners)
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        corrected = cv2.warpPerspective(image, perspective_matrix, (max_width, max_height), flags=cv2.INTER_LINEAR)

        return corrected, perspective_matrix

    def _enhance_image(self, image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """
        Apply conservative image enhancement techniques.
        Uses milder parameters to avoid over-enhancement.

        Args:
            image: Input image

        Returns:
            Tuple of (enhanced_image, list_of_applied_enhancements)
        """
        enhanced = image.copy()
        applied_enhancements = ["clahe_mild", "bilateral_filter_mild"]

        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)

        # Apply CLAHE with milder parameters (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Apply very mild bilateral filter for subtle noise reduction
        enhanced = cv2.bilateralFilter(enhanced, 5, 25, 25)

        # Skip unsharp masking for now - it can be too aggressive
        # Only apply if specifically needed for very blurry images

        return enhanced, applied_enhancements

    def _enhance_image_office_lens(self, image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """
        Apply sophisticated image enhancement techniques inspired by Office Lens.
        Includes gamma correction, CLAHE, saturation boost, sharpening, and noise reduction.

        Args:
            image: Input image

        Returns:
            Tuple of (enhanced_image, list_of_applied_enhancements)
        """
        enhanced = image.copy()
        applied_enhancements = [
            "gamma_correction",
            "clahe_lab",
            "saturation_boost",
            "sharpening",
            "noise_reduction",
        ]

        # Step 1: Mild gamma correction for color correction
        gamma = 1.1
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)

        # Step 2: LAB Color Space with CLAHE
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Gentle CLAHE with larger tile size for more natural results
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
        l_channel = clahe.apply(l)

        # Merge channels
        lab = cv2.merge([l_channel, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Step 3: Smart saturation boost using HSV color space
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # Saturation mask using bilateral filter to preserve edges
        s_clean = cv2.bilateralFilter(s, 9, 75, 75)
        s = cv2.addWeighted(s, 1.2, s_clean, 0.3, 0)
        s = np.clip(s, 20, 230).astype(np.uint8)

        hsv = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Step 4: Controlled sharpening using unsharp masking
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

        # Step 5: Final noise reduction with bilateral filter
        final = cv2.bilateralFilter(sharpened, 9, 75, 75)

        return final, applied_enhancements

    def _enhance_text_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Apply text-specific enhancement techniques.

        Args:
            image: Input image

        Returns:
            Enhanced image with improved text regions
        """
        # Convert to grayscale for text processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply morphological operations to enhance text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Adaptive thresholding for text binarization
        thresh = cv2.adaptiveThreshold(morphed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Combine with original image for enhanced contrast
        # Convert original image to grayscale to avoid color artifacts
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        enhanced_text_gray = cv2.bitwise_and(image_gray, thresh)

        # Convert back to 3 channels for blending with original color image
        enhanced_text_color = cv2.cvtColor(enhanced_text_gray, cv2.COLOR_GRAY2RGB)

        # Blend with original for natural appearance
        alpha = 0.7
        return cv2.addWeighted(image, 1 - alpha, enhanced_text_color, alpha, 0)

    def _resize_to_target(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target dimensions while maintaining aspect ratio.

        Args:
            image: Input image

        Returns:
            Resized image
        """
        if self.target_size is None:
            return image

        target_width, target_height = self.target_size

        # Use longest max size approach (similar to current transforms)
        height, width = image.shape[:2]
        scale = min(target_width / width, target_height / height)

        new_width = round(width * scale)
        new_height = round(height * scale)

        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Calculate padding to center the image properly
        pad_left = (target_width - new_width) // 2
        pad_top = (target_height - new_height) // 2
        pad_right = target_width - new_width - pad_left
        pad_bottom = target_height - new_height - pad_top

        return cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )


class LensStylePreprocessorAlbumentations:
    """
    Albumentations-compatible wrapper for Lens-style preprocessing.
    """

    def __init__(self, preprocessor: DocumentPreprocessor):
        self.preprocessor = preprocessor

    def __call__(self, image, **kwargs):
        # Apply preprocessing
        result = self.preprocessor(image)

        # Return processed image (Albumentations expects just the image)
        return result["image"]

    def get_transform_init_args_names(self):
        return []
