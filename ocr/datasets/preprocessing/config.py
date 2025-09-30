"""Configuration objects for the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DocumentPreprocessorConfig:
    """Configuration values for the document preprocessing pipeline."""

    enable_document_detection: bool = True
    enable_perspective_correction: bool = True
    enable_enhancement: bool = True
    enable_text_enhancement: bool = False
    enhancement_method: str = "conservative"
    target_size: tuple[int, int] | None = (640, 640)
    enable_final_resize: bool = True
    enable_orientation_correction: bool = False
    orientation_angle_threshold: float = 2.0
    orientation_expand_canvas: bool = True
    orientation_preserve_original_shape: bool = False
    use_doctr_geometry: bool = False
    doctr_assume_horizontal: bool = False
    enable_padding_cleanup: bool = False
    document_detection_min_area_ratio: float = 0.18
    document_detection_use_adaptive: bool = True
    document_detection_use_fallback_box: bool = True

    def __post_init__(self) -> None:
        if self.target_size is not None:
            width, height = self.target_size
            self.target_size = (int(width), int(height))
        self.document_detection_min_area_ratio = float(max(0.0, min(self.document_detection_min_area_ratio, 1.0)))


__all__ = ["DocumentPreprocessorConfig"]
