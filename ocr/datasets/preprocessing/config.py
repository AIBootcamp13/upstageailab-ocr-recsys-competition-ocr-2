"""Configuration objects for the preprocessing pipeline."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class EnhancementMethod(str, Enum):
    """Valid enhancement methods for document preprocessing."""

    CONSERVATIVE = "conservative"
    OFFICE_LENS = "office_lens"


class DocumentPreprocessorConfig(BaseModel):
    """Validated configuration for the document preprocessing pipeline."""

    # Core processing flags
    enable_document_detection: bool = Field(default=True, description="Enable automatic document boundary detection")
    enable_perspective_correction: bool = Field(default=True, description="Enable perspective distortion correction")
    enable_enhancement: bool = Field(default=True, description="Enable image enhancement techniques")

    enhancement_method: EnhancementMethod = Field(default=EnhancementMethod.CONSERVATIVE, description="Enhancement algorithm to use")

    @field_validator("enhancement_method", mode="before")
    @classmethod
    def validate_enhancement_method(cls, v):
        """Convert string enhancement methods to enum for backward compatibility."""
        if isinstance(v, str):
            try:
                return EnhancementMethod(v.lower())
            except ValueError:
                raise ValueError(f"enhancement_method must be 'conservative' or 'office_lens', got '{v}'")
        return v

    # Size and resize settings
    target_size: tuple[int, int] | None = Field(default=(640, 640), description="Target output size (width, height) in pixels")
    enable_final_resize: bool = Field(default=True, description="Enable final resize to target_size")

    # Orientation correction settings
    enable_orientation_correction: bool = Field(default=False, description="Enable automatic orientation correction")
    orientation_angle_threshold: float = Field(
        default=2.0, gt=0.0, le=45.0, description="Minimum angle threshold for orientation correction (degrees)"
    )
    orientation_expand_canvas: bool = Field(default=True, description="Expand canvas when rotating to preserve content")
    orientation_preserve_original_shape: bool = Field(
        default=False, description="Preserve original image shape after orientation correction"
    )

    # DOCTR integration settings
    use_doctr_geometry: bool = Field(default=False, description="Use DOCTR for geometric document detection")
    doctr_assume_horizontal: bool = Field(default=False, description="Assume horizontal text orientation in DOCTR")

    # Advanced processing settings
    enable_padding_cleanup: bool = Field(default=False, description="Enable cleanup of artificial padding")

    # Document detection parameters
    document_detection_min_area_ratio: float = Field(
        default=0.18, ge=0.0, le=1.0, description="Minimum area ratio for valid document detection (0.0-1.0)"
    )
    document_detection_use_adaptive: bool = Field(default=True, description="Use adaptive thresholding for document detection")
    document_detection_use_fallback_box: bool = Field(default=True, description="Use fallback bounding box when detection fails")
    document_detection_use_camscanner: bool = Field(default=False, description="Use CamScanner-style document detection")
    document_detection_use_doctr_text: bool = Field(default=False, description="Use DOCTR text detection for document boundaries")

    @field_validator("target_size", mode="before")
    @classmethod
    def validate_target_size(cls, v):
        """Validate target_size as tuple of positive integers or None."""
        if v is None:
            return None
        if isinstance(v, list | tuple) and len(v) == 2:
            width, height = v
            if isinstance(width, int | float) and isinstance(height, int | float):
                return (int(width), int(height))
        raise ValueError("target_size must be None or a tuple of two numbers (width, height)")

    @field_validator("target_size")
    @classmethod
    def validate_target_size_positive(cls, v):
        """Ensure target_size dimensions are positive."""
        if v is None:
            return None
        width, height = v
        if width <= 0 or height <= 0:
            raise ValueError("target_size dimensions must be positive")
        if width > 10000 or height > 10000:
            raise ValueError("target_size dimensions cannot exceed 10000 pixels")
        return v

    @model_validator(mode="after")
    def validate_cross_field_dependencies(self):
        """Validate interdependent configuration settings."""
        # If document detection is disabled but perspective correction is enabled,
        # perspective correction will simply be skipped in the pipeline
        # This is allowed for flexibility

        # Validate DOCTR settings consistency
        if self.use_doctr_geometry and not self.enable_document_detection:
            raise ValueError("use_doctr_geometry requires enable_document_detection to be True")

        return self

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        data = self.model_dump()
        # Convert enum to string for backward compatibility
        data["enhancement_method"] = self.enhancement_method.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> DocumentPreprocessorConfig:
        """Create configuration from dictionary."""
        # Handle enum conversion
        if "enhancement_method" in data:
            data = data.copy()
            data["enhancement_method"] = EnhancementMethod(data["enhancement_method"])
        return cls(**data)


__all__ = ["DocumentPreprocessorConfig", "EnhancementMethod"]
