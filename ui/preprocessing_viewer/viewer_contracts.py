"""
Pydantic v2 data contracts for Streamlit Preprocessing Viewer components.

This module defines shared data models for viewer configuration, ROI requests,
and export payloads, following the established preprocessing data contract standards.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, validate_call


class ViewerConfiguration(BaseModel):
    """Configuration model for preprocessing viewer pipeline."""

    model_config = {"arbitrary_types_allowed": True}

    # Core processing flags
    enable_document_detection: bool = Field(default=True, description="Enable document corner detection")
    enable_perspective_correction: bool = Field(default=True, description="Enable perspective distortion correction")
    enable_enhancement: bool = Field(default=True, description="Enable image enhancement")
    enhancement_method: Literal["conservative", "moderate", "aggressive"] = Field(
        default="conservative", description="Enhancement intensity level"
    )

    # Color preprocessing
    enable_color_preprocessing: bool = Field(default=True, description="Enable color preprocessing steps")
    convert_to_grayscale: bool = Field(default=False, description="Convert image to grayscale")
    color_inversion: bool = Field(default=False, description="Invert image colors")

    # Advanced processing
    enable_document_flattening: bool = Field(default=False, description="Enable document geometric flattening")
    enable_orientation_correction: bool = Field(default=False, description="Enable orientation correction")
    enable_noise_elimination: bool = Field(default=True, description="Enable noise elimination")
    enable_brightness_adjustment: bool = Field(default=True, description="Enable intelligent brightness adjustment")

    # Size settings
    target_size: list[int] = Field(default=[640, 640], description="Target image size [width, height]")
    enable_final_resize: bool = Field(default=True, description="Enable final resize to target size")

    # Detection parameters
    document_detection_min_area_ratio: float = Field(default=0.18, ge=0.0, le=1.0, description="Minimum document area ratio for detection")
    document_detection_use_adaptive: bool = Field(default=True, description="Use adaptive detection thresholds")
    document_detection_use_fallback_box: bool = Field(default=True, description="Use fallback bounding box detection")

    # Orientation settings
    orientation_angle_threshold: float = Field(
        default=2.0, ge=0.0, le=180.0, description="Minimum angle threshold for orientation correction"
    )
    orientation_expand_canvas: bool = Field(default=True, description="Expand canvas when rotating")
    orientation_preserve_original_shape: bool = Field(default=False, description="Preserve original image shape")

    # DOCTR settings
    use_doctr_geometry: bool = Field(default=False, description="Use DOCTR for geometry detection")
    doctr_assume_horizontal: bool = Field(default=False, description="Assume horizontal document orientation in DOCTR")

    # Advanced settings
    enable_padding_cleanup: bool = Field(default=False, description="Enable padding cleanup")

    @field_validator("target_size")
    @classmethod
    def validate_target_size(cls, v: list[int]) -> list[int]:
        """Validate target size dimensions."""
        if len(v) != 2:
            raise ValueError("target_size must be a list of exactly 2 integers [width, height]")
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError("target_size dimensions must be positive")
        if v[0] > 4096 or v[1] > 4096:
            raise ValueError("target_size dimensions cannot exceed 4096 pixels")
        return v


class ROIRequest(BaseModel):
    """Request model for region of interest selection."""

    x: int = Field(ge=0, description="X coordinate of ROI top-left corner")
    y: int = Field(ge=0, description="Y coordinate of ROI top-left corner")
    width: int = Field(gt=0, description="Width of ROI rectangle")
    height: int = Field(gt=0, description="Height of ROI rectangle")

    @field_validator("width", "height")
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        """Validate ROI dimensions are reasonable."""
        if v > 10000:
            raise ValueError("ROI dimension cannot exceed 10000 pixels")
        return v

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Convert to tuple format (x, y, w, h)."""
        return (self.x, self.y, self.width, self.height)

    @classmethod
    def from_tuple(cls, roi_tuple: tuple[int, int, int, int]) -> ROIRequest:
        """Create ROIRequest from tuple format (x, y, w, h)."""
        x, y, w, h = roi_tuple
        return cls(x=x, y=y, width=w, height=h)


class ExportPayload(BaseModel):
    """Payload model for configuration export operations."""

    configuration: ViewerConfiguration = Field(description="Preprocessing configuration to export")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Export metadata")
    version: str = Field(default="1.0", description="Configuration version")
    exported_at: datetime = Field(default_factory=datetime.utcnow, description="Export timestamp")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version format."""
        if not v.replace(".", "").replace("-", "").isalnum():
            raise ValueError("Version must contain only alphanumeric characters, dots, and hyphens")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "configuration": self.configuration.model_dump(),
            "metadata": self.metadata,
            "version": self.version,
            "exported_at": self.exported_at.isoformat(),
        }


class ViewerErrorResponse(BaseModel):
    """Standardized error response for viewer operations."""

    error_code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error message")
    details: dict[str, Any] | None = Field(default=None, description="Additional error context")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

    @field_validator("error_code")
    @classmethod
    def validate_error_code(cls, v: str) -> str:
        """Validate error code format."""
        if not v.replace("_", "").isalnum():
            raise ValueError("Error code must contain only alphanumeric characters and underscores")
        return v.upper()


# Validation decorators for viewer entry points
@validate_call
def validate_viewer_config(func):
    """Decorator to validate viewer configuration inputs."""

    def wrapper(config: dict[str, Any] | ViewerConfiguration, *args, **kwargs):
        if isinstance(config, dict):
            # Convert dict to ViewerConfiguration for validation
            config = ViewerConfiguration(**config)
        return func(config, *args, **kwargs)

    return wrapper


@validate_call
def validate_roi_request(func):
    """Decorator to validate ROI request inputs."""

    def wrapper(roi: tuple[int, int, int, int] | ROIRequest | None, *args, **kwargs):
        if isinstance(roi, tuple):
            # Convert tuple to ROIRequest for validation
            roi = ROIRequest.from_tuple(roi)
        return func(roi, *args, **kwargs)

    return wrapper


@validate_call
def validate_export_payload(func):
    """Decorator to validate export payload inputs."""

    def wrapper(payload: dict[str, Any] | ExportPayload, *args, **kwargs):
        if isinstance(payload, dict):
            # Convert dict to ExportPayload for validation
            payload = ExportPayload(**payload)
        return func(payload, *args, **kwargs)

    return wrapper


__all__ = [
    "ViewerConfiguration",
    "ROIRequest",
    "ExportPayload",
    "ViewerErrorResponse",
    "validate_viewer_config",
    "validate_roi_request",
    "validate_export_payload",
]
