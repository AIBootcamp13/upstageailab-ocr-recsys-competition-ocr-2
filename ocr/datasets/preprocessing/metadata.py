"""Metadata structures for the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ImageShape(BaseModel):
    """Validated image shape specification with dimension constraints."""

    height: int = Field(gt=0, le=10000, description="Image height in pixels")
    width: int = Field(gt=0, le=10000, description="Image width in pixels")
    channels: int = Field(ge=1, le=4, description="Number of color channels")

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> ImageShape:
        """Create ImageShape from numpy array dimensions.

        Args:
            array: Input numpy array

        Returns:
            ImageShape instance with validated dimensions
        """
        if len(array.shape) < 2:
            raise ValueError(f"Array must have at least 2 dimensions, got {len(array.shape)}")

        height, width = array.shape[:2]
        channels = array.shape[2] if len(array.shape) > 2 else 1

        return cls(height=height, width=width, channels=channels)


class DocumentMetadata(BaseModel):
    """Structured metadata describing preprocessing outcomes with validated data contracts."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    original_shape: ImageShape | tuple[int, ...] = Field(
        description="Original image shape as ImageShape or tuple for backward compatibility"
    )
    final_shape: tuple[int, ...] | None = Field(default=None, description="Final processed image shape")
    processing_steps: list[str] = Field(default_factory=list, description="List of processing steps applied")
    document_corners: np.ndarray | None = Field(default=None, description="Detected document corner coordinates")
    document_detection_method: str | None = Field(default=None, description="Method used for document detection")
    perspective_matrix: np.ndarray | None = Field(default=None, description="Perspective transformation matrix")
    perspective_method: str | None = Field(default=None, description="Method used for perspective correction")
    enhancement_applied: list[str] = Field(default_factory=list, description="List of enhancement techniques applied")
    orientation: dict[str, Any] | None = Field(default=None, description="Orientation detection results and metadata")
    error: str | None = Field(default=None, description="Error message if processing failed")

    # Intermediate images for debugging
    image_after_document_detection: np.ndarray | None = Field(default=None, description="Image after document detection")
    image_after_orientation_correction: np.ndarray | None = Field(default=None, description="Image after orientation correction")
    image_after_perspective_correction: np.ndarray | None = Field(default=None, description="Image after perspective correction")
    image_after_enhancement: np.ndarray | None = Field(default=None, description="Image after enhancement")

    @field_validator("original_shape", mode="before")
    @classmethod
    def validate_original_shape(cls, v: Any) -> ImageShape | tuple[int, ...]:
        """Convert tuple shapes to ImageShape or validate existing ImageShape."""
        if isinstance(v, tuple):
            # Convert tuple to ImageShape for validation, but allow invalid shapes to pass through
            if len(v) >= 2:
                try:
                    height, width = v[:2]
                    channels = v[2] if len(v) > 2 else 1
                    return ImageShape(height=height, width=width, channels=channels)
                except Exception:
                    # If validation fails, keep as tuple
                    return v  # type: ignore
            else:
                # Invalid shape (less than 2 dimensions), keep as tuple
                return v  # type: ignore
        elif isinstance(v, ImageShape):
            return v
        else:
            raise ValueError(f"original_shape must be ImageShape or tuple, got {type(v)}")

    @field_validator("document_corners", mode="before")
    @classmethod
    def validate_document_corners(cls, v: Any) -> np.ndarray | None:
        """Validate document corners as numpy array."""
        if v is None:
            return None
        if not isinstance(v, np.ndarray):
            try:
                v = np.array(v)
            except Exception as e:
                raise ValueError(f"document_corners must be convertible to numpy array: {e}")
        if v.size == 0:
            raise ValueError("document_corners cannot be empty")
        return v

    @field_validator("perspective_matrix", mode="before")
    @classmethod
    def validate_perspective_matrix(cls, v: Any) -> np.ndarray | None:
        """Validate perspective matrix as numpy array."""
        if v is None:
            return None
        if not isinstance(v, np.ndarray):
            try:
                v = np.array(v)
            except Exception as e:
                raise ValueError(f"perspective_matrix must be convertible to numpy array: {e}")
        # Perspective matrix should be 3x3
        if v.shape != (3, 3):
            raise ValueError(f"perspective_matrix must be 3x3, got shape {v.shape}")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility.

        Returns:
            Dictionary representation with original types preserved
        """
        data: dict[str, Any] = {
            "original_shape": self._get_original_shape_as_tuple(),
            "processing_steps": list(self.processing_steps),
            "document_corners": self.document_corners,
            "perspective_matrix": self.perspective_matrix,
            "enhancement_applied": list(self.enhancement_applied),
        }

        # Add optional fields only if they are not None
        if self.document_detection_method is not None:
            data["document_detection_method"] = self.document_detection_method
        if self.perspective_method is not None:
            data["perspective_method"] = self.perspective_method
        if self.orientation is not None:
            data["orientation"] = self.orientation
        if self.error is not None:
            data["error"] = self.error
        if self.final_shape is not None:
            data["final_shape"] = self.final_shape

        # Add intermediate images for debugging
        if self.image_after_document_detection is not None:
            data["image_after_document_detection"] = self.image_after_document_detection
        if self.image_after_orientation_correction is not None:
            data["image_after_orientation_correction"] = self.image_after_orientation_correction
        if self.image_after_perspective_correction is not None:
            data["image_after_perspective_correction"] = self.image_after_perspective_correction
        if self.image_after_enhancement is not None:
            data["image_after_enhancement"] = self.image_after_enhancement

        return data

    def _get_original_shape_as_tuple(self) -> tuple[int, ...]:
        """Get original_shape as tuple for backward compatibility."""
        if isinstance(self.original_shape, ImageShape):
            return (self.original_shape.height, self.original_shape.width, self.original_shape.channels)
        return self.original_shape


@dataclass(slots=True)
class PreprocessingState:
    """Mutable state passed between preprocessing stages."""

    image: np.ndarray
    metadata: DocumentMetadata
    corners: np.ndarray | None = None

    def update_final_shape(self) -> None:
        self.metadata.final_shape = tuple(int(dim) for dim in self.image.shape)


__all__ = ["ImageShape", "DocumentMetadata", "PreprocessingState"]
