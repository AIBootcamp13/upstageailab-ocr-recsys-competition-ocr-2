"""Data contracts for inference results and predictions using Pydantic v2.

These models validate the data structures used in the inference pipeline to prevent
datatype mismatches and ensure consistency across the Streamlit UI.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _InferenceBase(BaseModel):
    """Common configuration for inference data models."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)


class Predictions(_InferenceBase):
    """Data contract for OCR inference predictions."""

    polygons: str = Field(..., description="Pipe-separated polygon coordinates as strings")
    texts: list[str] = Field(default_factory=list, description="List of detected text strings")
    confidences: list[float] = Field(default_factory=list, description="Confidence scores for each detection")

    @field_validator("polygons")
    @classmethod
    def _validate_polygons(cls, value: str) -> str:
        """Validate that polygons string is properly formatted."""
        if not isinstance(value, str):
            raise TypeError("Polygons must be a string")
        if not value.strip():
            return value  # Allow empty polygons
        # Basic validation that it's pipe-separated coordinate strings
        for polygon_str in value.split("|"):
            if not polygon_str.strip():
                continue
            coords = polygon_str.split(",")
            if len(coords) < 8 or len(coords) % 2 != 0:
                raise ValueError(f"Invalid polygon format: {polygon_str}. Must have even number of coordinates >= 8")
            try:
                [float(coord) for coord in coords if coord.strip()]
            except ValueError as exc:
                raise ValueError(f"Polygon coordinates must be numeric: {polygon_str}") from exc
        return value

    @field_validator("confidences")
    @classmethod
    def _validate_confidences(cls, value: list[float]) -> list[float]:
        """Validate confidence scores are between 0 and 1."""
        for conf in value:
            if not (0.0 <= conf <= 1.0):
                raise ValueError(f"Confidence scores must be between 0.0 and 1.0, got {conf}")
        return value

    @model_validator(mode="after")
    def _validate_consistency(self) -> Predictions:
        """Validate that texts and confidences have matching lengths after all fields are set."""
        if len(self.texts) != len(self.confidences):
            raise ValueError(f"texts and confidences must have same length: {len(self.texts)} vs {len(self.confidences)}")
        return self


class PreprocessingInfo(_InferenceBase):
    """Data contract for preprocessing metadata and results."""

    enabled: bool = Field(default=False, description="Whether preprocessing was enabled")
    metadata: dict[str, Any] | None = Field(default=None, description="Preprocessing metadata from docTR")
    original: np.ndarray | None = Field(default=None, description="Original image before preprocessing")
    processed: np.ndarray | None = Field(default=None, description="Processed image after preprocessing")
    doctr_available: bool = Field(default=False, description="Whether docTR library is available")
    mode: str = Field(default="docTR:off", description="Preprocessing mode indicator")
    error: str | None = Field(default=None, description="Error message if preprocessing failed")

    @field_validator("original", "processed")
    @classmethod
    def _validate_image(cls, value: np.ndarray | None) -> np.ndarray | None:
        """Validate image arrays have correct shape and type."""
        if value is None:
            return value
        if not isinstance(value, np.ndarray):
            raise TypeError("Image must be a numpy array")
        if value.ndim != 3 or value.shape[2] != 3:
            raise ValueError(f"Image must be shaped (H, W, 3); received {value.shape}")
        return value

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        """Validate preprocessing mode format."""
        if value not in ["docTR:on", "docTR:off"]:
            raise ValueError(f"Mode must be 'docTR:on' or 'docTR:off', got {value}")
        return value


class InferenceResult(_InferenceBase):
    """Data contract for complete inference result."""

    filename: str = Field(..., description="Name of the processed file")
    success: bool = Field(default=True, description="Whether inference succeeded")
    image: np.ndarray = Field(..., description="Processed image array")
    predictions: Predictions = Field(..., description="OCR predictions")
    preprocessing: PreprocessingInfo = Field(..., description="Preprocessing information")
    error: str | None = Field(default=None, description="Error message if inference failed")

    @field_validator("image")
    @classmethod
    def _validate_image(cls, value: np.ndarray) -> np.ndarray:
        """Validate result image has correct shape and type."""
        if not isinstance(value, np.ndarray):
            raise TypeError("Image must be a numpy array")
        if value.ndim != 3 or value.shape[2] != 3:
            raise ValueError(f"Image must be shaped (H, W, 3); received {value.shape}")
        return value

    @field_validator("error")
    @classmethod
    def _validate_error_consistency(cls, value: str | None, info) -> str | None:
        """Validate error field consistency with success flag."""
        success = info.data.get("success", True)
        if success and value is not None:
            raise ValueError("Cannot have error message when success is True")
        if not success and value is None:
            raise ValueError("Must provide error message when success is False")
        return value
