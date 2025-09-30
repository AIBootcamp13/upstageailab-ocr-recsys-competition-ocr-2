"""Metadata structures for the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class DocumentMetadata:
    """Structured metadata describing preprocessing outcomes."""

    original_shape: Any
    final_shape: tuple[int, ...] | None = None
    processing_steps: list[str] = field(default_factory=list)
    document_corners: np.ndarray | None = None
    document_detection_method: str | None = None
    perspective_matrix: np.ndarray | None = None
    perspective_method: str | None = None
    enhancement_applied: list[str] = field(default_factory=list)
    orientation: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "original_shape": self.original_shape,
            "processing_steps": list(self.processing_steps),
            "document_corners": self.document_corners,
            "perspective_matrix": self.perspective_matrix,
            "enhancement_applied": list(self.enhancement_applied),
        }
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
        return data


@dataclass(slots=True)
class PreprocessingState:
    """Mutable state passed between preprocessing stages."""

    image: np.ndarray
    metadata: DocumentMetadata
    corners: np.ndarray | None = None

    def update_final_shape(self) -> None:
        self.metadata.final_shape = tuple(int(dim) for dim in self.image.shape)


__all__ = ["DocumentMetadata", "PreprocessingState"]
