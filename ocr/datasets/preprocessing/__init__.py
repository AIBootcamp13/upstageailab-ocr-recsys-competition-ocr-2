"""Preprocessing submodule exposing modular document preprocessing components."""

from __future__ import annotations

from .config import DocumentPreprocessorConfig
from .external import (
    ALBUMENTATIONS_AVAILABLE,
    DOCTR_AVAILABLE,
    A,
    doctr_remove_image_padding,
    doctr_rotate_image,
    estimate_page_angle,
    extract_rcrops,
)
from .metadata import DocumentMetadata, PreprocessingState
from .pipeline import DocumentPreprocessor, LensStylePreprocessorAlbumentations

__all__ = [
    "A",
    "ALBUMENTATIONS_AVAILABLE",
    "DOCTR_AVAILABLE",
    "DocumentMetadata",
    "DocumentPreprocessor",
    "DocumentPreprocessorConfig",
    "LensStylePreprocessorAlbumentations",
    "PreprocessingState",
    "doctr_remove_image_padding",
    "doctr_rotate_image",
    "estimate_page_angle",
    "extract_rcrops",
]
