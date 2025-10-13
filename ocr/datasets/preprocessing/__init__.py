"""Preprocessing submodule exposing modular document preprocessing components."""

from __future__ import annotations

from .advanced_detector import AdvancedDetectionConfig, AdvancedDocumentDetector
from .advanced_preprocessor import (
    AdvancedDocumentPreprocessor,
    AdvancedPreprocessingConfig,
    OfficeLensPreprocessorAlbumentations,
    create_high_accuracy_preprocessor,
    create_office_lens_preprocessor,
)
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
from .validators import (
    ContractValidator,
    CornerArray,
    ImageArray,
    ImageValidator,
    NumpyArray,
    TransformationMatrix,
)

__all__ = [
    "A",
    "ALBUMENTATIONS_AVAILABLE",
    "AdvancedDetectionConfig",
    "AdvancedDocumentDetector",
    "AdvancedDocumentPreprocessor",
    "AdvancedPreprocessingConfig",
    "ContractValidator",
    "CornerArray",
    "DOCTR_AVAILABLE",
    "DocumentMetadata",
    "DocumentPreprocessor",
    "DocumentPreprocessorConfig",
    "ImageArray",
    "ImageValidator",
    "LensStylePreprocessorAlbumentations",
    "NumpyArray",
    "OfficeLensPreprocessorAlbumentations",
    "PreprocessingState",
    "TransformationMatrix",
    "create_high_accuracy_preprocessor",
    "create_office_lens_preprocessor",
    "doctr_remove_image_padding",
    "doctr_rotate_image",
    "estimate_page_angle",
    "extract_rcrops",
]
