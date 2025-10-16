"""
AI_DOCS: Image Utils - Image Processing & Loading Utilities

This module provides specialized image processing utilities for the OCR dataset:
- EXIF-aware image loading with orientation correction
- RGB conversion and normalization
- Memory-efficient PIL image handling
- TurboJPEG integration for performance

ARCHITECTURE OVERVIEW:
- Utilities extracted from monolithic dataset code
- Focus on image loading and preprocessing
- Memory-safe PIL image lifecycle management
- Performance optimizations for large datasets

DATA CONTRACTS:
- Input: Path objects or strings (file paths)
- Output: PIL Images or numpy arrays
- Configuration: ImageLoadingConfig (Pydantic model)
- Metadata: ImageData with EXIF information

CORE CONSTRAINTS:
- ALWAYS close PIL images to prevent memory leaks
- PRESERVE EXIF orientation correction logic
- USE TurboJPEG when available for performance
- VALIDATE image loading before processing
- MAINTAIN backward compatibility with existing code

PERFORMANCE FEATURES:
- Lazy loading prevents memory bloat
- TurboJPEG acceleration for JPEG files
- EXIF orientation correction without full image rotation
- Memory-efficient image processing pipeline

VALIDATION REQUIREMENTS:
- Check file existence before loading
- Validate image formats and dimensions
- Handle corrupted image files gracefully
- Provide meaningful error messages

RELATED DOCUMENTATION:
- Base Dataset: ocr/datasets/base.py
- Configuration: ocr/datasets/schemas.py
- EXIF Handling: ocr/utils/orientation.py
- Performance Guide: docs/ai_handbook/04_performance/

MIGRATION NOTES:
- Utilities extracted from ValidatedOCRDataset._load_image_data
- Pydantic models ensure data integrity
- Memory management prevents leaks in long-running training
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ocr.datasets.schemas import ImageLoadingConfig
from ocr.utils.image_loading import load_image_optimized


def safe_get_image_size(image: Image.Image | np.ndarray) -> tuple[int, int]:
    """Return ``(width, height)`` for a PIL image or numpy array."""
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
        return int(width), int(height)
    if hasattr(image, "size"):
        width, height = image.size
        return int(width), int(height)
    raise TypeError(f"Unsupported image type: {type(image)}")


def load_pil_image(path: Path, config: ImageLoadingConfig) -> Image.Image:
    """
    AI_DOCS: PIL Image Loading with EXIF Support

    Loads PIL image with proper EXIF orientation handling.

    CRITICAL CONSTRAINTS:
    - ALWAYS apply EXIF orientation correction
    - USE TurboJPEG if available and enabled
    - VALIDATE file exists before loading
    - RETURN PIL Image (caller responsible for closing)
    - HANDLE OSError for corrupted files

    Memory Responsibility: Caller MUST close returned PIL image
    """
    return load_image_optimized(
        path,
        use_turbojpeg=config.use_turbojpeg,
        turbojpeg_fallback=config.turbojpeg_fallback,
    )


def ensure_rgb(image: Image.Image) -> Image.Image:
    """
    AI_DOCS: RGB Conversion Utility

    Converts any PIL image mode to RGB format.

    CRITICAL CONSTRAINTS:
    - ALWAYS convert to RGB (3 channels)
    - PRESERVE image dimensions
    - HANDLE all PIL image modes (L, P, RGBA, etc.)
    - RETURN new PIL image (original unchanged)

    Use Case: Ensures consistent 3-channel input for neural networks
    """
    if image.mode == "RGB":
        return image.copy()
    return image.convert("RGB")


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    AI_DOCS: PIL to NumPy Conversion

    Converts PIL image to numpy array with proper dtype.

    CRITICAL CONSTRAINTS:
    - PRESERVE image dimensions and channels
    - USE uint8 dtype for original images
    - RETURN C-contiguous arrays for PyTorch compatibility
    - DO NOT modify original PIL image

    Output Format: (H, W, C) with dtype=uint8
    """
    return np.array(image)


def prenormalize_imagenet(image_array: np.ndarray) -> np.ndarray:
    """
    AI_DOCS: Dataset Normalization

    Applies dataset-specific normalization to image array.

    CRITICAL CONSTRAINTS:
    - USE standard dataset mean/std values
    - CONVERT to float32 dtype
    - APPLY per-channel normalization
    - RETURN normalized array (0-1 range → normalized)

    Output: float32 array with dataset normalization applied
    """
    if image_array.dtype != np.float32:
        image_array = image_array.astype(np.float32)
    image_array /= 255.0
    mean = np.array([0.5900, 0.5610, 0.5292], dtype=np.float32)
    std = np.array([0.2136, 0.2290, 0.2511], dtype=np.float32)
    return (image_array - mean) / std


__all__ = [
    "ensure_rgb",
    "load_pil_image",
    "pil_to_numpy",
    "prenormalize_imagenet",
    "safe_get_image_size",
]
