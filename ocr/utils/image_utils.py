"""Shared image helpers for the validated OCR dataset."""

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
    """Load an image from disk using the configured backend."""
    return load_image_optimized(
        path,
        use_turbojpeg=config.use_turbojpeg,
        turbojpeg_fallback=config.turbojpeg_fallback,
    )


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Return an RGB copy of the provided PIL image."""
    if image.mode == "RGB":
        return image.copy()
    return image.convert("RGB")


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a numpy array without modifying the original."""
    return np.array(image)


def prenormalize_imagenet(image_array: np.ndarray) -> np.ndarray:
    """Apply ImageNet normalization to an ``HWC`` float image array."""
    if image_array.dtype != np.float32:
        image_array = image_array.astype(np.float32)
    image_array /= 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (image_array - mean) / std


__all__ = [
    "ensure_rgb",
    "load_pil_image",
    "pil_to_numpy",
    "prenormalize_imagenet",
    "safe_get_image_size",
]
