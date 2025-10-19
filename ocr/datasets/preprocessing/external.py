"""Optional dependency handling for preprocessing components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

ALBUMENTATIONS_AVAILABLE = False
A: Any | None = None
ImageOnlyTransform: Any = None

estimate_page_angle: Callable[[np.ndarray], float] | None = None
extract_rcrops: Callable[..., list[np.ndarray]] | None = None
doctr_remove_image_padding: Callable[[np.ndarray], np.ndarray] | None = None
doctr_rotate_image: Callable[..., np.ndarray] | None = None

DOCTR_AVAILABLE = False

# Background removal
BackgroundRemoval: Any = None
create_background_removal_transform: Callable[..., Any] | None = None
REMBG_AVAILABLE = False

try:  # pragma: no cover - optional dependency guard
    import albumentations as _albumentations
    from albumentations.core.transforms_interface import ImageOnlyTransform as _ImageOnlyTransform

    ALBUMENTATIONS_AVAILABLE = True
    A = _albumentations
    ImageOnlyTransform = _ImageOnlyTransform
except ImportError:  # pragma: no cover - optional dependency guard
    ALBUMENTATIONS_AVAILABLE = False
    A = None
    ImageOnlyTransform = None

try:  # pragma: no cover - optional dependency guard
    from doctr.utils.geometry import estimate_page_angle as _estimate_page_angle
    from doctr.utils.geometry import extract_rcrops as _extract_rcrops
    from doctr.utils.geometry import remove_image_padding as _doctr_remove_image_padding
    from doctr.utils.geometry import rotate_image as _doctr_rotate_image

    estimate_page_angle = _estimate_page_angle
    extract_rcrops = _extract_rcrops
    doctr_remove_image_padding = _doctr_remove_image_padding
    doctr_rotate_image = _doctr_rotate_image

    DOCTR_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency guard
    DOCTR_AVAILABLE = False
    estimate_page_angle = None
    extract_rcrops = None
    doctr_remove_image_padding = None
    doctr_rotate_image = None

try:  # pragma: no cover - optional dependency guard
    from .background_removal import BackgroundRemoval as _BackgroundRemoval
    from .background_removal import create_background_removal_transform as _create_background_removal_transform

    BackgroundRemoval = _BackgroundRemoval
    create_background_removal_transform = _create_background_removal_transform
    REMBG_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency guard
    BackgroundRemoval = None
    create_background_removal_transform = None
    REMBG_AVAILABLE = False

__all__ = [
    "A",
    "ALBUMENTATIONS_AVAILABLE",
    "ImageOnlyTransform",
    "DOCTR_AVAILABLE",
    "estimate_page_angle",
    "extract_rcrops",
    "doctr_remove_image_padding",
    "doctr_rotate_image",
]
