"""Constants and mappings for EXIF orientation handling.

This module centralizes all orientation-related definitions to ensure consistency
across the codebase and make maintenance easier.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from PIL import Image


class OrientationTransform(Enum):
    """EXIF orientation transformations with their corresponding operations."""

    NORMAL = 1
    FLIP_HORIZONTAL = 2
    ROTATE_180 = 3
    FLIP_VERTICAL = 4
    TRANSPOSE = 5  # 90° clockwise + flip horizontal
    ROTATE_90_CW = 6  # 90° clockwise
    TRANSVERSE = 7  # 90° counter-clockwise + flip horizontal
    ROTATE_90_CCW = 8  # 90° counter-clockwise

    @property
    def requires_rotation(self) -> bool:
        """True if this orientation requires any transformation."""
        return self != OrientationTransform.NORMAL

    @property
    def inverse(self) -> OrientationTransform:
        """Get the inverse transformation that undoes this orientation."""
        return _ORIENTATION_INVERSE[self]

    def apply_to_image(self, image: Image.Image, **kwargs: Any) -> Image.Image:
        """Apply this orientation transformation to a PIL image."""
        if not self.requires_rotation:
            return image

        # Get resampling method
        try:
            resampling = Image.Resampling.BICUBIC  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - Pillow<9 fallback
            resampling = Image.BICUBIC  # type: ignore[attr-defined,assignment]

        kwargs.setdefault("resample", resampling)

        if self == OrientationTransform.FLIP_HORIZONTAL:
            return image.transpose(_get_flip_left_right())
        elif self == OrientationTransform.ROTATE_180:
            return image.rotate(180, **kwargs)
        elif self == OrientationTransform.FLIP_VERTICAL:
            return image.rotate(180, **kwargs).transpose(_get_flip_left_right())
        elif self == OrientationTransform.TRANSPOSE:
            return image.rotate(-90, expand=True, **kwargs).transpose(_get_flip_left_right())
        elif self == OrientationTransform.ROTATE_90_CW:
            return image.rotate(-90, expand=True, **kwargs)
        elif self == OrientationTransform.TRANSVERSE:
            return image.rotate(90, expand=True, **kwargs).transpose(_get_flip_left_right())
        elif self == OrientationTransform.ROTATE_90_CCW:
            return image.rotate(90, expand=True, **kwargs)
        else:
            return image  # pragma: no cover

    def apply_to_points(self, points: np.ndarray, width: float, height: float) -> np.ndarray:
        """Apply coordinate transformation for this orientation.

        Maps coordinates from original sensor frame to rotated canonical frame.
        """
        if not self.requires_rotation:
            return points.copy()

        x, y = points[:, 0], points[:, 1]

        if self == OrientationTransform.FLIP_HORIZONTAL:
            x_new = width - 1.0 - x
            y_new = y
        elif self == OrientationTransform.ROTATE_180:
            x_new = width - 1.0 - x
            y_new = height - 1.0 - y
        elif self == OrientationTransform.FLIP_VERTICAL:
            x_new = x
            y_new = height - 1.0 - y
        elif self == OrientationTransform.TRANSPOSE:
            x_new = y
            y_new = x
        elif self == OrientationTransform.ROTATE_90_CW:
            x_new = height - 1.0 - y
            y_new = x
        elif self == OrientationTransform.TRANSVERSE:
            x_new = height - 1.0 - y
            y_new = width - 1.0 - x
        elif self == OrientationTransform.ROTATE_90_CCW:
            x_new = y
            y_new = width - 1.0 - x
        else:
            return points.copy()  # pragma: no cover

        return np.stack([x_new, y_new], axis=-1).astype(points.dtype, copy=False)


# Orientation inverse mapping: what orientation undoes each transformation
_ORIENTATION_INVERSE = {
    OrientationTransform.NORMAL: OrientationTransform.NORMAL,
    OrientationTransform.FLIP_HORIZONTAL: OrientationTransform.FLIP_HORIZONTAL,
    OrientationTransform.ROTATE_180: OrientationTransform.ROTATE_180,
    OrientationTransform.FLIP_VERTICAL: OrientationTransform.FLIP_VERTICAL,
    OrientationTransform.TRANSPOSE: OrientationTransform.TRANSVERSE,
    OrientationTransform.ROTATE_90_CW: OrientationTransform.ROTATE_90_CCW,
    OrientationTransform.TRANSVERSE: OrientationTransform.TRANSPOSE,
    OrientationTransform.ROTATE_90_CCW: OrientationTransform.ROTATE_90_CW,
}

# Legacy integer-based mapping for backward compatibility
ORIENTATION_INVERSE_INT = {
    1: 1,  # NORMAL
    2: 2,  # FLIP_HORIZONTAL
    3: 3,  # ROTATE_180
    4: 4,  # FLIP_VERTICAL
    5: 7,  # TRANSPOSE -> TRANSVERSE
    6: 8,  # ROTATE_90_CW -> ROTATE_90_CCW
    7: 5,  # TRANSVERSE -> TRANSPOSE
    8: 6,  # ROTATE_90_CCW -> ROTATE_90_CW
}

# Valid EXIF orientation values
VALID_ORIENTATIONS = frozenset({1, 2, 3, 4, 5, 6, 7, 8})

# EXIF orientation tag constant
EXIF_ORIENTATION_TAG = 274


def _get_flip_left_right() -> Any:
    """Get the FLIP_LEFT_RIGHT constant, handling Pillow version differences."""
    try:  # Pillow>=9
        return Image.Transpose.FLIP_LEFT_RIGHT  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - Pillow<9 fallback
        return Image.FLIP_LEFT_RIGHT  # type: ignore[attr-defined]


def get_orientation_transform(orientation: int) -> OrientationTransform:
    """Convert integer orientation value to OrientationTransform enum."""
    try:
        return OrientationTransform(orientation)
    except ValueError:
        return OrientationTransform.NORMAL


def get_inverse_orientation(orientation: int) -> int:
    """Get the inverse orientation value for the given orientation."""
    return ORIENTATION_INVERSE_INT.get(orientation, 1)
