"""Utilities for normalizing EXIF-oriented images and annotations.

This module centralizes rotation handling so every pipeline stage (datasets,
visualizers, inference, logging) can stay in sync.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from PIL import Image

# EXIF orientation tag constant (mirrors `ocr.datasets.base.EXIF_ORIENTATION`).
EXIF_ORIENTATION_TAG = 274

# Valid EXIF orientation values. Anything outside this set is treated as "normal".
_VALID_ORIENTATIONS = {1, 2, 3, 4, 5, 6, 7, 8}

FLIP_LEFT_RIGHT: Any
try:  # Pillow>=9
    FLIP_LEFT_RIGHT = Image.Transpose.FLIP_LEFT_RIGHT  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - Pillow<9 fallback
    FLIP_LEFT_RIGHT = Image.FLIP_LEFT_RIGHT  # type: ignore[attr-defined]


def get_exif_orientation(image: Image.Image) -> int:
    """Return EXIF orientation (defaults to 1 when missing)."""
    exif = image.getexif()
    return int(exif.get(EXIF_ORIENTATION_TAG, 1)) if exif else 1


def orientation_requires_rotation(orientation: int) -> bool:
    """True when EXIF orientation implies any rotation or mirroring."""
    return orientation in _VALID_ORIENTATIONS and orientation != 1


def normalize_pil_image(
    image: Image.Image,
    *,
    fillcolor: tuple[int, ...] | None = None,
) -> tuple[Image.Image, int]:
    """Rotate/mirror the PIL image according to its EXIF orientation.

    Returns the normalized image (copied only when necessary) and the orientation
    value that was applied. Orientation 1 returns the original image reference.
    """

    orientation = get_exif_orientation(image)
    if not orientation_requires_rotation(orientation):
        return image, orientation

    rotate_kwargs: dict[str, Any] = {}
    if fillcolor is not None:
        rotate_kwargs["fillcolor"] = fillcolor

    resampling: Any
    try:
        resampling = Image.Resampling.BICUBIC  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover
        resampling = Image.BICUBIC  # type: ignore[attr-defined]

    rotate_kwargs.setdefault("resample", resampling)

    if orientation == 2:
        normalized = image.transpose(FLIP_LEFT_RIGHT)
    elif orientation == 3:
        normalized = image.rotate(180, **rotate_kwargs)
    elif orientation == 4:
        normalized = image.rotate(180, **rotate_kwargs).transpose(FLIP_LEFT_RIGHT)
    elif orientation == 5:
        normalized = image.rotate(-90, expand=True, **rotate_kwargs).transpose(FLIP_LEFT_RIGHT)
    elif orientation == 6:
        normalized = image.rotate(-90, expand=True, **rotate_kwargs)
    elif orientation == 7:
        normalized = image.rotate(90, expand=True, **rotate_kwargs).transpose(FLIP_LEFT_RIGHT)
    elif orientation == 8:
        normalized = image.rotate(90, expand=True, **rotate_kwargs)
    else:  # pragma: no cover - orientation outside EXIF spec
        normalized = image

    return normalized, orientation


def normalize_ndarray(
    array: np.ndarray,
    orientation: int,
    *,
    fillcolor: int | Sequence[int] | None = None,
) -> np.ndarray:
    """Rotate/mirror a NumPy image array (HWC or CHW) for the given orientation."""
    if not orientation_requires_rotation(orientation):
        return array

    data = np.array(array, copy=True)
    if data.ndim == 3 and data.shape[-1] in {1, 3, 4}:  # HWC
        axes = (0, 1)
    elif data.ndim == 3:  # CHW
        axes = (1, 2)
    else:
        raise ValueError("normalize_ndarray expects 3D arrays in HWC or CHW format")

    if orientation == 2:
        data = np.flip(data, axis=axes[1])
    elif orientation == 3:
        data = np.rot90(data, 2, axes=axes)
    elif orientation == 4:
        data = np.flip(np.rot90(data, 2, axes=axes), axis=axes[1])
    elif orientation == 5:
        data = np.flip(np.rot90(data, -1, axes=axes), axis=axes[1])
    elif orientation == 6:
        data = np.rot90(data, -1, axes=axes)
    elif orientation == 7:
        data = np.flip(np.rot90(data, 1, axes=axes), axis=axes[1])
    elif orientation == 8:
        data = np.rot90(data, 1, axes=axes)

    if fillcolor is not None and data.ndim == 3:
        # Ensure that padding uses the requested fillcolor.
        expected_shape = data.shape[-1] if axes == (0, 1) else data.shape[0]
        fill = np.array(fillcolor)
        if fill.size not in {1, expected_shape}:
            raise ValueError("fillcolor must match channel count or be scalar")

    return data


def remap_polygons(
    polygons: Iterable[np.ndarray | Sequence[Sequence[float]]],
    width: float,
    height: float,
    orientation: int,
) -> list[np.ndarray]:
    """Transform polygon coordinates into the rotated orientation frame."""
    if not orientation_requires_rotation(orientation):
        return [np.array(poly, copy=True, dtype=np.float32) for poly in polygons]

    remapped: list[np.ndarray] = []
    for polygon in polygons:
        coords = np.asarray(polygon, dtype=np.float32)
        reshaped = coords.reshape(-1, 2)
        transformed = _transform_points(reshaped, width, height, orientation)
        remapped.append(transformed.reshape(coords.shape))
    return remapped


def _transform_points(
    points: np.ndarray,
    width: float,
    height: float,
    orientation: int,
) -> np.ndarray:
    """Map coordinates from original sensor frame to rotated canonical frame."""
    if orientation == 2:
        x_new = width - 1.0 - points[:, 0]
        y_new = points[:, 1]
    elif orientation == 3:
        x_new = width - 1.0 - points[:, 0]
        y_new = height - 1.0 - points[:, 1]
    elif orientation == 4:
        x_new = points[:, 0]
        y_new = height - 1.0 - points[:, 1]
    elif orientation == 5:
        x_new = points[:, 1]
        y_new = points[:, 0]
    elif orientation == 6:
        x_new = height - 1.0 - points[:, 1]
        y_new = points[:, 0]
    elif orientation == 7:
        x_new = height - 1.0 - points[:, 1]
        y_new = width - 1.0 - points[:, 0]
    elif orientation == 8:
        x_new = points[:, 1]
        y_new = width - 1.0 - points[:, 0]
    else:
        return points.copy()

    transformed = np.stack([x_new, y_new], axis=-1)
    return transformed.astype(points.dtype, copy=False)


def apply_affine_transform_to_polygons(
    polygons: Iterable[np.ndarray | Sequence[Sequence[float]] | Sequence[float]],
    matrix: np.ndarray,
) -> list[np.ndarray]:
    """Apply affine transform matrix to polygons provided as arrays or nested sequences."""
    polygons_iter = [np.asarray(poly, dtype=np.float32) for poly in polygons]
    if not polygons_iter:
        return []
    if matrix is None:
        return [poly.copy() for poly in polygons_iter]

    transformed: list[np.ndarray] = []
    for polygon in polygons_iter:
        coords = polygon
        original_shape = coords.shape

        if coords.size == 0:
            transformed.append(coords.copy())
            continue

        if coords.size % 2 != 0:
            raise ValueError("Polygon coordinates must contain an even number of values")

        coords_2d = coords.reshape(-1, 2)
        ones = np.ones((coords_2d.shape[0], 1), dtype=coords_2d.dtype)
        homogeneous = np.hstack([coords_2d, ones])

        transformed_coords = homogeneous @ matrix.T
        transformed_coords = transformed_coords[:, :2] / transformed_coords[:, 2:3]

        transformed.append(transformed_coords.reshape(original_shape))

    return transformed


def polygons_in_canonical_frame(
    polygons: Iterable[np.ndarray | Sequence[Sequence[float]] | Sequence[float]],
    width: float,
    height: float,
    orientation: int,
    *,
    tolerance: float = 1.5,
) -> bool:
    """Detect whether polygon coordinates already correspond to the rotation-corrected frame.

    When annotations are authored on images that were manually rotated before export, the
    recorded EXIF orientation can disagree with the coordinate frame. Remapping such
    polygons again would rotate them twice. This helper checks polygon extrema against
    both the raw sensor dimensions and the canonical (orientation-corrected) dimensions
    to spot that situation.
    """

    if not orientation_requires_rotation(orientation):
        return False

    arrays: list[np.ndarray] = []
    for polygon in polygons:
        coords = np.asarray(polygon, dtype=np.float32)
        if coords.size == 0:
            continue
        arrays.append(coords.reshape(-1, 2))

    if not arrays:
        return False

    stacked = np.vstack(arrays)
    min_x = float(stacked[:, 0].min(initial=float("inf")))
    max_x = float(stacked[:, 0].max(initial=float("-inf")))
    min_y = float(stacked[:, 1].min(initial=float("inf")))
    max_y = float(stacked[:, 1].max(initial=float("-inf")))

    within_raw = min_x >= -tolerance and max_x <= width - 1 + tolerance and min_y >= -tolerance and max_y <= height - 1 + tolerance
    if within_raw:
        return False

    if orientation in {5, 6, 7, 8}:
        canonical_width, canonical_height = height, width
    else:
        canonical_width, canonical_height = width, height

    within_canonical = (
        min_x >= -tolerance
        and max_x <= canonical_width - 1 + tolerance
        and min_y >= -tolerance
        and max_y <= canonical_height - 1 + tolerance
    )

    return within_canonical


__all__ = [
    "EXIF_ORIENTATION_TAG",
    "apply_affine_transform_to_polygons",
    "FLIP_LEFT_RIGHT",
    "get_exif_orientation",
    "normalize_ndarray",
    "normalize_pil_image",
    "orientation_requires_rotation",
    "polygons_in_canonical_frame",
    "remap_polygons",
]
