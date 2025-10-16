"""Pydantic models defining data contracts for dataset transforms."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class CacheConfig(BaseModel):
    """Configuration flags controlling dataset caching behaviour.

    Includes automatic cache versioning to prevent stale cache issues when
    configuration changes affect cached data validity.
    """

    cache_images: bool = True
    cache_maps: bool = True
    cache_transformed_tensors: bool = False
    log_statistics_every_n: int | None = Field(default=None, ge=1)

    def get_cache_version(self, load_maps: bool = False) -> str:
        """Generate cache version hash from configuration.

        The cache version ensures that cached data is invalidated when configuration
        changes affect data validity. Changes to any of these settings will result
        in a new cache version:
        - cache_transformed_tensors: Affects what gets cached
        - cache_images: Affects image caching behavior
        - cache_maps: Affects map caching behavior
        - load_maps: Critical - maps must be in cached data if load_maps=True

        Args:
            load_maps: Whether maps are being loaded (from parent DatasetConfig)

        Returns:
            8-character hex string uniquely identifying this configuration

        Example:
            >>> config = CacheConfig(cache_transformed_tensors=True, load_maps=True)
            >>> version = config.get_cache_version(load_maps=True)
            >>> print(version)  # e.g., "a3f2b8c1"
        """
        # Include all configuration that affects cached data validity
        config_str = (
            f"cache_transformed_tensors={self.cache_transformed_tensors}|"
            f"cache_images={self.cache_images}|"
            f"cache_maps={self.cache_maps}|"
            f"load_maps={load_maps}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class ImageLoadingConfig(BaseModel):
    """Configuration for image loading backends and fallbacks."""

    use_turbojpeg: bool = False
    turbojpeg_fallback: bool = False


class DatasetConfig(BaseModel):
    """All runtime configuration required to build a validated OCR dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_path: Path
    annotation_path: Path | None = None
    image_extensions: list[str] = Field(default_factory=lambda: [".jpg", ".jpeg", ".png"])
    preload_maps: bool = False
    load_maps: bool = False
    preload_images: bool = False
    prenormalize_images: bool = False
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    image_loading_config: ImageLoadingConfig = Field(default_factory=ImageLoadingConfig)

    @field_validator("image_extensions", mode="before")
    @classmethod
    def normalize_extensions(cls, value: Any) -> list[str]:
        if value is None:
            return [".jpg", ".jpeg", ".png"]

        if isinstance(value, str):
            value = [value]

        extensions: list[str] = []
        for ext in value:
            if not isinstance(ext, str) or not ext.strip():
                raise ValueError("Image extensions must be non-empty strings")
            normalized = ext.lower()
            if not normalized.startswith("."):
                normalized = f".{normalized}"
            extensions.append(normalized)
        return extensions


class ImageMetadata(BaseModel):
    """Metadata describing the context of an image being transformed."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    filename: str | None = None
    path: Path | None = None
    original_shape: tuple[int, int]
    orientation: int = Field(ge=0, le=8, default=1)
    is_normalized: bool = False
    dtype: str
    raw_size: tuple[int, int] | None = None
    polygon_frame: str | None = None
    cache_source: str | None = None
    cache_hits: int | None = Field(default=None, ge=0)
    cache_misses: int | None = Field(default=None, ge=0)

    @field_validator("original_shape")
    @classmethod
    def validate_original_shape(cls, value: tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2:
            raise ValueError("original_shape must be a tuple of (height, width)")
        height, width = value
        return (int(height), int(width))

    @field_validator("raw_size")
    @classmethod
    def validate_raw_size(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError("raw_size must be a tuple of (width, height)")
        width, height = value
        return (int(width), int(height))


class PolygonData(BaseModel):
    """Validated polygon representation with consistent shape."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    points: np.ndarray
    confidence: float | None = None
    label: str | None = None

    @field_validator("points", mode="before")
    @classmethod
    def validate_points(cls, value: Any) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            value = np.asarray(value, dtype=np.float32)

        if value.ndim == 1 and value.size % 2 == 0:
            value = value.reshape(-1, 2)
        elif value.ndim == 3 and value.shape[0] == 1:
            value = value.squeeze(0)

        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError(f"Polygon must be (N, 2) array, got shape {value.shape}")

        if value.shape[0] < 3:
            raise ValueError(f"Polygon must have at least 3 points, got {value.shape[0]}")

        return value.astype(np.float32)


class TransformInput(BaseModel):
    """Input payload for the OCR transform pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: np.ndarray
    polygons: list[PolygonData] | None = None
    metadata: ImageMetadata | None = None

    @field_validator("image", mode="before")
    @classmethod
    def validate_image(cls, value: Any) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Image must be numpy array, got {type(value)}")

        if value.ndim not in (2, 3):
            raise ValueError(f"Image must be 2D or 3D, got {value.ndim}D")

        if value.ndim == 3 and value.shape[2] not in (1, 3):
            raise ValueError(f"Image must have 1 or 3 channels, got {value.shape[2]}")

        return value


class TransformOutput(BaseModel):
    """Validated output generated by the OCR transform pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: torch.Tensor
    polygons: list[np.ndarray]
    inverse_matrix: np.ndarray
    metadata: dict[str, Any] | None = None

    @field_validator("image", mode="before")
    @classmethod
    def validate_output_image(cls, value: Any) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Output image must be torch.Tensor, got {type(value)}")

        if value.ndim != 3:
            raise ValueError(f"Output image must be 3D (C, H, W), got shape {tuple(value.shape)}")

        return value

    @field_validator("polygons", mode="before")
    @classmethod
    def validate_output_polygons(cls, value: Any) -> list[np.ndarray]:
        if value is None:
            return []

        if not isinstance(value, list):
            raise TypeError(f"Output polygons must be list, got {type(value)}")

        normalized: list[np.ndarray] = []
        for idx, polygon in enumerate(value):
            if not isinstance(polygon, np.ndarray):
                polygon = np.asarray(polygon, dtype=np.float32)

            if polygon.ndim == 2:
                polygon = polygon.reshape(1, -1, 2)

            if polygon.shape[0] != 1 or polygon.shape[2] != 2:
                raise ValueError(f"Polygon at index {idx} must have shape (1, N, 2), got {polygon.shape}")

            normalized.append(polygon.astype(np.float32))

        return normalized

    @field_validator("inverse_matrix", mode="before")
    @classmethod
    def validate_matrix(cls, value: Any) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            value = np.asarray(value, dtype=np.float32)

        if value.shape != (3, 3):
            raise ValueError(f"Inverse matrix must be (3, 3), got {value.shape}")

        return value.astype(np.float32)


class TransformConfig(BaseModel):
    """Configuration for image normalization and transform probabilities."""

    mean: tuple[float, float, float] = (0.5900, 0.5610, 0.5292)
    std: tuple[float, float, float] = (0.2136, 0.2290, 0.2511)
    always_apply: bool = False
    p: float = 1.0

    @field_validator("mean", "std")
    @classmethod
    def validate_mean_std(cls, value: tuple[float, ...]) -> tuple[float, float, float]:
        if len(value) != 3:
            raise ValueError("Mean and std must each provide 3 values for RGB channels")
        return tuple(float(v) for v in value)  # type: ignore[return-value]


class ImageData(BaseModel):
    """Cached image payload containing decoded pixel data and metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_array: np.ndarray
    raw_width: int
    raw_height: int
    orientation: int = Field(ge=0, le=8, default=1)
    is_normalized: bool = False

    @field_validator("image_array", mode="before")
    @classmethod
    def validate_image_array(cls, value: Any) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)
        if value.ndim not in (2, 3):
            raise ValueError("Cached image array must be 2D or 3D")
        return value


class MapData(BaseModel):
    """Cached probability/threshold maps aligned with an image sample."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prob_map: np.ndarray
    thresh_map: np.ndarray

    @field_validator("prob_map", "thresh_map", mode="before")
    @classmethod
    def validate_maps(cls, value: Any) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)
        if value.ndim != 3:
            raise ValueError("Maps must be rank-3 arrays shaped (C, H, W)")
        return value.astype(np.float32)

    @field_validator("thresh_map")
    @classmethod
    def ensure_shape_match(cls, thresh_map: np.ndarray, info: ValidationInfo) -> np.ndarray:
        prob_map = info.data.get("prob_map") if info.data else None
        if prob_map is not None and getattr(prob_map, "shape", None) != thresh_map.shape:
            raise ValueError("Probability and threshold maps must share identical shapes")
        return thresh_map


class DataItem(BaseModel):
    """Validated dataset sample returned by the OCR pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Any
    polygons: list[np.ndarray] = Field(default_factory=list)
    metadata: dict[str, Any] | ImageMetadata | None = None
    prob_map: np.ndarray | None = None
    thresh_map: np.ndarray | None = None
    inverse_matrix: np.ndarray | None = None

    @field_validator("image", mode="before")
    @classmethod
    def validate_tensor(cls, value: Any) -> Any:
        if isinstance(value, torch.Tensor | np.ndarray):
            return value
        raise TypeError(f"Image output must be torch.Tensor or np.ndarray, got {type(value)}")

    @field_validator("polygons", mode="before")
    @classmethod
    def validate_polygons(cls, value: Any) -> list[np.ndarray]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("Polygons must be provided as a list")
        normalized: list[np.ndarray] = []
        for poly in value:
            if not isinstance(poly, np.ndarray):
                poly = np.asarray(poly, dtype=np.float32)
            normalized.append(poly.astype(np.float32))
        return normalized

    @field_validator("inverse_matrix", mode="before")
    @classmethod
    def validate_inverse_matrix(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if not isinstance(value, np.ndarray):
            value = np.asarray(value, dtype=np.float32)
        if value.shape != (3, 3):
            raise ValueError("Inverse matrix must have shape (3, 3)")
        return value.astype(np.float32)
