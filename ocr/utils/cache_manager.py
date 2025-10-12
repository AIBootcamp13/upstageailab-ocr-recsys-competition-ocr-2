"""Centralized cache manager used by the validated OCR dataset."""

from __future__ import annotations

import logging
from collections.abc import Callable

from ocr.datasets.schemas import CacheConfig, DataItem, ImageData, MapData


class CacheManager:
    """Manage reusable dataset assets and track cache statistics."""

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.image_cache: dict[str, ImageData] = {}
        self.tensor_cache: dict[int, DataItem] = {}
        self.maps_cache: dict[str, MapData] = {}
        self._cache_hit_count = 0
        self._cache_miss_count = 0
        self._access_counter = 0

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    def _record_access(self, hit: bool) -> None:
        if hit:
            self._cache_hit_count += 1
        else:
            self._cache_miss_count += 1
        self._access_counter += 1

        if self.config.log_statistics_every_n and self._access_counter % self.config.log_statistics_every_n == 0:
            self.log_statistics()

    def _maybe_cache(self, enabled: bool, callback: Callable[[], None]) -> None:
        if enabled:
            callback()

    # ------------------------------------------------------------------
    # Image cache
    # ------------------------------------------------------------------
    def get_cached_image(self, filename: str) -> ImageData | None:
        if not self.config.cache_images:
            return None

        cached = self.image_cache.get(filename)
        self._record_access(hit=cached is not None)
        return cached

    def set_cached_image(self, filename: str, image_data: ImageData) -> None:
        self._maybe_cache(self.config.cache_images, lambda: self.image_cache.__setitem__(filename, image_data))

    # ------------------------------------------------------------------
    # Tensor cache
    # ------------------------------------------------------------------
    def get_cached_tensor(self, idx: int) -> DataItem | None:
        if not self.config.cache_transformed_tensors:
            return None

        cached = self.tensor_cache.get(idx)
        self._record_access(hit=cached is not None)
        return cached

    def set_cached_tensor(self, idx: int, data_item: DataItem) -> None:
        self._maybe_cache(self.config.cache_transformed_tensors, lambda: self.tensor_cache.__setitem__(idx, data_item))

    # ------------------------------------------------------------------
    # Maps cache
    # ------------------------------------------------------------------
    def get_cached_maps(self, filename: str) -> MapData | None:
        if not self.config.cache_maps:
            return None

        cached = self.maps_cache.get(filename)
        self._record_access(hit=cached is not None)
        return cached

    def set_cached_maps(self, filename: str, map_data: MapData) -> None:
        self._maybe_cache(self.config.cache_maps, lambda: self.maps_cache.__setitem__(filename, map_data))

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------
    def log_statistics(self) -> None:
        total = self._cache_hit_count + self._cache_miss_count
        hit_rate = (self._cache_hit_count / total * 100.0) if total else 0.0

        self.logger.info(
            "Cache Statistics - Hits: %d, Misses: %d, Hit Rate: %.1f%%, Image Cache Size: %d, Tensor Cache Size: %d, Maps Cache Size: %d",
            self._cache_hit_count,
            self._cache_miss_count,
            hit_rate,
            len(self.image_cache),
            len(self.tensor_cache),
            len(self.maps_cache),
        )

        self.reset_statistics()

    def reset_statistics(self) -> None:
        self._cache_hit_count = 0
        self._cache_miss_count = 0

    def get_hit_count(self) -> int:
        return self._cache_hit_count

    def get_miss_count(self) -> int:
        return self._cache_miss_count
