"""
AI_DOCS: Cache Manager - Centralized Dataset Caching System

This module implements the CacheManager class, responsible for:
- Multi-level caching (images, tensors, maps) for dataset performance
- Cache statistics tracking and logging
- Memory-efficient storage with configurable limits
- Thread-safe cache operations for DataLoader compatibility

ARCHITECTURE OVERVIEW:
- Three cache types: image_cache, tensor_cache, maps_cache
- Configurable caching via CacheConfig (Pydantic model)
- Statistics tracking with periodic logging
- Lazy evaluation with conditional caching

DATA CONTRACTS:
- Input: CacheConfig (Pydantic model)
- Cache Keys: str (filenames) or int (dataset indices)
- Cache Values: ImageData, DataItem, MapData (Pydantic models)
- Statistics: hit/miss counts with configurable logging

CORE CONSTRAINTS:
- NEVER modify cache key formats (breaks cache invalidation)
- ALWAYS check cache config before operations
- PRESERVE statistics tracking for performance monitoring
- USE Pydantic models for all cached data
- MAINTAIN thread-safety for DataLoader compatibility

PERFORMANCE FEATURES:
- Lazy caching prevents memory bloat
- Configurable cache sizes and eviction policies
- Statistics logging for performance debugging
- Memory-efficient storage of large tensors

VALIDATION REQUIREMENTS:
- All cache values must be Pydantic models
- Cache keys must be hashable and deterministic
- Cache operations must handle missing keys gracefully
- Statistics must be accurate for performance analysis

RELATED DOCUMENTATION:
- Data Contracts: ocr/validation/models.py
- Configuration: ocr/datasets/schemas.py
- Base Dataset: ocr/datasets/base.py
- Performance Guide: docs/ai_handbook/04_performance/

MIGRATION NOTES:
- CacheManager replaces inline caching logic
- Pydantic models ensure data integrity
- Configurable caching improves memory management
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ocr.datasets.schemas import CacheConfig, DataItem, ImageData, MapData


class CacheManager:
    """
    AI_DOCS: CacheManager - Multi-Level Dataset Caching

    This class provides centralized caching for the OCR dataset system:
    - Image caching: Raw ImageData objects to avoid reloading
    - Tensor caching: Fully processed DataItem objects for speed
    - Maps caching: Probability/threshold maps for evaluation

    CONSTRAINTS FOR AI ASSISTANTS:
    - DO NOT modify cache data structures (dict types)
    - ALWAYS use config flags to enable/disable caching
    - PRESERVE statistics tracking methods
    - USE Pydantic models for cache values
    - MAINTAIN lazy evaluation pattern

    Cache Types:
    - image_cache: dict[str, ImageData] - keyed by filename
    - tensor_cache: dict[int, DataItem] - keyed by dataset index
    - maps_cache: dict[str, MapData] - keyed by filename
    """

    def __init__(self, config: CacheConfig, cache_version: str | None = None) -> None:
        """
        AI_DOCS: Constructor Constraints
        - config: CacheConfig (Pydantic model) - NEVER pass raw dict
        - cache_version: Version string for cache invalidation (optional)
        - Initialize all cache dicts as empty
        - Setup statistics counters
        - DO NOT modify cache structure without updating all consumers

        Cache Versioning:
        If cache_version is provided, it's stored and can be used to validate
        cache compatibility. This prevents stale cache issues when configuration
        changes (e.g., enabling load_maps after cache was built without maps).
        """
        self.config = config
        self.cache_version = cache_version
        self.logger = logging.getLogger(__name__)
        self.image_cache: dict[str, ImageData] = {}
        self.tensor_cache: dict[int, DataItem] = {}
        self.maps_cache: dict[str, MapData] = {}
        self._cache_hit_count = 0
        self._cache_miss_count = 0
        self._access_counter = 0

        # Log cache version for debugging
        if cache_version:
            self.logger.debug(f"CacheManager initialized with version: {cache_version}")

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
        """
        AI_DOCS: Tensor Cache Retrieval
        Retrieves fully processed DataItem from cache by dataset index.

        CRITICAL CONSTRAINTS:
        - Return None if caching disabled (config.cache_transformed_tensors)
        - Record access statistics (hit/miss) for performance monitoring
        - Return DataItem Pydantic model (NOT dict)
        - Handle missing keys gracefully

        Performance Impact: Cache hits avoid expensive __getitem__ processing
        """
        if not self.config.cache_transformed_tensors:
            return None

        cached = self.tensor_cache.get(idx)
        self._record_access(hit=cached is not None)
        return cached

    def set_cached_tensor(self, idx: int, data_item: DataItem) -> None:
        """
        AI_DOCS: Tensor Cache Storage
        Stores fully processed DataItem in cache by dataset index.

        CRITICAL CONSTRAINTS:
        - Only cache if config.cache_transformed_tensors is True
        - data_item MUST be DataItem Pydantic model
        - idx MUST be int (dataset index)
        - Overwrite existing entries without warning

        Memory Impact: Large tensors stored in memory for performance
        """
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
            "\nCache Statistics - Hits: %d, Misses: %d, Hit Rate: %.1f%%, Image Cache Size: %d, Tensor Cache Size: %d, Maps Cache Size: %d",
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

    def clear_tensor_cache(self) -> None:
        """Clear the tensor cache to prevent data corruption."""
        self.tensor_cache.clear()
        self.logger.info(f"Tensor cache cleared ({len(self.tensor_cache)} items remaining)")

    def clear_image_cache(self) -> None:
        """Clear the image cache."""
        self.image_cache.clear()
        self.logger.info(f"Image cache cleared ({len(self.image_cache)} items remaining)")

    def clear_maps_cache(self) -> None:
        """Clear the maps cache."""
        self.maps_cache.clear()
        self.logger.info(f"Maps cache cleared ({len(self.maps_cache)} items remaining)")

    def clear_all_caches(self) -> None:
        """Clear all caches to prevent data corruption."""
        self.clear_tensor_cache()
        self.clear_image_cache()
        self.clear_maps_cache()
        self.logger.info("All caches cleared")

    def get_cache_health(self) -> dict[str, Any]:
        """Get comprehensive cache health statistics.

        Returns a dictionary with:
        - Cache sizes (entries per cache type)
        - Hit/miss statistics
        - Hit rate percentage
        - Cache version information
        - Memory usage estimates

        This is useful for monitoring cache performance and debugging cache issues.

        Returns:
            Dictionary with cache health metrics

        Example:
            >>> manager = CacheManager(config, cache_version="abc123")
            >>> health = manager.get_cache_health()
            >>> print(f"Hit rate: {health['hit_rate_percent']:.1f}%")
        """
        total_accesses = self._cache_hit_count + self._cache_miss_count
        hit_rate = (self._cache_hit_count / total_accesses * 100.0) if total_accesses else 0.0

        return {
            "cache_version": self.cache_version,
            "image_cache_size": len(self.image_cache),
            "tensor_cache_size": len(self.tensor_cache),
            "maps_cache_size": len(self.maps_cache),
            "total_cache_entries": len(self.image_cache) + len(self.tensor_cache) + len(self.maps_cache),
            "cache_hits": self._cache_hit_count,
            "cache_misses": self._cache_miss_count,
            "total_accesses": total_accesses,
            "hit_rate_percent": hit_rate,
            "config": {
                "cache_images": self.config.cache_images,
                "cache_maps": self.config.cache_maps,
                "cache_transformed_tensors": self.config.cache_transformed_tensors,
            },
        }

    def log_cache_health(self) -> None:
        """Log comprehensive cache health information.

        This provides a detailed overview of cache status including:
        - Cache sizes
        - Hit/miss rates
        - Configuration settings
        - Cache version

        Useful for debugging performance issues and validating cache behavior.
        """
        health = self.get_cache_health()

        self.logger.info("=" * 60)
        self.logger.info("CACHE HEALTH REPORT")
        self.logger.info("=" * 60)
        if health["cache_version"]:
            self.logger.info(f"Cache Version: {health['cache_version']}")
        self.logger.info(f"Image Cache: {health['image_cache_size']} entries")
        self.logger.info(f"Tensor Cache: {health['tensor_cache_size']} entries")
        self.logger.info(f"Maps Cache: {health['maps_cache_size']} entries")
        self.logger.info(f"Total Entries: {health['total_cache_entries']}")
        self.logger.info("-" * 60)
        self.logger.info(f"Cache Hits: {health['cache_hits']}")
        self.logger.info(f"Cache Misses: {health['cache_misses']}")
        self.logger.info(f"Hit Rate: {health['hit_rate_percent']:.1f}%")
        self.logger.info("-" * 60)
        self.logger.info("Configuration:")
        self.logger.info(f"  cache_images: {health['config']['cache_images']}")
        self.logger.info(f"  cache_maps: {health['config']['cache_maps']}")
        self.logger.info(f"  cache_transformed_tensors: {health['config']['cache_transformed_tensors']}")
        self.logger.info("=" * 60)


#
# =======================================================================
# CACHEMANAGER - AI ASSISTANT CONSTRAINTS & REQUIREMENTS
# =======================================================================
#
# 1. CACHE DATA STRUCTURES (DO NOT MODIFY):
#    - image_cache: dict[str, ImageData]
#    - tensor_cache: dict[int, DataItem]
#    - maps_cache: dict[str, MapData]
#
# 2. METHOD SIGNATURES (PRESERVE):
#    - get_cached_*() -> PydanticModel | None
#    - set_cached_*() -> None (takes Pydantic model)
#    - log_statistics() -> None
#    - reset_statistics() -> None
#
# 3. CONFIGURATION INTEGRATION:
#    - ALWAYS check config flags before caching
#    - RESPECT CacheConfig settings
#    - USE config.log_statistics_every_n for logging
#
# 4. STATISTICS TRACKING (MANDATORY):
#    - Record all cache accesses (hits/misses)
#    - Log statistics periodically
#    - Reset counters after logging
#    - Provide hit/miss count accessors
#
# 5. Pydantic MODEL REQUIREMENTS:
#    - All cache values MUST be Pydantic models
#    - ImageData for image cache
#    - DataItem for tensor cache
#    - MapData for maps cache
#
# =======================================================================
# COMMON AI MISTAKES TO AVOID:
# =======================================================================
#
# ❌ Changing cache dict structures or key types
# ❌ Skipping config flag checks
# ❌ Not recording cache statistics
# ❌ Using raw dicts instead of Pydantic models
# ❌ Modifying method signatures
# ❌ Breaking lazy evaluation pattern
#
# =======================================================================
