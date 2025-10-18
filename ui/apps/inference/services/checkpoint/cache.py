"""Caching layer for checkpoint catalog.

This module provides LRU caching for catalog builds to avoid re-processing
checkpoints on every UI interaction.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import CheckpointCatalog

LOGGER = logging.getLogger(__name__)


class CatalogCache:
    """LRU cache for checkpoint catalog builds.

    Cache invalidation is based on the outputs directory modification time,
    ensuring stale catalogs are rebuilt when new checkpoints are added.
    """

    def __init__(self, maxsize: int = 128):
        """Initialize cache with maximum size.

        Args:
            maxsize: Maximum number of catalogs to cache
        """
        self.maxsize = maxsize
        self._cache: dict[str, CheckpointCatalog] = {}

    def get_cache_key(self, outputs_dir: Path) -> str:
        """Generate cache key for outputs directory.

        Cache key includes:
        - Absolute path hash
        - Directory modification time (for invalidation)

        Args:
            outputs_dir: Outputs directory

        Returns:
            Cache key string
        """
        try:
            mtime = outputs_dir.stat().st_mtime
        except OSError:
            # Directory doesn't exist or is inaccessible
            mtime = 0

        path_hash = hashlib.md5(str(outputs_dir.absolute()).encode()).hexdigest()[:8]
        return f"{path_hash}:{mtime:.0f}"

    def get(self, outputs_dir: Path) -> CheckpointCatalog | None:
        """Get cached catalog for outputs directory.

        Args:
            outputs_dir: Outputs directory

        Returns:
            Cached catalog, or None if not in cache or stale
        """
        cache_key = self.get_cache_key(outputs_dir)
        catalog = self._cache.get(cache_key)

        if catalog is not None:
            LOGGER.debug("Cache hit for %s", outputs_dir)
        else:
            LOGGER.debug("Cache miss for %s", outputs_dir)

        return catalog

    def set(self, outputs_dir: Path, catalog: CheckpointCatalog) -> None:
        """Cache catalog for outputs directory.

        Args:
            outputs_dir: Outputs directory
            catalog: Catalog to cache
        """
        cache_key = self.get_cache_key(outputs_dir)

        # Simple LRU: remove oldest if at capacity
        if len(self._cache) >= self.maxsize:
            # Remove first (oldest) entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            LOGGER.debug("Cache eviction: %s", oldest_key)

        self._cache[cache_key] = catalog
        LOGGER.debug("Cached catalog for %s (%d entries)", outputs_dir, catalog.total_count)

    def clear(self) -> None:
        """Clear all cached catalogs."""
        self._cache.clear()
        LOGGER.info("Cache cleared")


# Global singleton cache instance
_global_cache = CatalogCache()


def get_global_cache() -> CatalogCache:
    """Get global catalog cache instance.

    Returns:
        Global cache instance
    """
    return _global_cache


def clear_global_cache() -> None:
    """Clear global catalog cache."""
    _global_cache.clear()
