"""
Polygon processing cache for validation speedup.

Caches expensive PyClipper polygon operations to reduce validation time
from 10x training to <2x training.
"""

import hashlib
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np


class PolygonCache:
    """LRU cache for polygon processing results.

    Caches the output of make_prob_thresh_map to avoid repeated
    expensive PyClipper operations during validation.

    Args:
        max_size: Maximum number of entries in cache
        persist_to_disk: If True, save/load cache from disk
        cache_dir: Directory for disk persistence (if enabled)
    """

    def __init__(
        self,
        max_size: int = 1000,
        persist_to_disk: bool = False,
        cache_dir: Path | str | None = None,
    ):
        # Handle falsy max_size values (like False from config)
        if not max_size or max_size <= 0:
            max_size = 1000  # Default to reasonable size

        self.max_size = max_size
        self.persist_to_disk = persist_to_disk
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/polygon_cache")

        # LRU cache implementation
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

        # Metrics
        self.hit_count = 0
        self.miss_count = 0

        # Load from disk if persistence enabled
        if self.persist_to_disk:
            self._load_from_disk()

    def _generate_key(
        self,
        polygons: np.ndarray,
        image_shape: tuple[int, int, int],
        params: tuple[float, float, float],
    ) -> str:
        """
        Generate deterministic cache key from polygon geometry.

        Args:
            polygons: Polygon coordinates (N, 4, 2)
            image_shape: (C, H, W) of image
            params: (shrink_ratio, thresh_min, thresh_max)

        Returns:
            Hexadecimal hash string
        """
        # Normalize polygons to ensure consistent hashing
        # Flatten polygons and sort the elements to normalize the order
        poly_flattened = polygons.flatten()
        poly_normalized = np.sort(poly_flattened)

        # Create composite key from all inputs
        key_data = [
            poly_normalized.tobytes(),
            str(image_shape).encode(),
            str(params).encode(),
        ]

        # Use fast hash (blake2b)
        hasher = hashlib.blake2b()
        for data in key_data:
            hasher.update(data)

        return hasher.hexdigest()

    def get(self, key: str) -> dict[str, np.ndarray] | None:
        """
        Retrieve cached result.

        Args:
            key: Cache key from _generate_key

        Returns:
            Cached result dict or None if not found
        """
        if key in self._cache:
            # Move to end (LRU)
            self._cache.move_to_end(key)
            self.hit_count += 1
            return self._cache[key]

        self.miss_count += 1
        return None

    def set(self, key: str, value: dict[str, np.ndarray]) -> None:
        """
        Store result in cache.

        Args:
            key: Cache key from _generate_key
            value: Result dict with 'prob_map' and 'thresh_map'
        """
        # Add to cache
        self._cache[key] = value
        self._cache.move_to_end(key)

        # Evict oldest if over size limit
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

        # Persist if enabled
        if self.persist_to_disk:
            self._save_to_disk()

    def invalidate(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self.hit_count = 0
        self.miss_count = 0

        if self.persist_to_disk:
            cache_file = self.cache_dir / "polygon_cache.pkl"
            if cache_file.exists():
                cache_file.unlink()

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return self.hit_count / total

    def get_stats(self) -> dict[str, int | float]:
        """
        Get cache statistics.

        Returns:
            Dict with hit_count, miss_count, total_requests, hit_rate, cache_size
        """
        total = self.hit_count + self.miss_count
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "total_requests": total,
            "hit_rate": self.hit_count / total if total > 0 else 0.0,
            "cache_size": len(self._cache),
        }

    def _save_to_disk(self) -> None:
        """Save cache to disk (internal method)."""
        if not self.persist_to_disk:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / "polygon_cache.pkl"

        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "cache": self._cache,
                    "hit_count": self.hit_count,
                    "miss_count": self.miss_count,
                },
                f,
            )

    def _load_from_disk(self) -> None:
        """Load cache from disk (internal method)."""
        if not self.persist_to_disk:
            return

        cache_file = self.cache_dir / "polygon_cache.pkl"
        if not cache_file.exists():
            return

        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                self._cache = data["cache"]
                self.hit_count = data["hit_count"]
                self.miss_count = data["miss_count"]
        except Exception:
            # Ignore corrupted cache, start fresh
            pass
