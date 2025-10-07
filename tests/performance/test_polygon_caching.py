"""
Test-Driven Development for Polygon Caching Optimization
Phase 1.1: Cache PyClipper Polygon Processing

Tests validate the polygon caching functionality to ensure:
- Correct caching of polygon processing results
- Performance improvements without accuracy loss
- Memory usage within acceptable limits
"""

import numpy as np
import pytest
import torch

from ocr.datasets.db_collate_fn import DBCollateFN
from ocr.datasets.polygon_cache import PolygonCache


class TestPolygonCache:
    """Test suite for polygon caching functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache = PolygonCache(max_size=100, persist_to_disk=False)
        self.collate_fn = DBCollateFN()

        # Sample polygon data
        self.sample_polygons = [
            np.array([[[10, 10], [50, 10], [50, 50], [10, 50]]]),
        ]
        self.sample_image = torch.randn(3, 64, 64)

    def test_cache_initialization(self):
        """Test that cache initializes correctly."""
        assert self.cache.max_size == 100
        assert len(self.cache) == 0
        assert self.cache.hit_count == 0
        assert self.cache.miss_count == 0

    def test_polygon_processing_caching(self):
        """Test that polygon processing results are cached correctly."""
        # Generate cache key
        key = self.cache._generate_key(
            self.sample_polygons[0],
            (self.sample_image.shape[0], self.sample_image.shape[1], self.sample_image.shape[2]),
            (0.4, 0.3, 0.7),
        )

        # Process polygons
        result = self.collate_fn.make_prob_thresh_map(self.sample_image, self.sample_polygons, "test.jpg")

        # Cache the result
        self.cache.set(key, result)

        # Retrieve from cache
        cached_result = self.cache.get(key)

        assert cached_result is not None
        np.testing.assert_array_equal(cached_result["prob_map"], result["prob_map"])
        np.testing.assert_array_equal(cached_result["thresh_map"], result["thresh_map"])

    def test_cache_hit_miss_tracking(self):
        """Test cache hit/miss ratio tracking."""
        key = self.cache._generate_key(
            self.sample_polygons[0],
            (self.sample_image.shape[0], self.sample_image.shape[1], self.sample_image.shape[2]),
            (0.4, 0.3, 0.7),
        )

        # First access - miss
        result = self.cache.get(key)
        assert result is None
        assert self.cache.miss_count == 1

        # Set and get - hit
        self.cache.set(key, {"prob_map": np.array([1, 2, 3]), "thresh_map": np.array([4, 5, 6])})
        result = self.cache.get(key)
        assert result is not None
        assert self.cache.hit_count == 1

    def test_cache_size_limits(self):
        """Test that cache respects size limits."""
        # Fill cache beyond max_size
        for i in range(110):
            polygons = np.array([[[i, i], [i + 10, i], [i + 10, i + 10], [i, i + 10]]])
            key = self.cache._generate_key(polygons, (3, 64, 64), (0.4, 0.3, 0.7))
            self.cache.set(key, {"prob_map": np.array([i]), "thresh_map": np.array([i + 1])})

        # Cache should not exceed max_size
        assert len(self.cache) <= self.cache.max_size

    @pytest.mark.performance
    def test_performance_improvement(self):
        """Test that caching improves performance."""
        import time

        key = self.cache._generate_key(
            self.sample_polygons[0],
            (self.sample_image.shape[0], self.sample_image.shape[1], self.sample_image.shape[2]),
            (0.4, 0.3, 0.7),
        )

        # Time uncached operation
        start = time.time()
        result = self.collate_fn.make_prob_thresh_map(self.sample_image, self.sample_polygons, "test.jpg")
        uncached_time = time.time() - start

        # Cache the result
        self.cache.set(key, result)

        # Time cached operation
        start = time.time()
        self.cache.get(key)
        cached_time = time.time() - start

        # Cached access should be at least 10x faster
        assert cached_time < uncached_time * 0.1

    def test_cache_invalidation(self):
        """Test cache invalidation functionality."""
        key = self.cache._generate_key(
            self.sample_polygons[0],
            (self.sample_image.shape[0], self.sample_image.shape[1], self.sample_image.shape[2]),
            (0.4, 0.3, 0.7),
        )
        self.cache.set(key, {"prob_map": np.array([1, 2, 3]), "thresh_map": np.array([4, 5, 6])})

        assert self.cache.get(key) is not None

        self.cache.invalidate()

        assert self.cache.get(key) is None
        assert len(self.cache) == 0

    def test_cache_persistence(self):
        """Test cache persistence to disk."""
        import tempfile
        from pathlib import Path

        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = PolygonCache(max_size=10, persist_to_disk=True, cache_dir=tmp_dir)

            # Add an item to the cache
            key = cache._generate_key(
                self.sample_polygons[0],
                (self.sample_image.shape[0], self.sample_image.shape[1], self.sample_image.shape[2]),
                (0.4, 0.3, 0.7),
            )
            cache.set(key, {"prob_map": np.array([1, 2, 3]), "thresh_map": np.array([4, 5, 6])})

            # Check that cache file was created
            cache_file = Path(tmp_dir) / "polygon_cache.pkl"
            assert cache_file.exists()

            # Create a new cache instance and load from disk
            new_cache = PolygonCache(max_size=10, persist_to_disk=True, cache_dir=tmp_dir)

            # The cache should have the same content
            result = new_cache.get(key)
            assert result is not None
            np.testing.assert_array_equal(result["prob_map"], np.array([1, 2, 3]))
            np.testing.assert_array_equal(result["thresh_map"], np.array([4, 5, 6]))


class TestDBCollateFNWithCaching:
    """Test DBCollateFN integration with caching."""

    def test_collate_with_cache_integration(self):
        """Test that collate function integrates with cache."""
        from ocr.datasets.db_collate_fn import DBCollateFN

        collate_fn = DBCollateFN()
        cache = PolygonCache(max_size=100, persist_to_disk=False)

        # Sample data
        sample_polygons = [np.array([[[10, 10], [50, 10], [50, 50], [10, 50]]])]
        sample_image = torch.randn(3, 64, 64)

        # Generate cache key
        key = cache._generate_key(
            sample_polygons[0],
            (sample_image.shape[0], sample_image.shape[1], sample_image.shape[2]),
            (0.4, 0.3, 0.7),
        )

        # Process and cache the result
        result = collate_fn.make_prob_thresh_map(sample_image, sample_polygons, "test.jpg")
        cache.set(key, result)

        # Verify we can retrieve from cache
        cached_result = cache.get(key)
        assert cached_result is not None
        assert "prob_map" in cached_result
        assert "thresh_map" in cached_result
