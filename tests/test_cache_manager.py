"""Comprehensive pytest test suite for the CacheManager class."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from ocr.datasets.schemas import CacheConfig, DataItem, ImageData, ImageMetadata, MapData
from ocr.utils.cache_manager import CacheManager


@pytest.fixture
def cache_config():
    """Fixture providing a basic cache configuration."""
    return CacheConfig(cache_transformed_tensors=True)


@pytest.fixture
def cache_manager(cache_config):
    """Fixture providing a CacheManager instance."""
    return CacheManager(config=cache_config)


@pytest.fixture
def sample_image_data():
    """Fixture providing sample image data."""
    return ImageData(image_array=np.random.rand(100, 100, 3), raw_width=100, raw_height=100, orientation=1, is_normalized=False)


@pytest.fixture
def sample_data_item():
    """Fixture providing sample data item."""
    image = np.random.rand(3, 100, 100).astype(np.float32)
    polygons = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)]
    metadata = ImageMetadata(
        filename="test.jpg",
        path=None,
        original_shape=(100, 100),
        orientation=1,
        is_normalized=False,
        dtype=str(image.dtype),
        raw_size=(100, 100),
        polygon_frame="raw",
    )
    return DataItem(
        image=image,
        polygons=polygons,
        metadata=metadata.model_dump(),
        prob_map=None,
        thresh_map=None,
        inverse_matrix=np.eye(3, dtype=np.float32),
    )


@pytest.fixture
def sample_map_data():
    """Fixture providing sample map data."""
    prob_map = np.random.rand(1, 100, 100).astype(np.float32)
    thresh_map = np.random.rand(1, 100, 100).astype(np.float32)
    return MapData(prob_map=prob_map, thresh_map=thresh_map)


class TestCacheManagerInitialization:
    """Test CacheManager initialization and basic attributes."""

    def test_initialization_with_config(self, cache_config):
        """Test that CacheManager initializes with provided config."""
        cache_manager = CacheManager(config=cache_config)

        assert cache_manager.config == cache_config
        assert isinstance(cache_manager.logger, logging.Logger)
        assert hasattr(cache_manager, "image_cache")
        assert hasattr(cache_manager, "tensor_cache")
        assert hasattr(cache_manager, "maps_cache")
        assert cache_manager._cache_hit_count == 0
        assert cache_manager._cache_miss_count == 0

    def test_initial_empty_caches(self, cache_manager):
        """Test that all caches are empty after initialization."""
        assert len(cache_manager.image_cache) == 0
        assert len(cache_manager.tensor_cache) == 0
        assert len(cache_manager.maps_cache) == 0


class TestImageCacheOperations:
    """Test image cache operations."""

    def test_get_cached_image_not_found(self, cache_manager):
        """Test retrieving a non-existent image from cache returns None."""
        result = cache_manager.get_cached_image("nonexistent.jpg")

        assert result is None
        assert cache_manager._cache_miss_count == 1

    def test_set_and_get_cached_image(self, cache_manager, sample_image_data):
        """Test setting and retrieving an image from cache."""
        filename = "test.jpg"

        # Set the image in cache
        cache_manager.set_cached_image(filename, sample_image_data)

        # Get the image from cache
        result = cache_manager.get_cached_image(filename)

        assert result == sample_image_data
        assert cache_manager._cache_hit_count == 1
        assert filename in cache_manager.image_cache

    def test_image_cache_statistics(self, cache_manager, sample_image_data):
        """Test cache hit/miss statistics for image operations."""
        filename = "test.jpg"

        # Miss: image not in cache
        result = cache_manager.get_cached_image(filename)
        assert result is None
        assert cache_manager._cache_miss_count == 1

        # Set the image
        cache_manager.set_cached_image(filename, sample_image_data)

        # Hit: image in cache
        result = cache_manager.get_cached_image(filename)
        assert result == sample_image_data
        assert cache_manager._cache_hit_count == 1


class TestTensorCacheOperations:
    """Test tensor cache operations."""

    def test_get_cached_tensor_not_found(self, cache_manager):
        """Test retrieving a non-existent tensor from cache returns None."""
        result = cache_manager.get_cached_tensor(0)

        assert result is None
        assert cache_manager._cache_miss_count == 1

    def test_set_and_get_cached_tensor(self, cache_manager, sample_data_item):
        """Test setting and retrieving a tensor from cache."""
        idx = 0

        # Set the tensor in cache
        cache_manager.set_cached_tensor(idx, sample_data_item)

        # Get the tensor from cache
        result = cache_manager.get_cached_tensor(idx)

        assert result == sample_data_item
        assert cache_manager._cache_hit_count == 1
        assert idx in cache_manager.tensor_cache

    def test_tensor_cache_statistics(self, cache_manager, sample_data_item):
        """Test cache hit/miss statistics for tensor operations."""
        idx = 0

        # Miss: tensor not in cache
        result = cache_manager.get_cached_tensor(idx)
        assert result is None
        assert cache_manager._cache_miss_count == 1

        # Set the tensor
        cache_manager.set_cached_tensor(idx, sample_data_item)

        # Hit: tensor in cache
        result = cache_manager.get_cached_tensor(idx)
        assert result == sample_data_item
        assert cache_manager._cache_hit_count == 1


class TestMapsCacheOperations:
    """Test maps cache operations."""

    def test_get_cached_maps_not_found(self, cache_manager):
        """Test retrieving non-existent maps from cache returns None."""
        result = cache_manager.get_cached_maps("nonexistent.jpg")

        assert result is None
        assert cache_manager._cache_miss_count == 1

    def test_set_and_get_cached_maps(self, cache_manager, sample_map_data):
        """Test setting and retrieving maps from cache."""
        filename = "test.jpg"

        # Set the maps in cache
        cache_manager.set_cached_maps(filename, sample_map_data)

        # Get the maps from cache
        result = cache_manager.get_cached_maps(filename)

        assert result == sample_map_data
        assert cache_manager._cache_hit_count == 1
        assert filename in cache_manager.maps_cache

    def test_maps_cache_statistics(self, cache_manager, sample_map_data):
        """Test cache hit/miss statistics for maps operations."""
        filename = "test.jpg"

        # Miss: maps not in cache
        result = cache_manager.get_cached_maps(filename)
        assert result is None
        assert cache_manager._cache_miss_count == 1

        # Set the maps
        cache_manager.set_cached_maps(filename, sample_map_data)

        # Hit: maps in cache
        result = cache_manager.get_cached_maps(filename)
        assert result == sample_map_data
        assert cache_manager._cache_hit_count == 1


class TestCacheStatistics:
    """Test cache statistics functionality."""

    def test_log_statistics(self, cache_manager, sample_image_data, caplog):
        """Test that log_statistics outputs correct cache statistics."""
        # Perform some cache operations to generate statistics
        cache_manager.get_cached_image("nonexistent.jpg")  # Miss
        cache_manager.set_cached_image("test.jpg", sample_image_data)
        cache_manager.get_cached_image("test.jpg")  # Hit

        with caplog.at_level(logging.INFO):
            cache_manager.log_statistics()

        # Check that statistics were logged
        assert len(caplog.records) == 1
        assert "Cache Statistics" in caplog.text
        assert "Hits: 1" in caplog.text
        assert "Misses: 1" in caplog.text
        assert "Hit Rate: 50.0%" in caplog.text
        assert "Image Cache Size: 1" in caplog.text

        # Check that counters were reset
        assert cache_manager._cache_hit_count == 0
        assert cache_manager._cache_miss_count == 0

    def test_log_statistics_with_empty_cache(self, cache_manager, caplog):
        """Test statistics logging with empty cache."""
        with caplog.at_level(logging.INFO):
            cache_manager.log_statistics()

        assert len(caplog.records) == 1
        assert "Cache Statistics" in caplog.text
        assert "Hits: 0" in caplog.text
        assert "Misses: 0" in caplog.text
        assert "Hit Rate: 0.0%" in caplog.text

    def test_log_statistics_with_zero_accesses(self, cache_manager, caplog):
        """Test statistics logging when there are no accesses."""
        with caplog.at_level(logging.INFO):
            cache_manager.log_statistics()

        assert "Hit Rate: 0.0%" in caplog.text

    def test_get_hit_count(self, cache_manager, sample_image_data):
        """Test getting cache hit count."""
        # Initially should be 0
        assert cache_manager.get_hit_count() == 0

        # Add some hits
        cache_manager.set_cached_image("test.jpg", sample_image_data)
        cache_manager.get_cached_image("test.jpg")

        assert cache_manager.get_hit_count() == 1

    def test_get_miss_count(self, cache_manager):
        """Test getting cache miss count."""
        # Initially should be 0
        assert cache_manager.get_miss_count() == 0

        # Add some misses
        cache_manager.get_cached_image("nonexistent.jpg")

        assert cache_manager.get_miss_count() == 1


class TestCacheIsolation:
    """Test that different cache types are properly isolated."""

    def test_cache_isolation(self, cache_manager, sample_image_data, sample_data_item, sample_map_data):
        """Test that image, tensor, and maps caches don't interfere with each other."""
        # Set items in different caches
        cache_manager.set_cached_image("image.jpg", sample_image_data)
        cache_manager.set_cached_tensor(0, sample_data_item)
        cache_manager.set_cached_maps("maps.jpg", sample_map_data)

        # Verify each cache has its respective item
        assert "image.jpg" in cache_manager.image_cache
        assert 0 in cache_manager.tensor_cache
        assert "maps.jpg" in cache_manager.maps_cache

        # Verify other caches don't have the items
        assert "image.jpg" not in cache_manager.tensor_cache
        assert 0 not in cache_manager.maps_cache
        assert "maps.jpg" not in cache_manager.image_cache

        # Verify correct items are retrieved
        assert cache_manager.get_cached_image("image.jpg") == sample_image_data
        assert cache_manager.get_cached_tensor(0) == sample_data_item
        assert cache_manager.get_cached_maps("maps.jpg") == sample_map_data


class TestCacheManagerEdgeCases:
    """Test edge cases and error conditions."""

    def test_cache_with_none_values(self, cache_manager):
        """Test cache behavior with None values."""
        # Should not crash when trying to get non-existent items
        assert cache_manager.get_cached_image("nonexistent.jpg") is None
        assert cache_manager.get_cached_tensor(999) is None
        assert cache_manager.get_cached_maps("nonexistent.jpg") is None

    def test_cache_with_special_filenames(self, cache_manager, sample_image_data):
        """Test cache with special filenames."""
        special_filenames = [
            "file with spaces.jpg",
            "file_with_underscores.png",
            "file.with.dots.jpeg",
            "file123.jpg",
            "FILE.JPG",
            "file.jpg.backup",
        ]

        for i, filename in enumerate(special_filenames):
            cache_manager.set_cached_image(filename, sample_image_data)
            retrieved = cache_manager.get_cached_image(filename)
            assert retrieved == sample_image_data

    def test_cache_with_large_indices(self, cache_manager, sample_data_item):
        """Test tensor cache with large indices."""
        large_indices = [1000, 10000, 100000, 999999]

        for idx in large_indices:
            cache_manager.set_cached_tensor(idx, sample_data_item)
            retrieved = cache_manager.get_cached_tensor(idx)
            assert retrieved == sample_data_item
