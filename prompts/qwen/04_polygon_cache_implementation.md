# Task: Implement PolygonCache Class (TDD Approach)

## Context
- **Project:** Receipt OCR Text Detection (200k+ LOC)
- **Framework:** PyTorch Lightning 2.1+ with Hydra 1.3+
- **Purpose:** Phase 2.1 - Implement polygon processing cache for 5-8x validation speedup
- **Code Style:** Follow `pyproject.toml` (ruff, mypy with type hints)
- **Approach:** Test-Driven Development (TDD) - tests exist, implement to pass them

## Objective
Implement a `PolygonCache` class that caches PyClipper polygon processing results to reduce validation time from 10x training to <2x training. This is the **critical bottleneck** identified in profiling.

## Requirements

### Functional Requirements
1. **LRU caching** - Least Recently Used eviction when cache full
2. **Deterministic key generation** - Hash polygon geometry + image dimensions + params
3. **Hit/miss tracking** - Track cache effectiveness metrics
4. **Size limits** - Configurable max_size to prevent OOM
5. **Optional disk persistence** - Save/load cache to disk (optional feature)
6. **Thread-safe** - Safe for DataLoader workers (if needed)

### Non-Functional Requirements
1. **Fast lookups** - O(1) get/set operations
2. **Memory efficient** - Store only essential data
3. **Type safe** - Full type hints, passes mypy
4. **Zero accuracy loss** - Cached results must be bit-exact

## Input Files to Reference

### Read These Files First:
```
ocr/datasets/db_collate_fn.py                 # Where PyClipper is called (line 97-107)
tests/performance/test_polygon_caching.py     # TDD tests to pass (currently skipped)
docs/ai_handbook/07_project_management/performance_optimization_plan.md  # Phase 2.1
```

### Understand This Code Pattern:
```python
# From db_collate_fn.py:71-107 (the bottleneck to cache)

def make_prob_thresh_map(self, image, polygons, filename):
    _, h, w = image.shape
    prob_map = np.zeros((h, w), dtype=np.float32)
    thresh_map = np.zeros((h, w), dtype=np.float32)

    for poly in polygons:
        # ... validation ...

        # EXPENSIVE OPERATION - This is what we're caching:
        pco = pyclipper.PyclipperOffset()
        pco.AddPaths(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

        # Shrink for prob_map
        shrinked = pco.Execute(-D)
        # ... fill prob_map ...

        # Dilate for thresh_map
        dilated = pco.Execute(D)
        # ... fill thresh_map ...

    return OrderedDict(prob_map=prob_map, thresh_map=thresh_map)
```

### Project Structure:
```
ocr/
  datasets/
    __init__.py
    base.py
    db_collate_fn.py              # Will integrate cache here
    polygon_cache.py              # CREATE THIS FILE
    preprocessing/
```

## Output Files

### Create:
- `ocr/datasets/polygon_cache.py`

### Update Tests (make them pass):
- `tests/performance/test_polygon_caching.py` - Remove skip decorators, implement test bodies

## Implementation Details

### PolygonCache Class Specification

```python
"""
Polygon processing cache for validation speedup.

Caches expensive PyClipper polygon operations to reduce validation time
from 10x training to <2x training.
"""

import hashlib
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any

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
        # Normalize polygons to canonical form (sort by coordinates)
        # This ensures same polygons produce same key
        poly_normalized = np.sort(polygons.flatten())

        # Create composite key from all inputs
        key_data = [
            poly_normalized.tobytes(),
            str(image_shape).encode(),
            str(params).encode(),
        ]

        # Use fast hash (xxhash or blake2b)
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
        if self.persist_to_disk and len(self._cache) % 100 == 0:
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

    def _save_to_disk(self) -> None:
        """Save cache to disk (internal method)."""
        if not self.persist_to_disk:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / "polygon_cache.pkl"

        with open(cache_file, "wb") as f:
            pickle.dump({
                "cache": self._cache,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
            }, f)

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
```

### Test Requirements (Must Pass These)

The implementation must pass all tests in `tests/performance/test_polygon_caching.py`.

**Update the test file to remove skip decorators and implement test bodies:**

```python
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
            self.sample_image.shape,
            (0.4, 0.3, 0.7),
        )

        # Process polygons
        result = self.collate_fn.make_prob_thresh_map(
            self.sample_image, self.sample_polygons, "test.jpg"
        )

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
            self.sample_image.shape,
            (0.4, 0.3, 0.7),
        )

        # First access - miss
        result = self.cache.get(key)
        assert result is None
        assert self.cache.miss_count == 1

        # Set and get - hit
        self.cache.set(key, {"test": "data"})
        result = self.cache.get(key)
        assert result is not None
        assert self.cache.hit_count == 1

    def test_cache_size_limits(self):
        """Test that cache respects size limits."""
        # Fill cache beyond max_size
        for i in range(110):
            polygons = np.array([[[i, i], [i + 10, i], [i + 10, i + 10], [i, i + 10]]])
            key = self.cache._generate_key(polygons, (3, 64, 64), (0.4, 0.3, 0.7))
            self.cache.set(key, {"index": i})

        # Cache should not exceed max_size
        assert len(self.cache) <= self.cache.max_size

    @pytest.mark.performance
    def test_performance_improvement(self):
        """Test that caching improves performance."""
        import time

        key = self.cache._generate_key(
            self.sample_polygons[0],
            self.sample_image.shape,
            (0.4, 0.3, 0.7),
        )

        # Time uncached operation
        start = time.time()
        result = self.collate_fn.make_prob_thresh_map(
            self.sample_image, self.sample_polygons, "test.jpg"
        )
        uncached_time = time.time() - start

        # Cache the result
        self.cache.set(key, result)

        # Time cached operation
        start = time.time()
        cached_result = self.cache.get(key)
        cached_time = time.time() - start

        # Cached access should be at least 10x faster
        assert cached_time < uncached_time * 0.1

    def test_cache_invalidation(self):
        """Test cache invalidation functionality."""
        key = self.cache._generate_key(
            self.sample_polygons[0],
            self.sample_image.shape,
            (0.4, 0.3, 0.7),
        )
        self.cache.set(key, {"test": "data"})

        assert self.cache.get(key) is not None

        self.cache.invalidate()

        assert self.cache.get(key) is None
        assert len(self.cache) == 0
```

## Validation

### Run These Commands:
```bash
# Type checking
uv run mypy ocr/datasets/polygon_cache.py

# Linting
uv run ruff check ocr/datasets/polygon_cache.py

# Format
uv run ruff format ocr/datasets/polygon_cache.py

# Import test
uv run python -c "from ocr.datasets.polygon_cache import PolygonCache; print('✅ Import successful')"

# Run TDD tests (these MUST pass)
uv run pytest tests/performance/test_polygon_caching.py -v

# Performance test
uv run pytest tests/performance/test_polygon_caching.py::TestPolygonCache::test_performance_improvement -v
```

### Expected Behavior:
- ✅ All type checks pass
- ✅ No linting errors
- ✅ All TDD tests pass (100% pass rate)
- ✅ Performance test shows >10x speedup
- ✅ Cache hit rate >80% after warmup

## Example Usage

After implementation:

```python
from ocr.datasets.polygon_cache import PolygonCache
from ocr.datasets.db_collate_fn import DBCollateFN

# Create cache
cache = PolygonCache(max_size=1000, persist_to_disk=False)

# Use with collate function (integration will be done separately)
collate_fn = DBCollateFN(shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7)

# In collate_fn.make_prob_thresh_map, check cache first:
key = cache._generate_key(polygons, image.shape, (shrink_ratio, thresh_min, thresh_max))
cached = cache.get(key)
if cached is not None:
    return cached

# Otherwise, compute and cache
result = ... # expensive PyClipper operations
cache.set(key, result)
return result
```

## Success Criteria

- [ ] File `ocr/datasets/polygon_cache.py` created
- [ ] All type hints present and mypy passes
- [ ] No ruff linting errors
- [ ] All tests in `test_polygon_caching.py` pass
- [ ] Performance test shows >10x cached vs uncached
- [ ] Cache hit rate >80% on second epoch
- [ ] LRU eviction works correctly
- [ ] Thread-safe for DataLoader workers

## Additional Notes

- **Critical Path:** This is the #1 performance bottleneck (10x slowdown)
- **Expected Impact:** 5-8x validation speedup
- **Memory Budget:** ~200MB for cache with max_size=1000
- **Hash Function:** Use blake2b (fast, secure) or xxhash (fastest)
- **Integration:** Will be done in separate task (Task 2.2)
- **Testing:** TDD approach - tests exist, implement to pass them
