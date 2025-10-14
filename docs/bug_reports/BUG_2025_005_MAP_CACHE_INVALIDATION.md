# BUG #2025-005: Map Generation Fallback Due to Stale Tensor Cache

**Status**: IDENTIFIED
**Severity**: Medium
**Component**: Dataset Caching, Collate Function
**Date Identified**: 2025-10-14
**Identified By**: Claude Code

## Summary

When `cache_transformed_tensors=true` and `load_maps=true` are both enabled, the collate function reports "⚠ Fallback to on-the-fly generation: 16/16 samples (100.0%)" after the first epoch, even though pre-computed maps exist on disk. This causes unnecessary computation and performance degradation.

## Root Cause

The tensor cache stores complete DataItem objects after first access. If the cache was built when `load_maps=false` or before maps were properly loaded, the cached DataItems will have `prob_map=None` and `thresh_map=None`. When these cached items are returned in subsequent epochs, the collate function sees missing maps and falls back to on-the-fly generation.

### Code Flow

1. **First Access** (cache miss):
   - Dataset loads image, polygons, and maps (if `load_maps=true`)
   - Creates DataItem with all fields including maps
   - Caches the complete DataItem
   - Collate function receives maps: "✓ Using .npz maps"

2. **Subsequent Access** (cache hit):
   - Dataset returns cached DataItem via `model_dump()`
   - **Problem**: If cache was built without maps, cached item has `prob_map=None`
   - Collate function sees `None` and falls back to generation

### Evidence from Logs

```
# First validation (Epoch 0 start)
[2025-10-14 16:45:30,034][ocr.datasets.db_collate_fn][INFO] - ✓ Using .npz maps (from cache or disk): 16/16 samples (100.0%)

# Second validation (Epoch 0 end, after cache warm-up)
[2025-10-14 16:45:52,380][ocr.datasets.db_collate_fn][WARNING] - ⚠ Fallback to on-the-fly generation: 16/16 samples (100.0%)

# Cache hits confirmed
[2025-10-14 16:46:04,473][ocr.datasets.base][INFO] - [CACHE HIT] Returning cached tensor for index 0
```

## Impact

### Performance Impact
- **Validation time increase**: On-the-fly map generation adds ~5-10ms per sample
- **Wasted computation**: Re-computes maps that already exist on disk
- **Memory efficiency**: Maps aren't cached in memory despite `cache_maps=true`

### Correctness Impact
- **No accuracy impact**: Generated maps are identical to pre-computed ones
- **Determinism maintained**: Map generation is deterministic

## Affected Configurations

This bug affects systems with:
- `cache_config.cache_transformed_tensors: true`
- `load_maps: true`
- Pre-existing tensor cache from runs with different configuration

## Workaround

**Immediate Fix**: Clear tensor cache when configuration changes

```bash
# Option 1: Clear all caches
rm -rf /tmp/ocr_cache/

# Option 2: Disable tensor caching for datasets that use load_maps
# In configs/data/base.yaml:
cache_transformed_tensors: false  # For validation dataset only
```

## Recommended Solution

### Short-term Fix

Add cache versioning based on configuration hash:

```python
# In ocr/datasets/schemas.py CacheConfig
def get_cache_version(self) -> str:
    """Generate cache version hash from configuration."""
    config_str = f"{self.cache_transformed_tensors}_{self.cache_images}_{self.cache_maps}_{self.load_maps}"
    return hashlib.md5(config_str.encode()).hexdigest()[:8]
```

### Long-term Fix

1. **Cache Versioning**: Include configuration hash in cache keys
2. **Cache Validation**: Check cache compatibility on load
3. **Automatic Invalidation**: Clear cache when configuration changes
4. **Better Logging**: Warn when cache may be stale

### Implementation Plan

1. Add `cache_version` field to CacheConfig
2. Modify CacheManager to include version in cache keys
3. Add cache validation on dataset initialization
4. Update documentation with cache management best practices

## Testing Plan

1. **Scenario 1**: Enable `load_maps` with existing cache
   - **Expected**: Cache invalidated automatically
   - **Verification**: Check "Using .npz maps" appears in all epochs

2. **Scenario 2**: Disable `cache_transformed_tensors`
   - **Expected**: Maps loaded from disk every time
   - **Verification**: No fallback warnings

3. **Scenario 3**: Fresh cache with `load_maps=true`
   - **Expected**: Maps cached and reused
   - **Verification**: Cache hits + maps loaded

## Related Issues

- **BUG_2025_002**: Mixed precision performance degradation
- **BUG_2025_004**: Cache performance impact investigation

## References

- [ocr/datasets/base.py:422-427](../../ocr/datasets/base.py#L422-L427) - Cache retrieval
- [ocr/datasets/base.py:570-584](../../ocr/datasets/base.py#L570-L584) - DataItem creation with maps
- [ocr/datasets/db_collate_fn.py:138-174](../../ocr/datasets/db_collate_fn.py#L138-L174) - Map fallback logic
- [configs/data/base.yaml:44-74](../../configs/data/base.yaml#L44-L74) - Dataset configuration

## Status Updates

- **2025-10-14**: Bug identified during mixed precision investigation
- **2025-10-14**: Workaround documented, fix designed
- **Status**: Awaiting implementation of cache versioning system
