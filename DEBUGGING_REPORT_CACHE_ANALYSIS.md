# Debugging Report: OCR Dataset Caching Analysis

**Date:** October 10, 2025
**Analyst:** GitHub Copilot
**Files Analyzed:**
- `ocr/datasets/base.py`
- `ocr/datasets/db_collate_fn.py`
- `configs/data/base.yaml`

---

## Executive Summary

Two separate caching mechanisms were identified:
1. **RAM Preloading Cache** (`preload_maps=False`) - Controlled by configuration
2. **Lazy Disk Loading** (always active) - Runs independently in `__getitem__`

The `.npz` maps are loading despite `preload_maps=False` because the lazy loading mechanism in `__getitem__` (lines 333-345 of `base.py`) **always attempts to load maps from disk** when they're not in the RAM cache. This is the intended fallback behavior.

---

## 1. Root Cause Analysis: `preload_maps` Behavior

### The Confusion

You observed this log output:
```log
INFO ocr.datasets.base - Tensor caching enabled - will cache 404 transformed samples...
✓ Using pre-loaded .npz maps: 16/16 samples (100.0%)
```

The message "Using pre-loaded .npz maps" is **misleading**. It doesn't mean maps were preloaded into RAM.

### The Two Mechanisms

#### Mechanism A: RAM Preloading (`preload_maps=True`)
- **Location:** `_preload_maps_to_ram()` method (lines 115-142)
- **Trigger:** Only runs when `self.preload_maps = True` in `__init__`
- **Behavior:** Loads all `.npz` files into `self.maps_cache` dict at initialization
- **Current Status:** **DISABLED** (config shows `preload_maps: false`)

#### Mechanism B: Lazy Disk Loading (Always Active)
- **Location:** `__getitem__()` method (lines 333-345)
- **Trigger:** Runs on every sample access when map is not in `self.maps_cache`
- **Behavior:** Attempts to load `.npz` file from disk for the requested sample
- **Current Status:** **ACTIVE** (no config to disable)

### The Code Flow

```python
# In __getitem__ (line 333-345)
if image_filename in self.maps_cache:
    # Path A: Use RAM cache (only if preload_maps=True was set)
    item["prob_map"] = self.maps_cache[image_filename]["prob_map"]
    item["thresh_map"] = self.maps_cache[image_filename]["thresh_map"]
else:
    # Path B: ALWAYS ACTIVE - Load from disk on-demand
    maps_dir = self.image_path.parent / f"{self.image_path.name}_maps"
    map_filename = maps_dir / f"{Path(image_filename).stem}.npz"

    if map_filename.exists():
        maps_data = np.load(map_filename)
        item["prob_map"] = maps_data["prob_map"]
        item["thresh_map"] = maps_data["thresh_map"]
```

### Why the Log Says "pre-loaded"

The log message comes from `db_collate_fn.py` (line 87):
```python
if "prob_map" in item and "thresh_map" in item:
    # This checks if maps EXIST in the item, not HOW they were loaded
    console.print(f"✓ Using pre-loaded .npz maps: {preloaded_count}/{total_samples}")
```

**The issue:** The collate function can't distinguish between:
1. Maps loaded from RAM cache (`self.maps_cache`)
2. Maps loaded lazily from disk in `__getitem__`

Both result in `"prob_map"` and `"thresh_map"` keys existing in the item dict.

### The Truth

With `preload_maps=False`:
- ✅ Maps are **NOT** preloaded into RAM at initialization
- ✅ Maps **ARE** loaded from disk on-demand in `__getitem__`
- ❌ The log message is **misleading** - it should say "Using .npz maps" not "pre-loaded"

---

## 2. Verification Plan: `cache_transformed_tensors`

### Current Behavior

The tensor cache is working correctly:
```python
# Line 221-223: Early return if cached
if self.cache_transformed_tensors and idx in self.tensor_cache:
    return self.tensor_cache[idx]

# Line 353-355: Store after first access
if self.cache_transformed_tensors:
    self.tensor_cache[idx] = item
```

### Verification Code

Add this logging to confirm cache hits:

```python
def __getitem__(self, idx):
    image_filename = list(self.anns.keys())[idx]

    # Check if final transformed tensor is cached (Phase 6E)
    if self.cache_transformed_tensors and idx in self.tensor_cache:
        # ADD THIS LINE:
        self.logger.debug(f"[CACHE HIT] Returning cached tensor for index {idx} (file: {image_filename})")
        return self.tensor_cache[idx]
```

**Better approach for production monitoring:**

Add epoch-level statistics tracking:

```python
def __init__(self, ...):
    # ... existing code ...
    self.tensor_cache = {}
    # ADD THESE:
    self._cache_hit_count = 0
    self._cache_miss_count = 0

def __getitem__(self, idx):
    image_filename = list(self.anns.keys())[idx]

    # Check if final transformed tensor is cached (Phase 6E)
    if self.cache_transformed_tensors and idx in self.tensor_cache:
        # ADD THIS:
        self._cache_hit_count += 1
        return self.tensor_cache[idx]

    # ADD THIS (after line 223):
    if self.cache_transformed_tensors:
        self._cache_miss_count += 1

    # ... rest of __getitem__ ...

# ADD NEW METHOD:
def log_cache_statistics(self):
    """Call this at the end of each epoch from training loop"""
    if not self.cache_transformed_tensors:
        return

    total_accesses = self._cache_hit_count + self._cache_miss_count
    hit_rate = (self._cache_hit_count / total_accesses * 100) if total_accesses > 0 else 0

    self.logger.info(
        f"Tensor Cache Stats - Hits: {self._cache_hit_count}, "
        f"Misses: {self._cache_miss_count}, "
        f"Hit Rate: {hit_rate:.1f}%, "
        f"Cache Size: {len(self.tensor_cache)}"
    )

    # Reset counters for next epoch
    self._cache_hit_count = 0
    self._cache_miss_count = 0
```

---

## 3. Action Plan

### Immediate Actions

1. **Fix Misleading Log Message** (Low Priority)
   - Update `db_collate_fn.py` line 87 to be more accurate
   - Suggestion: Change "pre-loaded" to "disk-loaded" or "available"

2. **Verify Tensor Cache** (High Priority)
   - Add the debug logging code above
   - Run training for 2 epochs with `cache_transformed_tensors: true`
   - Check logs for `[CACHE HIT]` messages in epoch 2
   - Expected: First epoch = 0 hits, Second epoch = 100% hits

3. **Benchmark Performance** (Medium Priority)
   - Compare training speed between:
     - `cache_transformed_tensors: false` (baseline)
     - `cache_transformed_tensors: true` (should be faster in epoch 2+)
   - Measure RAM usage to ensure it doesn't exceed limits

### Configuration Recommendations

For your validation dataset (404 samples):
```yaml
val_dataset:
  preload_maps: false          # Keep disabled - lazy loading is sufficient
  preload_images: true          # ENABLE - 404 images fit in RAM
  cache_transformed_tensors: true  # ENABLE - cache after transforms
```

**Rationale:**
- Lazy map loading is fast enough (only 404 samples)
- Image preloading eliminates disk I/O completely
- Tensor caching eliminates transform overhead after first epoch

### Long-Term Improvements

1. **Clarify Terminology**
   - Rename `preload_maps` → `preload_maps_to_ram`
   - Add docstrings explaining lazy loading fallback

2. **Add Cache Metrics**
   - Implement `log_cache_statistics()` method above
   - Integrate with WandB logging for monitoring

3. **Optional: Disable Lazy Loading**
   - Add config flag `enable_lazy_map_loading: bool = True`
   - Allow users to completely disable map loading if not needed

---

## 4. Expected Outcomes

### After Implementing Verification Code

**Epoch 1 Logs:**
```
INFO ocr.datasets.base - Tensor caching enabled - will cache 404 transformed samples...
[First batch - lots of transform operations]
✓ Using .npz maps: 16/16 samples (100.0%)
[End of epoch 1]
Tensor Cache Stats - Hits: 0, Misses: 404, Hit Rate: 0.0%, Cache Size: 404
```

**Epoch 2+ Logs:**
```
[Second batch - should be much faster]
✓ Using .npz maps: 16/16 samples (100.0%)
[End of epoch 2]
Tensor Cache Stats - Hits: 404, Misses: 0, Hit Rate: 100.0%, Cache Size: 404
```

### Performance Expectations

With `cache_transformed_tensors=true`:
- **Epoch 1:** Normal speed (building cache)
- **Epoch 2+:** ~2-5x faster data loading (no transforms, no image decoding)
- **RAM Usage:** +~200-400 MB for 404 cached samples

---

## Appendix: Code Snippets

### A. Quick Debug Snippet (Add to line 223)

```python
if self.cache_transformed_tensors and idx in self.tensor_cache:
    # QUICK DEBUG: Print once every 50 samples to avoid spam
    if idx % 50 == 0:
        self.logger.info(f"[CACHE HIT] idx={idx}, file={image_filename}")
    return self.tensor_cache[idx]
```

### B. Enhanced Stats Tracking (Full Implementation)

```python
# In __init__ (after line 50):
self._cache_stats = {
    'hits': 0,
    'misses': 0,
    'disk_loads': 0,
    'ram_loads': 0
}

# In __getitem__ (replace lines 221-223):
if self.cache_transformed_tensors and idx in self.tensor_cache:
    self._cache_stats['hits'] += 1
    return self.tensor_cache[idx]

if self.cache_transformed_tensors:
    self._cache_stats['misses'] += 1

# Track map loading method (line 333):
if image_filename in self.maps_cache:
    self._cache_stats['ram_loads'] += 1
    # ... existing code ...
else:
    self._cache_stats['disk_loads'] += 1
    # ... existing code ...
```

---

**End of Report**
