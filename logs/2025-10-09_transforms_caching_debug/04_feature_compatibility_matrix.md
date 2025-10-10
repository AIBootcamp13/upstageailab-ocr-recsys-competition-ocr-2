# Performance Feature Compatibility Matrix
**Date**: 2025-10-10
**Last Updated**: After pipeline restoration

---

## Feature Overview

The OCR pipeline has several performance optimization features that can be enabled in `configs/data/base.yaml`:

| Feature | Parameter | Purpose | Memory Impact | Performance Impact |
|---------|-----------|---------|---------------|-------------------|
| **NPZ Map Loading** | `load_maps` | Load pre-generated probability/threshold maps from .npz files | None | ⚡ Very Fast (no generation) |
| **NPZ Map Preloading** | `preload_maps` | Preload .npz maps into RAM | 🔴 High | ⚡⚡ Extremely Fast |
| **Image Preloading** | `preload_images` | Preload decoded images into RAM | 🟡 Medium | ⚡ Fast (no disk I/O) |
| **Image Pre-normalization** | `prenormalize_images` | Normalize images during preloading | 🟡 Medium+ | ⚡ Fast (no runtime normalization) |
| **Tensor Caching** | `cache_transformed_tensors` | Cache transformed tensors after augmentation | 🟡 Medium | ⚡ Fast (validation only) |
| **Document Preprocessing** | N/A | Preprocessing pipeline in transforms | None | Normal |

---

## Compatibility Matrix

### ✅ Verified Safe Combinations

| load_maps | preload_maps | preload_images | prenormalize | cache_tensors | Status | Notes |
|:---------:|:------------:|:--------------:|:------------:|:-------------:|:------:|-------|
| false | false | false | false | false | ✅ **Verified** | Current baseline - all tests pass |
| false | false | true | false | false | ⏳ To test | Image preloading only |
| false | false | true | true | false | ⏳ To test | Image preloading + normalization |
| false | false | true | false | true | ⏳ To test | Image + tensor caching |
| false | false | false | false | true | ⚠️ **Untested** | Tensor caching alone |

### ⚠️ Requires .npz Maps (Not Generated Yet)

| load_maps | preload_maps | preload_images | prenormalize | cache_tensors | Status | Notes |
|:---------:|:------------:|:--------------:|:------------:|:-------------:|:------:|-------|
| true | false | false | false | false | ⚠️ **Requires maps** | Must generate .npz files first |
| true | true | false | false | false | ⚠️ **Requires maps** | Preload from disk to RAM |
| true | any | true | false | false | ⚠️ **Unclear** | May conflict - needs testing |

### ❌ Known Incompatible Combinations

| load_maps | preload_maps | preload_images | prenormalize | cache_tensors | Status | Reason |
|:---------:|:------------:|:--------------:|:------------:|:-------------:|:------:|--------|
| false | true | any | any | any | ❌ **Invalid** | Can't preload maps that aren't loaded |
| true | false | any | any | true | ⚠️ **Untested** | Caching with map loading - unclear behavior |

---

## Feature Dependencies

### Dependency Graph

```
load_maps
  ├─> preload_maps (requires load_maps=true)
  └─> May conflict with cache_transformed_tensors?

preload_images
  ├─> prenormalize_images (optional, requires preload_images=true)
  └─> cache_transformed_tensors (works well together)

cache_transformed_tensors
  └─> Validation speedup only (train uses random augmentation)
```

---

## Feature Details

### 1. load_maps

**Purpose**: Load pre-generated .npz map files instead of generating on-the-fly

**Requirements**:
- ✅ .npz files must exist in maps directory
- ✅ Maps must match current images
- ✅ Maps must be properly generated

**Current Status**: ⚠️ **Disabled** - .npz maps may not exist or be outdated

**Config**:
```yaml
datasets:
  train_dataset:
    load_maps: false  # Disable .npz map loading
```

**When to Enable**:
- After generating .npz maps with `test_preprocess_maps.py`
- When maps are verified to be correct
- When disk I/O for maps is faster than on-the-fly generation

---

### 2. preload_maps

**Purpose**: Preload .npz maps from disk into RAM for faster access

**Requirements**:
- ✅ `load_maps=true` (must be enabled first)
- ✅ Sufficient RAM (maps can be large)

**Current Status**: ⚠️ **Disabled** - Requires load_maps=true first

**Config**:
```yaml
datasets:
  val_dataset:
    load_maps: true
    preload_maps: true  # Preload to RAM
```

**When to Enable**:
- After load_maps is working
- When RAM is available
- For maximum validation speed

---

### 3. preload_images

**Purpose**: Decode and preload images into RAM to eliminate disk I/O

**Requirements**:
- ✅ Sufficient RAM for decoded images
- ✅ Training dataset may be too large (use for validation only)

**Current Status**: ⚠️ **Disabled** - Works but not tested recently

**Config**:
```yaml
datasets:
  val_dataset:
    preload_images: true  # Decode images to RAM
```

**When to Enable**:
- For validation dataset (smaller)
- When RAM is available (~1-2GB for 400 images)
- To speed up data loading

**Memory Estimate**:
- ~400 validation images × ~640×640×3 × 1 byte = ~500MB (uint8)
- ~800 training images × ~640×640×3 × 1 byte = ~1GB (uint8)

---

### 4. prenormalize_images

**Purpose**: Pre-normalize images during preloading (ImageNet normalization)

**Requirements**:
- ✅ `preload_images=true` (must be enabled first)
- ✅ Slightly more RAM (float32 vs uint8)

**Current Status**: ⚠️ **Disabled** - Requires preload_images=true first

**Config**:
```yaml
datasets:
  val_dataset:
    preload_images: true
    prenormalize_images: true  # Store as normalized float32
```

**When to Enable**:
- After preload_images is working
- For maximum speed (no runtime normalization)
- When RAM is available

**Memory Impact**:
- uint8: 1 byte per value
- float32: 4 bytes per value
- Memory increase: ~4x

---

### 5. cache_transformed_tensors

**Purpose**: Cache final transformed tensors after augmentation (validation only)

**Requirements**:
- ✅ Validation dataset only (train uses random augmentation)
- ✅ RAM for cached tensors

**Current Status**: ⚠️ **Disabled** - Was working in Phase 6E, needs retesting

**Config**:
```yaml
datasets:
  val_dataset:
    cache_transformed_tensors: true  # Cache validation tensors
```

**When to Enable**:
- For validation dataset only
- After other features are stable
- To maximize validation speed

**How It Works**:
1. First access: Transform image, cache result
2. Subsequent access: Return cached tensor
3. Validation runs multiple times → big speedup

---

## Recommended Configurations

### Development (Current - Safest)
```yaml
datasets:
  train_dataset:
    preload_maps: false
    load_maps: false
    preload_images: false
    cache_transformed_tensors: false

  val_dataset:
    preload_maps: false
    load_maps: false
    preload_images: false
    cache_transformed_tensors: false
```

**Pros**: ✅ Safe, verified, no surprises
**Cons**: ❌ Slower data loading

---

### Fast Validation (Recommended Next Step)
```yaml
datasets:
  train_dataset:
    preload_maps: false
    load_maps: false
    preload_images: false  # Training dataset too large
    cache_transformed_tensors: false

  val_dataset:
    preload_maps: false
    load_maps: false
    preload_images: true   # ← Enable for validation
    cache_transformed_tensors: false
```

**Pros**: ✅ Faster validation, minimal risk
**Cons**: ❌ Uses ~500MB RAM

---

### Maximum Validation Speed (After Testing)
```yaml
datasets:
  train_dataset:
    preload_maps: false
    load_maps: false
    preload_images: false
    cache_transformed_tensors: false

  val_dataset:
    preload_maps: false
    load_maps: false
    preload_images: true
    prenormalize_images: true  # ← Enable after preload_images works
    cache_transformed_tensors: true  # ← Enable for maximum speed
```

**Pros**: ✅ Maximum validation speed
**Cons**: ❌ Uses ~2GB RAM, needs thorough testing

---

### With NPZ Maps (Future - After Map Generation)
```yaml
datasets:
  train_dataset:
    preload_maps: false
    load_maps: true   # ← Load pre-generated maps
    preload_images: false
    cache_transformed_tensors: false

  val_dataset:
    preload_maps: true  # ← Preload maps to RAM
    load_maps: true
    preload_images: true
    cache_transformed_tensors: true
```

**Requirements**: ✅ Generate .npz maps first
**Pros**: ✅ Fastest possible (no on-the-fly generation)
**Cons**: ❌ High RAM usage, requires map generation

---

## Testing Checklist

Before enabling any feature:

### Unit Tests
- [ ] Run `pytest tests/unit/` - all should pass
- [ ] Run dataset tests specifically
- [ ] Run preprocessing tests

### Integration Test (Small)
```bash
HYDRA_FULL_ERROR=1 uv run python runners/train.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=50 \
  trainer.limit_val_batches=10 \
  data.train_num_samples=800 \
  data.val_num_samples=100 \
  logger.wandb.enabled=false \
  datasets.val_dataset.NEW_FEATURE=true  # Enable feature to test
```

### Integration Test (Full)
```bash
HYDRA_FULL_ERROR=1 uv run python runners/train.py \
  trainer.max_epochs=3 \
  trainer.limit_val_batches=50 \
  datasets.val_dataset.NEW_FEATURE=true  # Enable feature to test
```

### Monitoring
- Check for memory leaks
- Monitor RAM usage
- Verify no warnings/errors
- Check validation speed improvement
- Verify metrics are correct

---

## Open Questions

### 1. load_maps + cache_transformed_tensors
**Question**: Can these work together? Or does map loading bypass transforms?

**Hypothesis**: If load_maps bypasses transform pipeline, tensor caching won't help.

**Test**: Enable both, check if caching activates.

---

### 2. preload_images + load_maps
**Question**: What happens if both are enabled?

**Hypothesis**: May conflict if map loading expects images from disk.

**Test**: Enable both, check for errors.

---

### 3. prenormalize_images Type Handling
**Question**: Does prenormalize_images change image type in a way that breaks transforms?

**Status**: ✅ Should work - BUG-2025-002 fix handles both uint8 and float32

**Test**: Enable and verify transforms work correctly.

---

## Summary

### Current Baseline (Safe)
```yaml
All features disabled → ✅ Verified working
```

### Next Steps
1. ⏳ Test `preload_images=true` for validation
2. ⏳ Test `cache_transformed_tensors=true` for validation
3. ⏳ Test combined `preload_images + cache_transformed_tensors`
4. ⏳ Generate .npz maps if needed
5. ⏳ Test map loading features

### Long-term Goals
- Achieve 0.25+ h-mean with 800/100 images, 1 epoch
- Enable safe performance features for faster iteration
- Document all compatible configurations
- Create automated tests for feature combinations

---

**Last Updated**: 2025-10-10 20:15:00
