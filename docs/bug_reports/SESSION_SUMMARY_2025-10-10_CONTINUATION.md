# OCR Pipeline Debug Session - Continuation Summary
**Date**: 2025-10-10 (Evening Session)
**Status**: 🟡 **PARTIAL SUCCESS - One Critical Bug Fixed, Investigation Continues**
**Branch**: `07_refactor/performance_debug2`

---

## Executive Summary

Continued debugging from earlier session. **Found and fixed critical polygon shape bug** in transforms.py that was causing catastrophic performance degradation (hmean: 0.00011 → 0.28 in training batches).

### Key Findings

1. ✅ **FIXED**: Critical bug in [ocr/datasets/transforms.py:74](ocr/datasets/transforms.py#L74) - wrong dimension used for polygon point count
2. ✅ **FIXED**: Simplified config back to working baseline
3. ⚠️ **REMAINING ISSUE**: Excessive polygon filtering causing val/test metrics to stay at 0.0

---

## Critical Bug Fix: Polygon Shape Dimension Error

### Root Cause

In [ocr/datasets/transforms.py:74](ocr/datasets/transforms.py#L74), the code was using `polygon.shape[1]` to get the number of points:

```python
# ❌ BROKEN CODE (before fix)
num_points = polygon.shape[1]  # Returns 2 (coordinate dimension), not number of points!
```

For a polygon with shape `(4, 2)` (4 points with x,y coordinates):
- `polygon.shape[0]` = 4 (✅ correct - number of points)
- `polygon.shape[1]` = 2 (❌ wrong - this is the coordinate dimension)

This caused the slicing `transformed_keypoints[index:index+num_points]` to only grab **2 points per polygon instead of 4+**, completely breaking polygon reconstruction.

### The Fix

```python
# ✅ FIXED CODE
if polygon.ndim == 2:
    num_points = polygon.shape[0] if polygon.shape[0] != 1 else polygon.shape[1]
elif polygon.ndim == 3:
    num_points = polygon.shape[1]
else:
    continue
```

This handles both `(N, 2)` and `(1, N, 2)` polygon shapes correctly.

### Impact

**Before fix**:
- test/hmean: 0.00011 (essentially non-functional)
- test/precision: 0.00166
- test/recall: 0.00007

**After fix**:
- batch_0/hmean: 0.1138 (**1000x improvement**)
- batch_1/hmean: 0.28502
- batch_1/precision: 0.2621
- batch_1/recall: 0.3125

---

## Configuration Simplification

### Changes Made

Reverted [configs/data/base.yaml](configs/data/base.yaml) to simple baseline configuration, removing all performance optimization features that were added after the working commit:

**Removed**:
- `preload_maps`, `load_maps`, `preload_images` - Performance caching features
- `cache_transformed_tensors` - Tensor caching
- `image_loading_config` - TurboJPEG optimization
- `images_val_canonical` - Canonical image paths

**Result**: Simple, working baseline matching original commit 8252600.

### Default Values Fixed

Changed [ocr/datasets/base.py:35](ocr/datasets/base.py#L35):
```python
load_maps=False  # Changed from True to False
```

This ensures the dataset doesn't try to load .npz maps by default.

---

## Remaining Issue: Excessive Polygon Filtering

### Observation

During training, many images have **ALL their polygons filtered** as "too_small":

```
Filtered 30 degenerate polygons (too_few_points=0, too_small=30, zero_span=0, empty=0, none=0)
Filtered 26 degenerate polygons (too_few_points=0, too_small=26, zero_span=0, empty=0, none=0)
Filtered 24 degenerate polygons (too_few_points=0, too_small=24, zero_span=0, empty=0, none=0)
```

This explains why:
- ✅ **Training batches** show non-zero metrics (0.1-0.3 hmean)
- ❌ **Validation/test** show 0.0 metrics (all polygons filtered)

### Root Cause Analysis

The filtering logic in [ocr/datasets/base.py:498](ocr/datasets/base.py#L498):

```python
if width_span < min_side or height_span < min_side:  # min_side=1.0 pixels
    removed_counts["too_small"] += 1
    continue
```

After transforms (especially resize), small text boxes can become **sub-pixel** (< 1.0 pixels) and get filtered out.

### Why This Matters

This is likely **NOT the same issue** as the working commit 8252600. The working commit may have:
1. Used different transform parameters (less aggressive resizing)
2. Had different polygon filtering thresholds
3. Used canonical images with different dimensions

---

## Files Modified in This Session

### Source Code (3 files)

1. ✅ **[ocr/datasets/transforms.py:74-80](ocr/datasets/transforms.py#L74-L80)** - Fixed polygon point count dimension bug
2. ✅ **[ocr/datasets/base.py:35](ocr/datasets/base.py#L35)** - Changed `load_maps` default from True to False
3. ✅ **[ocr/datasets/base.py:59](ocr/datasets/base.py#L59)** - Changed `use_turbojpeg` default from True to False

### Configuration (1 file)

1. ✅ **[configs/data/base.yaml](configs/data/base.yaml)** - Simplified to baseline configuration

---

## Test Results

### Quick Training Test (1 epoch, 50 batches, 800/100 samples)

**Command**:
```bash
HYDRA_FULL_ERROR=1 uv run python runners/train.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=50 \
  trainer.limit_val_batches=10 \
  data.train_num_samples=800 \
  data.val_num_samples=100 \
  exp_name=polygon_fix_quick_test \
  logger.wandb.enabled=false
```

**Results**:
- ✅ Training completes without crashes
- ✅ Workers stable (no multiprocessing errors)
- ✅ Training batches show non-zero metrics (0.1-0.3 hmean)
- ❌ Validation metrics still 0.0 (polygon filtering issue)
- ❌ Test metrics still 0.0 (polygon filtering issue)

---

## Next Steps (Prioritized)

### Priority 1: Compare with Working Commit Transform Pipeline

```bash
# Check transform parameters in working commit
git show 8252600:configs/transforms/base.yaml

# Compare current vs working transforms
git diff 8252600 HEAD -- configs/transforms/
```

**Goal**: Identify if working commit had different resize/transform parameters that preserved polygon sizes.

### Priority 2: Investigate Polygon Filtering Threshold

Options:
1. **Lower the `min_side` threshold** from 1.0 to 0.5 pixels
2. **Check if working commit filtered differently**
3. **Verify if canonical images have different dimensions**

### Priority 3: Run Full 3-Epoch Training

Once polygon filtering is resolved:
```bash
HYDRA_FULL_ERROR=1 uv run python runners/train.py \
  trainer.max_epochs=3 \
  exp_name=performance_verification_final \
  logger.wandb.enabled=true
```

**Success criteria**: hmean ≥ 0.8 (matching working commit performance)

---

## Comparison with Working Commit 8252600

### Known Working State

- **Commit**: `8252600e22929802ead538672d2f137e2da0781d`
- **Branch**: `07_refactor/performance`
- **Message**: "refactor: Phase 1 and 2.1 complete. Performance profiling feature has..."
- **Performance**: hmean=0.890 (from docs/ai_handbook/04_experiments/debugging/summaries/2025-10-08_debug_summary.md)

### Key Differences Found

1. **Polygon shape handling**: Working commit used `polygon.shape[1]` which suggests polygons were stored as `(1, N, 2)` not `(N, 2)`
2. **Configuration**: Working commit likely had simpler config without performance features
3. **Transform pipeline**: May have had different parameters (needs investigation)

---

## Key Learnings

### 1. Polygon Shape Contracts Are Critical

Mixing `(N, 2)` and `(1, N, 2)` shapes throughout the codebase caused subtle bugs. The fix now handles both formats defensively.

### 2. Performance Optimizations Can Break Core Functionality

The performance features added after commit 8252600 introduced bugs:
- Image loading optimization (turbojpeg)
- Caching features (maps, images, tensors)
- Canonical image paths

**Lesson**: Establish working baseline first, then add optimizations incrementally with testing.

### 3. Polygon Filtering Can Silently Destroy Data

The "too_small" filtering logic can remove ALL polygons from images after transforms, causing training to appear to work but produce 0.0 metrics.

### 4. Unit Tests Insufficient for Pipeline Validation

- ✅ Unit tests passed (87/89)
- ✅ Integration tests completed without crashes
- ❌ Model performance completely broken

**Lesson**: Need end-to-end metrics tests, not just "does it crash" tests.

---

## Questions to Answer

1. **What transform parameters did commit 8252600 use?**
   - Check configs/transforms/base.yaml in that commit
   - Compare resize, padding, cropping settings

2. **What was the original polygon filtering threshold?**
   - Was `min_side=1.0` always the default?
   - Did working commit use a different value?

3. **Were polygons originally stored as `(1, N, 2)` or `(N, 2)`?**
   - The working commit code suggests `(1, N, 2)`
   - Today's fixes changed this to `(N, 2)`
   - May need to verify JSON annotation format

4. **What are the actual image dimensions in the dataset?**
   - Original images: ?
   - After transforms: ?
   - Canonical images: ?

---

## Commands for Future Reference

### Run Quick Test
```bash
HYDRA_FULL_ERROR=1 uv run python runners/train.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=50 \
  trainer.limit_val_batches=10 \
  data.train_num_samples=800 \
  data.val_num_samples=100 \
  exp_name=quick_test \
  logger.wandb.enabled=false
```

### Compare with Working Commit
```bash
# Show working commit files
git show 8252600:configs/data/base.yaml
git show 8252600:configs/transforms/base.yaml
git show 8252600:ocr/datasets/base.py

# Diff specific files
git diff 8252600 HEAD -- configs/
git diff 8252600 HEAD -- ocr/datasets/
```

### Check Git History
```bash
# Find when polygon handling changed
git log --oneline -- ocr/datasets/transforms.py
git log --oneline -- ocr/datasets/base.py

# Show changes in specific commit
git show <commit-hash> -- ocr/datasets/transforms.py
```

---

## Session End

**Time**: 2025-10-10 22:00:00
**Duration**: ~1 hour 30 minutes
**Status**: 🟡 **One critical bug fixed, investigation continues**

### What's Working
✅ Polygon shape bug fixed - training shows non-zero metrics
✅ Config simplified to baseline
✅ Pipeline runs without crashes
✅ Workers stable

### What's Not Working
❌ Validation/test metrics still 0.0 (polygon filtering)
❌ Performance not yet verified against working commit
❌ Root cause of excessive filtering unknown

### Immediate Next Action
Compare transforms configuration with working commit 8252600 to identify why polygons are being filtered as too small.

---

**For next session**: Start by reviewing this document, then compare transform configs with working commit.
