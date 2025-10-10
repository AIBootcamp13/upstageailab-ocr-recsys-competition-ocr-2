# ✅ COMPLETED: Disable .npz Maps Loading

## Summary

Successfully implemented and tested a new `load_maps` parameter to completely disable `.npz` map loading in the OCR dataset.

## Changes Made

### 1. Code Changes (`ocr/datasets/base.py`)

- ✅ Added `load_maps=True` parameter to `__init__`
- ✅ Wrapped map loading logic in `if self.load_maps:` check
- ✅ Backward compatible (defaults to `True` to maintain existing behavior)

### 2. Configuration Updates

- ✅ `configs/data/base.yaml` - Set `load_maps: false` for all datasets
- ✅ `configs/data/canonical.yaml` - Set `load_maps: false` for all datasets

### 3. Testing

- ✅ Created test script (`test_load_maps_disabled.py`)
- ✅ Verified `load_maps=False` prevents map loading
- ✅ Verified `load_maps=True` enables map loading
- ✅ All tests passed

## How to Use

### Disable .npz Map Loading

```yaml
# In your config file (e.g., configs/data/canonical.yaml)
datasets:
  val_dataset:
    load_maps: false  # No .npz files will be loaded
```

### Enable .npz Map Loading (Default)

```yaml
datasets:
  val_dataset:
    load_maps: true  # Load .npz files from disk or RAM cache
```

## Configuration Options

| Parameter | Default | Behavior |
|-----------|---------|----------|
| `load_maps: false` | No | Completely disables map loading (no RAM, no disk) |
| `load_maps: true, preload_maps: false` | - | Lazy load maps from disk on-demand |
| `load_maps: true, preload_maps: true` | - | Preload all maps into RAM at initialization |

## Verification

Run your training command:

```bash
HYDRA_FULL_ERROR=1 uv run python runners/train.py \
  exp_name=test_no_maps \
  trainer.max_epochs=1 \
  data=canonical \
  logger.wandb.enabled=false
```

**Expected output:**
```log
⚠ Fallback to on-the-fly generation: 16/16 samples (100.0%)
```

Instead of:
```log
✓ Using .npz maps (from cache or disk): 16/16 samples (100.0%)
```

## What Happens When Maps Are Disabled?

When `load_maps: false`:
1. ❌ No `.npz` files are loaded from disk
2. ❌ No `.npz` files are preloaded into RAM
3. ✅ The `DBCollateFN` automatically generates maps on-the-fly
4. ✅ Training continues normally (with slight performance overhead)

## Performance Impact

| Configuration | Speed | RAM Usage | Disk I/O |
|---------------|-------|-----------|----------|
| `load_maps: false` | Slower (map generation) | Low | None |
| `load_maps: true, preload_maps: false` | Fast | Low | Per batch |
| `load_maps: true, preload_maps: true` | Fastest | High | At init |

## Files Modified

1. `ocr/datasets/base.py` - Added load_maps parameter and logic
2. `configs/data/base.yaml` - Set load_maps: false
3. `configs/data/canonical.yaml` - Set load_maps: false
4. `test_load_maps_disabled.py` - Test script (NEW)
5. `DISABLE_NPZ_MAPS_IMPLEMENTATION.md` - Documentation (NEW)

## Next Steps

Your configurations are now set to **disable** .npz map loading. The maps will be generated on-the-fly by the collate function.

If you want to **re-enable** map loading in the future, simply change:
```yaml
load_maps: false  # Change to true
```

---

**Status:** ✅ Complete and tested
**Date:** October 10, 2025
