# Disabling .npz Map Loading - Implementation Summary

## Problem

The `.npz` map files were being loaded even when `preload_maps: false` was set in the configuration. This was because:

1. **`preload_maps`** only controlled RAM preloading at initialization
2. **Lazy disk loading** in `__getitem__` (lines 362-379) happened **unconditionally** with no way to disable it

## Solution

Added a new configuration parameter `load_maps` to completely disable map loading.

### Code Changes

#### 1. Added `load_maps` Parameter to Dataset (`ocr/datasets/base.py`)

**New parameter in `__init__`:**
```python
def __init__(
    self,
    image_path,
    annotation_path,
    transform,
    image_extensions=None,
    preload_maps=False,
    load_maps=True,  # NEW: Controls whether to load maps at all
    preload_images=False,
    ...
):
    ...
    self.load_maps = load_maps
```

**Updated map loading logic in `__getitem__`:**
```python
# Load pre-processed probability and threshold maps (only if enabled)
if self.load_maps:
    # First check RAM cache
    if image_filename in self.maps_cache:
        item["prob_map"] = self.maps_cache[image_filename]["prob_map"]
        item["thresh_map"] = self.maps_cache[image_filename]["thresh_map"]
    else:
        # Fallback to loading from disk
        maps_dir = self.image_path.parent / f"{self.image_path.name}_maps"
        # ... rest of lazy loading logic
```

#### 2. Updated Configuration Files

**`configs/data/base.yaml`:**
```yaml
datasets:
  train_dataset:
    load_maps: false  # Disable .npz map loading completely
  val_dataset:
    load_maps: false  # Disable .npz map loading completely
  test_dataset:
    load_maps: false  # Disable .npz map loading completely
  predict_dataset:
    load_maps: false  # Disable .npz map loading completely
```

**`configs/data/canonical.yaml`:**
```yaml
datasets:
  train_dataset:
    load_maps: false  # Disable .npz map loading completely
  val_dataset:
    load_maps: false  # Disable .npz map loading completely
  test_dataset:
    load_maps: false  # Disable .npz map loading completely
  predict_dataset:
    load_maps: false  # Disable .npz map loading completely
```

## Configuration Options Explained

| Parameter | Default | Purpose | When to Use |
|-----------|---------|---------|-------------|
| `load_maps` | `true` | Enable/disable ALL map loading (RAM + disk) | Set to `false` to completely disable maps |
| `preload_maps` | `false` | Preload all maps into RAM at init | Set to `true` for small datasets (<500 images) |

### Decision Matrix

```
load_maps=false, preload_maps=*     → No maps loaded at all ✅
load_maps=true,  preload_maps=false → Lazy load from disk on-demand
load_maps=true,  preload_maps=true  → Preload all maps into RAM at init
```

## Verification

### Before Changes
```log
✓ Using .npz maps (from cache or disk): 16/16 samples (100.0%)
```

### After Changes (with `load_maps: false`)
```log
⚠ Fallback to on-the-fly generation: 16/16 samples (100.0%)
```

The collate function will generate maps on-the-fly since no pre-loaded maps are available.

## Testing

Run training with the updated config:

```bash
HYDRA_FULL_ERROR=1 uv run python runners/train.py \
  exp_name=test_no_maps \
  trainer.max_epochs=1 \
  data=canonical \
  logger.wandb.enabled=false
```

**Expected behavior:**
- ✅ No `.npz` files loaded from disk
- ✅ Maps generated on-the-fly in collate function
- ✅ Training continues normally (slightly slower due to map generation)

## Performance Impact

| Configuration | Map Loading Time | Training Speed | RAM Usage |
|---------------|-----------------|----------------|-----------|
| `load_maps: true, preload_maps: true` | ~5-10s (init) | Fastest | High |
| `load_maps: true, preload_maps: false` | ~0.1s per batch | Fast | Low |
| `load_maps: false` | 0s | Moderate (map generation overhead) | Low |

**Recommendation:** Use `load_maps: false` if:
- You don't have pre-generated `.npz` files
- You want to test without map preprocessing
- You want to avoid any disk I/O for maps

## Notes

- The collate function (`DBCollateFN`) has fallback logic to generate maps on-the-fly if they're missing
- Setting `load_maps: false` does NOT break training - maps are just generated dynamically
- If you have pre-generated `.npz` files and want to use them, set `load_maps: true`
