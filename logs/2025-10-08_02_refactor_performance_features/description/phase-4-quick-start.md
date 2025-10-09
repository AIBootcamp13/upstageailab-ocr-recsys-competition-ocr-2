# Phase 4 Quick Start: Performance Investigation

## Problem
- Phases 1-3 complete (preprocessing implemented)
- Expected 5-8x speedup NOT observed
- Performance gain is minimal

## Current Status
✅ Pre-processing works (maps generated and loaded)
✅ H-mean stable (no regression)
❌ Speed improvement not noticeable

## Investigation Steps

### 1. Verify Maps Are Being Loaded
Check if fallback is being triggered:
```python
# Add to DBCollateFN.__call__ in ocr/datasets/db_collate_fn.py
maps_loaded = sum(1 for item in batch if "prob_map" in item)
print(f"Loaded maps: {maps_loaded}/{len(batch)}")
```

### 2. Profile the Pipeline
```bash
# Time validation epoch
time uv run python runners/train.py trainer.limit_train_batches=0 trainer.limit_val_batches=100 trainer.max_epochs=1
```

### 3. Likely Bottlenecks
- I/O overhead from loading .npz files
- Transform pipeline still dominates time
- DataLoader configuration (num_workers)
- Tensor conversion overhead

## Quick Fixes to Try

### Fix 1: Check num_workers
Edit `configs/dataloaders/default.yaml`:
```yaml
val_dataloader:
  num_workers: 0  # Try single-threaded first
```

### Fix 2: Memory-mapped loading
Edit `ocr/datasets/base.py` line 178:
```python
maps_data = np.load(map_filename, mmap_mode='r')  # Memory-mapped
```

### Fix 3: RAM preloading (if dataset small)
Add to `OCRDataset.__init__`:
```python
self.maps_cache = {}
if hasattr(self, 'image_path'):
    maps_dir = self.image_path.parent / f"{self.image_path.name}_maps"
    if maps_dir.exists():
        for filename in self.anns.keys():
            npz_file = maps_dir / f"{Path(filename).stem}.npz"
            if npz_file.exists():
                self.maps_cache[filename] = dict(np.load(npz_file))
```

## Key Files
- `ocr/datasets/base.py:172-184` - Map loading
- `ocr/datasets/db_collate_fn.py:57-78` - Map usage
- `configs/dataloaders/default.yaml` - DataLoader settings

## Goal
Achieve 2-3x faster validation epochs minimum
