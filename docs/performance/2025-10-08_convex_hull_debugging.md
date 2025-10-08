# 2025-10-08 – Convex Hull Degeneracy Hardening

## Overview
- **Issue**: `ConvexHull` calls in `ocr/metrics/box_types.py` crashed when polygons collapsed to vertical or horizontal lines during evaluation.
- **Root cause**: Albumentations transforms, integer rounding, and prediction geometry produced zero-width polygons that slipped past existing float-space checks.
- **Resolution**: Added integer-space guards in both the dataset loader and the CLEval metric helper, switched validation/test datasets to canonical images, and captured failure samples in a structured JSON log.

## Changes Implemented
- `configs/data/base.yaml`: Updated `val_dataset.image_path` and `test_dataset.image_path` to `images_val_canonical`.
- `ocr/datasets/base.py`: `_filter_degenerate_polygons` now drops polygons if `np.rint(...).ptp()` returns zero for either axis.
- `ocr/metrics/box_types.py`: `custom_MinAreaRect` converts input polygons to `float32`, checks integer spans before calling `ConvexHull`, and appends failures to `logs/convex_hull_failures.jsonl`.
- `ocr/datasets/base.py`: Degenerate polygon filtering now logs aggregated counts (`INFO` level) so long-running jobs surface data quality drift without overwhelming stdout.
- Polygon cache remains **disabled** (`polygon_cache.enabled: false` and `collate_fn.cache: null`) because prior experiments showed a significant slowdown.

## Validation
1. Rotated any existing hull log to `logs/convex_hull_failures.prev.jsonl`.
2. Ran a smoke training pass:
   ```bash
   uv run python runners/train.py trainer.limit_train_batches=10 trainer.limit_val_batches=5 trainer.max_epochs=1
   ```
3. Confirmed:
   - No runtime `ConvexHull` exceptions.
   - New log entries (if any) are filtered predictions labeled with `"message": "degenerate polygon after rounding"`.
   - Data pipeline now excludes degenerate polygons before batching.

## Inspecting Failure Logs
- Each log line is standalone JSON:
  ```bash
  jq '.points' logs/convex_hull_failures.jsonl
  ```
- Archive or delete `logs/convex_hull_failures.prev.jsonl` if historical samples are no longer needed.

## Next Steps
- Re-run the longer benchmark (`trainer.limit_train_batches=500`, `trainer.limit_val_batches=50`) to confirm metrics return to the ~0.74 hmean baseline.
- If further degenerate predictions appear, consider visualizing samples via the W&B run referenced above or add a lightweight debug plot in `ocr/datasets/debug/`.
- Once the guard is proven stable, regenerate any cached preprocessing artifacts so they inherit the stricter filtering.
