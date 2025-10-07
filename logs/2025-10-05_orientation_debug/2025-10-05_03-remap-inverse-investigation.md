# 2025-10-05 — Remap Inverse Investigation

## Summary
- Re-ran the orientation utility, dataset, and integration pytest suites; all were initially green but training metrics remained unstable.
- Investigated the inference remapping path and identified incorrect inverse mappings for EXIF orientations 5 and 7.
- Confirmed the bug by demonstrating that `remap_polygons` followed by the current inverse mapping failed to recover raw coordinates for mirrored rotations.
- Updated `_ORIENTATION_INVERSE` in `ui/utils/inference/engine.py` so mirrored orientations (5 and 7) map to themselves.
- Extended `tests/ocr/utils/test_orientation.py` to cover orientations 5, 6, and 7 when remapping predictions back to raw space.

## Commands Executed
```bash
uv run pytest tests/ocr/utils/test_orientation.py
uv run pytest tests/ocr/datasets/test_exif_rotation.py
uv run pytest tests/integration/test_exif_orientation_smoke.py
HYDRA_FULL_ERROR=1 uv run python runners/train.py +trainer.limit_train_batches=1 +trainer.limit_val_batches=1 trainer.max_epochs=1 dataloaders.train_dataloader.batch_size=2 dataloaders.val_dataloader.batch_size=2 logger.wandb.enabled=false
```

## Evidence
- Added parameterized test that now fails on orientations 5 and 7 prior to the fix.
- Manual Python probes (`python - <<'PY'`) confirmed that mirrored orientations are self-inverse.
- Post-fix pytest suites pass (20/20 utils tests, 3/3 dataset tests).

## Next Steps
- Re-run a targeted training smoke once additional fixes land to verify recall stabilizes when mirrored EXIF samples are present.
- Audit downstream consumers (callbacks, visualizers) to ensure they are not caching the old inverse mapping.
- Monitor W&B overlays for samples tagged with orientation 5/7 to confirm alignment.
