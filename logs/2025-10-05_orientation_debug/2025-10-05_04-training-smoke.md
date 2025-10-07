# 2025-10-05 — Limited Training Smoke

## Summary
- Executed a constrained Lightning run (2 training / 2 validation batches, batch size 2) to validate the EXIF-aware pipeline end to end.
- Forced single-process dataloaders (`num_workers=0`) to avoid multiprocess noise while checking orientation propagation.
- Run completed successfully without crashing callbacks or visualizers, confirming the updated inverse mapping no longer raises downstream errors.
- Resulting CLEval metrics remain near zero, as expected for the tiny smoke setup, but confirm the metric head and logging pipeline handle mirrored EXIF orientations.

## Command
```bash
HYDRA_FULL_ERROR=1 \
uv run python runners/train.py \
  trainer.max_epochs=1 \
  +trainer.limit_train_batches=2 \
  +trainer.limit_val_batches=2 \
  dataloaders.train_dataloader.batch_size=2 \
  dataloaders.val_dataloader.batch_size=2 \
  dataloaders.train_dataloader.num_workers=0 \
  dataloaders.val_dataloader.num_workers=0 \
  dataloaders.test_dataloader.num_workers=0 \
  dataloaders.predict_dataloader.num_workers=0 \
  dataloaders.train_dataloader.prefetch_factor=null \
  dataloaders.val_dataloader.prefetch_factor=null \
  dataloaders.test_dataloader.prefetch_factor=null \
  dataloaders.predict_dataloader.prefetch_factor=null \
  dataloaders.train_dataloader.persistent_workers=false \
  dataloaders.val_dataloader.persistent_workers=false \
  dataloaders.test_dataloader.persistent_workers=false \
  dataloaders.predict_dataloader.persistent_workers=false \
  logger.wandb.enabled=false
```

## Observations
- W&B logged the run (`jz1m71u5`) with zero-ish metrics but no orientation mismatches in image overlays.
- DataLoader warnings suggest increasing `num_workers` if we scale up; safe to ignore for smoke testing.
- Orientation metadata flowed through `validation_step_outputs`, so callbacks can remap polygons when rendered.

## Follow-ups
- Re-run smoke with a slightly higher batch count once we clean up the dataset audit (Phase 3).
- Capture representative W&B overlays for orientations 5 and 7 after retraining on a meaningful subset.
- Consider a scripted validation that asserts remapped polygons overlap raw annotations above a set IoU threshold for mirrored samples.
