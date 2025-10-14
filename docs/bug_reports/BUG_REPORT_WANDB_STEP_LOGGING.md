# BUG REPORT: WandB Step Logging Non-Monotonic Warnings

## Issue Summary
The PerformanceProfilerCallback was logging metrics to WandB with non-monotonic step values during testing phase, causing WandB to reject the logs with warnings: "Tried to log to step X that is less than the current step Y. Steps must be monotonically increasing".

## Root Cause
The callback used a single monotonic step counter (`_last_wandb_step`) across both training/validation and testing phases. During testing, the trainer's global step remained high from training, but the callback attempted to log with lower step values calculated from the test batch indices, violating WandB's monotonic requirement.

## Impact
- WandB warnings during every training run with testing enabled
- Performance metrics from testing phase not logged to WandB
- Potential confusion in experiment tracking

## Reproduction Steps
1. Enable WandB logging (`logger.wandb.enabled=true`)
2. Run training with testing enabled (`trainer.limit_test_batches > 0`)
3. Observe WandB warnings during testing phase

## Solution Implemented
Modified `PerformanceProfilerCallback` to use separate monotonic step counters:
- `_last_wandb_step`: For training/validation phases
- `_last_test_wandb_step`: For testing phase (starts from 0)

Reset the test step counter at the start of testing to ensure clean monotonic progression.

## Files Modified
- `ocr/lightning_modules/callbacks/performance_profiler.py`:
  - Added `_last_test_wandb_step` counter
  - Modified `on_test_epoch_start()` to reset test step counter
  - Updated `on_test_batch_end()` and `on_test_epoch_end()` to use separate test counter

## Testing
- Verified training completes without WandB warnings
- Confirmed performance metrics are logged correctly during testing
- Validated preprocessing presets work without validation errors

## Status
✅ RESOLVED - No more WandB monotonic step warnings during testing
