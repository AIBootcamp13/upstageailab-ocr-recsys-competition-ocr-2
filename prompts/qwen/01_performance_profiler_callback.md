# Task: Create Performance Profiler Callback for PyTorch Lightning

## Context
- **Project:** Receipt OCR Text Detection (200k+ LOC)
- **Framework:** PyTorch Lightning 2.1+ with Hydra 1.3+
- **Purpose:** Phase 1.1 of Performance Optimization Plan - establish monitoring infrastructure
- **Code Style:** Follow `pyproject.toml` (ruff, mypy with type hints)

## Objective
Create a PyTorch Lightning callback that profiles validation performance to identify the PyClipper polygon processing bottleneck (currently 10x slower than training).

## Requirements

### Functional Requirements
1. **Track validation batch timing** - measure time per batch, per epoch
2. **Log PyClipper operation times** - if possible, instrument the collate function
3. **Monitor memory usage** - track GPU/CPU memory at validation start/end
4. **Report to WandB** - log all metrics to Weights & Biases
5. **Generate summary statistics** - mean, median, p95, p99 of batch times

### Non-Functional Requirements
1. **Minimal overhead** - profiling should not slow down training >5%
2. **Optional** - can be disabled via Hydra config
3. **Backward compatible** - works with existing training pipeline
4. **Type safe** - full type hints, passes mypy

## Input Files to Reference

### Read These Files First:
```
ocr/lightning_modules/callbacks/wandb_image_logging.py  # Example callback pattern
ocr/lightning_modules/callbacks/wandb_completion.py      # Example WandB integration
ocr/lightning_modules/ocr_pl.py                          # Lightning module structure
ocr/datasets/db_collate_fn.py                            # Where PyClipper is called
```

### Project Structure:
```
ocr/
  lightning_modules/
    callbacks/
      __init__.py                    # Import new callback here
      performance_profiler.py        # CREATE THIS FILE
```

## Output Files

### Create:
- `ocr/lightning_modules/callbacks/performance_profiler.py`

### Modify:
- `ocr/lightning_modules/callbacks/__init__.py` (add import)

## Implementation Details

### Class Structure
```python
"""Performance profiling callback for validation pipeline."""

import time
from typing import Any, Dict, Optional
import psutil
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import LightningModule, Trainer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class PerformanceProfilerCallback(Callback):
    """Profile validation performance to identify bottlenecks.

    Tracks:
    - Validation batch timing (per batch, per epoch)
    - Memory usage (GPU and CPU)
    - Summary statistics (mean, median, p95, p99)

    Logs all metrics to WandB if available.

    Args:
        enabled: Whether profiling is enabled
        log_interval: How often to log batch-level metrics (default: every 10 batches)
        profile_memory: Whether to profile memory usage
        verbose: Whether to print profiling info to console
    """

    def __init__(
        self,
        enabled: bool = True,
        log_interval: int = 10,
        profile_memory: bool = True,
        verbose: bool = False,
    ):
        super().__init__()
        self.enabled = enabled
        self.log_interval = log_interval
        self.profile_memory = profile_memory
        self.verbose = verbose

        # Tracking variables
        self.validation_batch_times: list[float] = []
        self.epoch_start_time: Optional[float] = None
        self.batch_start_time: Optional[float] = None

        # Memory tracking
        self.gpu_memory_allocated: list[float] = []
        self.cpu_memory_percent: list[float] = []

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Record validation epoch start time and memory."""
        if not self.enabled:
            return

        self.epoch_start_time = time.time()
        self.validation_batch_times = []

        if self.profile_memory:
            if torch.cuda.is_available():
                self.gpu_memory_allocated.append(
                    torch.cuda.memory_allocated() / 1024**3  # GB
                )
            self.cpu_memory_percent.append(psutil.virtual_memory().percent)

    def on_validation_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        """Record batch start time."""
        if not self.enabled:
            return
        self.batch_start_time = time.time()

    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any,
        batch: Any, batch_idx: int
    ) -> None:
        """Record batch end time and log metrics."""
        if not self.enabled or self.batch_start_time is None:
            return

        batch_time = time.time() - self.batch_start_time
        self.validation_batch_times.append(batch_time)

        # Log per-batch metrics at intervals
        if batch_idx % self.log_interval == 0:
            metrics = {
                "performance/val_batch_time": batch_time,
                "performance/val_batch_idx": batch_idx,
            }

            if self.verbose:
                print(f"Validation batch {batch_idx}: {batch_time:.3f}s")

            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log(metrics, step=trainer.global_step)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Compute and log summary statistics."""
        if not self.enabled or not self.validation_batch_times:
            return

        import numpy as np

        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        batch_times = np.array(self.validation_batch_times)

        metrics = {
            "performance/val_epoch_time": epoch_time,
            "performance/val_batch_mean": float(np.mean(batch_times)),
            "performance/val_batch_median": float(np.median(batch_times)),
            "performance/val_batch_p95": float(np.percentile(batch_times, 95)),
            "performance/val_batch_p99": float(np.percentile(batch_times, 99)),
            "performance/val_batch_std": float(np.std(batch_times)),
            "performance/val_num_batches": len(batch_times),
        }

        if self.profile_memory:
            if torch.cuda.is_available():
                metrics["performance/gpu_memory_gb"] = torch.cuda.memory_allocated() / 1024**3
                metrics["performance/gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            metrics["performance/cpu_memory_percent"] = psutil.virtual_memory().percent

        # Log to console
        if self.verbose:
            print("\n=== Validation Performance Summary ===")
            print(f"Epoch time: {epoch_time:.2f}s")
            print(f"Batch times: mean={metrics['performance/val_batch_mean']:.3f}s, "
                  f"median={metrics['performance/val_batch_median']:.3f}s, "
                  f"p95={metrics['performance/val_batch_p95']:.3f}s")
            if self.profile_memory and torch.cuda.is_available():
                print(f"GPU memory: {metrics['performance/gpu_memory_gb']:.2f}GB")
            print("=" * 40 + "\n")

        # Log to WandB
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(metrics, step=trainer.global_step)

        # Log to Lightning logger as well
        pl_module.log_dict(metrics, on_epoch=True, sync_dist=True)
```

### Integration Pattern

The callback should follow the same pattern as existing callbacks:

```python
# In ocr/lightning_modules/callbacks/__init__.py
from ocr.lightning_modules.callbacks.performance_profiler import PerformanceProfilerCallback

__all__ = [
    "PerformanceProfilerCallback",
    # ... existing callbacks
]
```

### Edge Cases to Handle
1. **WandB not available** - gracefully degrade, still log to Lightning
2. **CUDA not available** - skip GPU memory profiling
3. **Empty validation set** - handle zero batches
4. **Callback disabled** - minimal overhead, early return

## Validation

### Run These Commands:
```bash
# Type checking
uv run mypy ocr/lightning_modules/callbacks/performance_profiler.py

# Linting
uv run ruff check ocr/lightning_modules/callbacks/performance_profiler.py

# Format
uv run ruff format ocr/lightning_modules/callbacks/performance_profiler.py

# Import test
uv run python -c "from ocr.lightning_modules.callbacks import PerformanceProfilerCallback; print('Import successful')"
```

### Expected Behavior:
- ✅ All type checks pass
- ✅ No linting errors
- ✅ Import successful
- ✅ Callback can be instantiated: `PerformanceProfilerCallback(enabled=True)`

## Example Usage

After implementation, the callback will be used like this:

```python
# In training script
from ocr.lightning_modules.callbacks import PerformanceProfilerCallback

profiler = PerformanceProfilerCallback(
    enabled=True,
    log_interval=10,
    profile_memory=True,
    verbose=True,
)

trainer = pl.Trainer(
    callbacks=[profiler, ...],
    ...
)
```

Or via Hydra config (to be created separately):
```yaml
# configs/callbacks/performance_profiler.yaml
performance_profiler:
  _target_: ocr.lightning_modules.callbacks.PerformanceProfilerCallback
  enabled: true
  log_interval: 10
  profile_memory: true
  verbose: false
```

## Success Criteria

- [ ] File `ocr/lightning_modules/callbacks/performance_profiler.py` created
- [ ] All type hints present and mypy passes
- [ ] No ruff linting errors
- [ ] Callback can be imported successfully
- [ ] Follows existing callback patterns in the codebase
- [ ] Handles all edge cases (no WandB, no CUDA, disabled mode)

## Additional Notes

- **Dependencies:** Uses `psutil` for CPU memory (already in pyproject.toml)
- **WandB:** Optional dependency, graceful degradation
- **Performance:** Profiling overhead should be negligible (<100ms per epoch)
- **Future enhancement:** Could add collate_fn instrumentation via monkey-patching
