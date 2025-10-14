"""Performance profiling callback for validation pipeline."""

import time
from typing import Any

import psutil
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

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
        self.epoch_start_time: float | None = None
        self.batch_start_time: float | None = None

        # Memory tracking
        self.gpu_memory_allocated: list[float] = []
        self.cpu_memory_percent: list[float] = []

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
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

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Record test epoch start time and memory."""
        # Use the same logic as validation
        self.on_validation_epoch_start(trainer, pl_module)

    def on_validation_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Record batch start time."""
        if not self.enabled:
            return
        self.batch_start_time = time.time()

    def on_test_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Record test batch start time."""
        # Use the same logic as validation
        self.on_validation_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
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

            # Use trainer.fit_loop.epoch_loop.total_batch_idx for monotonic step
            # This ensures step is always increasing, even during validation
            # Fallback to global_step if total_batch_idx is invalid, and ensure >= 0
            total_batch_idx = getattr(trainer.fit_loop.epoch_loop, "total_batch_idx", trainer.global_step)
            step = max(0, total_batch_idx if total_batch_idx >= 0 else trainer.global_step)
            wandb.log(metrics, step=step)  # type: ignore

    def on_test_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Record test batch end time and log metrics."""
        # Use the same logic as validation
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
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
            print(
                f"Batch times: mean={metrics['performance/val_batch_mean']:.3f}s, "
                f"median={metrics['performance/val_batch_median']:.3f}s, "
                f"p95={metrics['performance/val_batch_p95']:.3f}s"
            )
            if self.profile_memory and torch.cuda.is_available():
                print(f"GPU memory: {metrics['performance/gpu_memory_gb']:.2f}GB")
            print("=" * 40 + "\n")

        # Log to WandB with monotonic step
        if WANDB_AVAILABLE and wandb.run is not None:  # type: ignore
            total_batch_idx = getattr(trainer.fit_loop.epoch_loop, "total_batch_idx", trainer.global_step)
            step = max(0, total_batch_idx if total_batch_idx >= 0 else trainer.global_step)
            wandb.log(metrics, step=step)  # type: ignore

        # Log to Lightning logger as well
        pl_module.log_dict(metrics, on_epoch=True, sync_dist=True)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log test summary statistics."""
        # Use the same logic as validation
        self.on_validation_epoch_end(trainer, pl_module)
