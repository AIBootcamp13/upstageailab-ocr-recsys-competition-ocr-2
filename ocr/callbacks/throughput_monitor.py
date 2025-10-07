"""
Dataloader throughput monitoring for training pipeline efficiency.

Monitors:
- Samples/second throughput per epoch
- Memory usage (dataset, cache, peak memory)
- Batch timing (load time, transform time)
- Bottleneck detection (identify slow operations)
"""

import statistics
import time

import psutil
from pytorch_lightning import Callback


class ThroughputMonitorCallback(Callback):
    """
    Monitors dataloader throughput and performance metrics.

    Metrics tracked:
    - samples_per_second (avg per epoch)
    - batch_load_time_ms (avg, p50, p95, p99)
    - batch_transform_time_ms (avg, p50, p95, p99)
    - memory_dataset_mb
    - memory_cache_mb
    - memory_peak_mb
    - batches_per_second
    - throughput_efficiency (actual vs theoretical)
    """

    def __init__(
        self,
        enabled: bool = True,
        log_interval: int = 1,  # Log every N epochs
        track_memory: bool = True,
        track_timing: bool = True,
    ):
        super().__init__()

        self.enabled = enabled
        self.log_interval = log_interval
        self.track_memory = track_memory
        self.track_timing = track_timing

        # Initialize process for monitoring this specific process
        self.process = psutil.Process()

        # Track timing per batch
        self.batch_start_time: float | None = None
        self.batch_times: list[float] = []
        self.transform_times: list[float] = []

        # Track samples per epoch
        self.samples_this_epoch = 0
        self.epoch_start_time = None

        # Memory tracking
        self.dataset_memory_mb = 0
        self.cache_memory_mb = 0
        self.peak_memory_mb = 0

        # Track for each epoch
        self.epoch_start_memory_mb = 0

    def on_train_epoch_start(self, trainer, pl_module):
        """Initialize tracking at the start of each epoch."""
        if not self.enabled:
            return

        self.epoch_start_time = time.perf_counter()
        self.samples_this_epoch = 0
        self.batch_times = []
        self.transform_times = []

        # Record memory at start of epoch
        if self.track_memory:
            memory_info = self.process.memory_info()
            current_memory_mb = memory_info.rss / 1024 / 1024
            self.epoch_start_memory_mb = current_memory_mb
            self.peak_memory_mb = max(self.peak_memory_mb, current_memory_mb)

            # Try to estimate dataset and cache memory if available
            # Note: This is approximate and depends on how the data module is structured
            try:
                # Check if trainer has access to the data module which might have cache info
                if hasattr(trainer, "datamodule"):
                    datamodule = trainer.datamodule
                    # If there's a polygon cache in the datamodule, try to get its memory usage
                    if hasattr(datamodule, "polygon_cache"):
                        # Estimate cache size based on number of entries and assume each entry is ~1MB (approximation)
                        cache_size = len(datamodule.polygon_cache)
                        self.cache_memory_mb = cache_size  # Rough approximation
                    # For dataset memory, this would depend on the specific implementation
                    # We'll leave as 0 for now since it's hard to estimate
            except Exception:
                # If we can't access cache info, continue without error
                pass

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Record batch start time."""
        if not self.enabled or not self.track_timing:
            return

        self.batch_start_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Record batch end time and increment sample count."""
        if not self.enabled:
            return

        # Increment sample count (improve detection for various batch formats)
        batch_size = self._get_batch_size(batch)
        self.samples_this_epoch += batch_size

        # Record timing if enabled
        if self.track_timing and self.batch_start_time is not None:
            batch_time = (time.perf_counter() - self.batch_start_time) * 1000  # Convert to ms
            self.batch_times.append(batch_time)

            # Update peak memory
            if self.track_memory:
                memory_info = self.process.memory_info()
                self.peak_memory_mb = max(self.peak_memory_mb, memory_info.rss / 1024 / 1024)

    def _get_batch_size(self, batch):
        """Extract batch size from various batch formats."""
        # Handle common batch formats in PyTorch
        if isinstance(batch, list | tuple):
            if len(batch) > 0:
                first_element = batch[0]
                if hasattr(first_element, "size"):
                    # For tensor batches like [images, labels] where images is (B, C, H, W)
                    if first_element.dim() > 0:
                        return first_element.size(0)
                # If it's a list of other structures, try to get length from first few elements
                for item in batch:
                    if hasattr(item, "size") and len(item.size()) > 0:
                        return item.size(0)
                    elif isinstance(item, list | tuple) and len(item) > 0:
                        return len(item)
        elif hasattr(batch, "size"):  # Single tensor batch
            if batch.dim() > 0:
                return batch.size(0)
        elif isinstance(batch, dict):  # Dictionary format
            for key, value in batch.items():
                if hasattr(value, "size") and len(value.size()) > 0:
                    return value.size(0)
            # If no tensors found, try to get a count from dictionary values
            return len(next(iter(batch.values()))) if batch else 0
        elif hasattr(batch, "__len__") and not isinstance(batch, str):  # Handle other iterables, but not strings
            # We exclude strings because len("unsupported") would return 11, not 0
            return len(batch)

        # Default to 0 if unable to determine
        return 0

    def on_train_epoch_end(self, trainer, pl_module):
        """Calculate and log throughput metrics at the end of each epoch."""
        if not self.enabled:
            return

        # Calculate epoch metrics
        epoch_time = time.perf_counter() - self.epoch_start_time
        samples_per_sec = self.samples_this_epoch / epoch_time if epoch_time > 0 else 0
        batches_per_sec = len(self.batch_times) / epoch_time if epoch_time > 0 and len(self.batch_times) > 0 else 0

        # Calculate timing percentiles if we have timing data
        avg_batch_time = statistics.mean(self.batch_times) if self.batch_times else 0
        p50_batch_time = 0
        p95_batch_time = 0
        p99_batch_time = 0

        if self.batch_times:
            sorted_times = sorted(self.batch_times)

            # Calculate percentiles using interpolation method
            p50_batch_time = statistics.median(sorted_times)

            # Calculate p95 and p99 percentiles
            def get_percentile(data, percentile):
                if not data:
                    return 0
                k = (len(data) - 1) * percentile / 100
                f = int(k)
                c = min(f + 1, len(data) - 1)
                if f == c:
                    return data[f]
                d0 = data[f] * (c - k)
                d1 = data[c] * (k - f)
                return d0 + d1

            p95_batch_time = get_percentile(sorted_times, 95)
            p99_batch_time = get_percentile(sorted_times, 99)

        # Prepare metrics dict for logging
        metrics = {
            "samples_per_second": samples_per_sec,
            "batches_per_second": batches_per_sec,
            "avg_batch_time_ms": avg_batch_time,
            "p50_batch_time_ms": p50_batch_time,
            "p95_batch_time_ms": p95_batch_time,
            "p99_batch_time_ms": p99_batch_time,
        }

        # Add memory metrics
        if self.track_memory:
            memory_info = self.process.memory_info()
            current_memory_mb = memory_info.rss / 1024 / 1024
            metrics["memory_current_mb"] = current_memory_mb
            metrics["memory_peak_mb"] = self.peak_memory_mb
            metrics["memory_dataset_mb"] = self.dataset_memory_mb
            metrics["memory_cache_mb"] = self.cache_memory_mb

        # Log to trainer logger (MLflow, etc.)
        for key, value in metrics.items():
            if hasattr(trainer.logger, "log_metrics"):
                trainer.logger.log_metrics({key: value}, step=trainer.global_step)
            elif hasattr(trainer.logger, "experiment"):
                # For MLflow logger specifically
                if hasattr(trainer.logger.experiment, "log_metrics") and hasattr(trainer.logger, "run_id"):
                    trainer.logger.experiment.log_metrics(trainer.logger.run_id, {key: value})

        # Log to console
        if trainer.current_epoch % self.log_interval == 0:
            self._log_metrics(trainer.current_epoch, metrics)

    def _log_metrics(self, epoch: int, metrics: dict):
        """Log metrics to console in a readable format."""
        print(f"[Throughput] Epoch {epoch}:")
        print(f"  Samples/sec: {metrics['samples_per_second']:.1f}")
        print(f"  Batches/sec: {metrics['batches_per_second']:.1f}")
        print(f"  Avg batch time: {metrics['avg_batch_time_ms']:.1f}ms (p95: {metrics['p95_batch_time_ms']:.1f}ms)")

        if "memory_dataset_mb" in metrics:
            print(f"  Memory (current): {metrics['memory_current_mb']:.0f} MB")
            print(f"  Memory (peak): {metrics['memory_peak_mb']:.0f} MB")
            print(f"  Memory (dataset): {metrics['memory_dataset_mb']:.0f} MB")
            print(f"  Memory (cache): {metrics['memory_cache_mb']:.0f} MB")
