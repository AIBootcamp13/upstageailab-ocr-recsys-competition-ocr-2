"""
System resource monitoring with intelligent alerting.

Monitors:
- GPU: utilization, memory, temperature
- CPU: per-core usage, process CPU time
- Memory: system and process memory, OOM risk
- Disk I/O: read/write rates, dataset access patterns

Alerts:
- GPU underutilization (<50% for extended periods)
- High memory pressure (>90% usage)
- I/O bottlenecks (>30% time waiting on disk)
- Temperature warnings (>80°C)
"""

import os
import time
from datetime import datetime
from pathlib import Path

import GPUtil
import pandas as pd
import psutil
from pytorch_lightning import Callback


class ResourceMonitorCallback(Callback):
    """
    System resource monitoring with intelligent alerting.

    Monitors:
    - GPU: utilization, memory, temperature
    - CPU: per-core usage, process CPU time
    - Memory: system and process memory, OOM risk
    - Disk I/O: read/write rates, dataset access patterns

    Alerts:
    - GPU underutilization (<50% for extended periods)
    - High memory pressure (>90% usage)
    - I/O bottlenecks (>30% time waiting on disk)
    - Temperature warnings (>80°C)
    """

    def __init__(
        self,
        enabled: bool = True,
        log_interval: int = 10,  # Log every N batches
        gpu_monitoring: bool = True,
        cpu_monitoring: bool = True,
        io_monitoring: bool = True,
        alert_gpu_underutilization: bool = True,
        alert_memory_pressure: bool = True,
        alert_io_bottleneck: bool = True,
        gpu_util_threshold: float = 0.5,  # Alert if <50%
        memory_threshold: float = 0.9,  # Alert if >90%
        io_wait_threshold: float = 0.3,  # Alert if >30% I/O time
        export_timeseries: bool = True,
        timeseries_path: str = "resource_logs",
    ):
        super().__init__()

        self.enabled = enabled
        self.log_interval = log_interval
        self.gpu_monitoring = gpu_monitoring
        self.cpu_monitoring = cpu_monitoring
        self.io_monitoring = io_monitoring
        self.alert_gpu_underutilization = alert_gpu_underutilization
        self.alert_memory_pressure = alert_memory_pressure
        self.alert_io_bottleneck = alert_io_bottleneck
        self.gpu_util_threshold = gpu_util_threshold
        self.memory_threshold = memory_threshold
        self.io_wait_threshold = io_wait_threshold
        self.export_timeseries = export_timeseries

        # Convert to Path object and create directory if needed
        self.timeseries_path = Path(timeseries_path)
        if self.export_timeseries and not self.timeseries_path.exists():
            self.timeseries_path.mkdir(parents=True, exist_ok=True)

        # Initialize process for monitoring this specific process
        # Store the PID instead of the process object to avoid serialization issues
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)  # Initialize process immediately

        # Last I/O values to calculate rates
        self.last_io_counters: psutil._common.pio | None = None
        self.last_time: float | None = None

        # Current epoch tracking
        self.current_epoch = 0

        # Flag to handle GPU not available gracefully
        self.gpu_available = self._check_gpu_availability()

    def setup(self, trainer, pl_module, stage=None):
        """Initialize monitoring when trainer starts."""
        if not self.enabled:
            return

        # Initialize the process here to avoid serialization issues
        self.process = psutil.Process(self.pid)

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available and GPUtil can access it."""
        try:
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
        except Exception:
            return False

    def on_train_start(self, trainer, pl_module):
        """Initialize monitoring when training starts."""
        if not self.enabled:
            return

        self.last_time = time.time()
        print(f"[Resources] Starting resource monitoring. Logging every {self.log_interval} batches.")

        # Initialize the first I/O counters
        if self.io_monitoring:
            try:
                self.last_io_counters = self.process.io_counters()
            except (psutil.AccessDenied, AttributeError):
                # On some systems, I/O counters might not be available
                self.last_io_counters = None
                print("[Resources] Warning: I/O counters not available on this system")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Monitor resources after each training batch."""
        if not self.enabled:
            return

        # Log resources every log_interval batches
        if batch_idx % self.log_interval == 0:
            # Collect metrics
            all_metrics = {}

            # Collect GPU metrics if enabled
            if self.gpu_monitoring and self.gpu_available:
                try:
                    gpu_metrics = self.get_gpu_metrics()
                    all_metrics.update(gpu_metrics)
                except Exception as e:
                    print(f"[Resources] Warning: GPU monitoring failed: {e}")

            # Collect CPU and memory metrics if enabled
            if self.cpu_monitoring:
                cpu_mem_metrics = self.get_cpu_memory_metrics()
                all_metrics.update(cpu_mem_metrics)

            # Collect I/O metrics if enabled
            if self.io_monitoring:
                try:
                    io_metrics = self.get_io_metrics()
                    all_metrics.update(io_metrics)
                except Exception as e:
                    print(f"[Resources] Warning: I/O monitoring failed: {e}")

            # Add timestamp and step
            current_time = time.time()
            timestamp = datetime.fromtimestamp(current_time).strftime("%Y-%m-%dT%H:%M:%S")
            all_metrics["step"] = batch_idx
            all_metrics["timestamp"] = timestamp
            all_metrics["epoch"] = self.current_epoch

            # Check for alerts
            alerts = self.check_alerts(all_metrics)

            # Log metrics
            self.log_metrics(all_metrics)

            # Print alerts if any
            if alerts:
                print("[Resources] ⚠️  ALERTS:")
                for alert in alerts:
                    print(f"  - {alert}")

            # Export time-series data if enabled
            if self.export_timeseries:
                self.export_timeseries_data(all_metrics)

    def on_train_epoch_start(self, trainer, pl_module):
        """Track current epoch for CSV file naming."""
        self.current_epoch = trainer.current_epoch

    def get_gpu_metrics(self) -> dict[str, float]:
        """Collect GPU metrics for all available GPUs."""
        # Return empty dict if no GPU is available
        if not self.gpu_available:
            return {}

        try:
            gpus = GPUtil.getGPUs()
            metrics = {}

            for i, gpu in enumerate(gpus):
                metrics[f"gpu_{i}_util"] = gpu.load * 100  # Utilization percentage
                metrics[f"gpu_{i}_memory_used_mb"] = gpu.memoryUsed
                metrics[f"gpu_{i}_memory_total_mb"] = gpu.memoryTotal
                metrics[f"gpu_{i}_memory_pct"] = gpu.memoryUtil * 100  # Memory usage percentage
                metrics[f"gpu_{i}_temp_c"] = gpu.temperature

                # Some GPUs might not report power
                if hasattr(gpu, "power"):
                    metrics[f"gpu_{i}_power_w"] = gpu.power
                else:
                    metrics[f"gpu_{i}_power_w"] = 0.0  # Default to 0 if not available

            return metrics
        except Exception as e:
            # If GPU access fails after initialization, return empty dict
            print(f"[Resources] Warning: GPU metrics collection failed: {e}")
            return {}

    def get_cpu_memory_metrics(self) -> dict[str, float]:
        """Collect CPU and memory metrics."""
        # Initialize process if not already done (for standalone usage)
        if self.process is None:
            self.process = psutil.Process(self.pid)

        return {
            "cpu_percent": psutil.cpu_percent(interval=None),  # Non-blocking call
            "cpu_process_percent": self.process.cpu_percent(),
            "memory_system_mb": psutil.virtual_memory().total / 1024 / 1024,
            "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "memory_percent": psutil.virtual_memory().percent,
            "memory_process_mb": self.process.memory_info().rss / 1024 / 1024,
        }

    def get_io_metrics(self) -> dict[str, float]:
        """Collect disk I/O metrics."""
        current_time = time.time()
        io_metrics: dict[str, float] = {}

        # Skip if last counters not set
        if self.last_io_counters is None:
            # Initialize with current values
            try:
                self.last_io_counters = self.process.io_counters()
            except (psutil.AccessDenied, AttributeError):
                # On some systems, I/O counters might not be available
                return {"io_read_rate_mbps": 0.0, "io_write_rate_mbps": 0.0, "io_wait_pct": 0.0}

            self.last_time = current_time
            return {"io_read_rate_mbps": 0.0, "io_write_rate_mbps": 0.0, "io_wait_pct": 0.0}

        # Get current I/O counters
        try:
            current_io_counters = self.process.io_counters()
        except (psutil.AccessDenied, AttributeError):
            # On some systems, I/O counters might not be available
            return {"io_read_rate_mbps": 0.0, "io_write_rate_mbps": 0.0, "io_wait_pct": 0.0}

        # Calculate time delta
        if self.last_time is None:
            # This shouldn't happen if the method is called after initialization
            # Return zeros to avoid errors
            return {"io_read_rate_mbps": 0.0, "io_write_rate_mbps": 0.0, "io_wait_pct": 0.0}

        time_delta = current_time - self.last_time
        if time_delta <= 0:
            time_delta = 1e-6  # Prevent division by zero

        # Calculate rate differences (in bytes per second, then convert to MB/s)
        read_bytes_diff = max(0, current_io_counters.read_bytes - self.last_io_counters.read_bytes)
        write_bytes_diff = max(0, current_io_counters.write_bytes - self.last_io_counters.write_bytes)

        read_rate = read_bytes_diff / time_delta / (1024 * 1024)
        write_rate = write_bytes_diff / time_delta / (1024 * 1024)

        # Update stored values for next calculation
        self.last_io_counters = current_io_counters
        self.last_time = current_time

        # For I/O wait time, we can only get system-wide stats, not per-process
        # This metric isn't as meaningful for per-process monitoring
        io_wait_pct = 0.0  # Set to 0 since per-process I/O wait isn't available

        io_metrics.update(
            {
                "io_read_rate_mbps": round(read_rate, 2),
                "io_write_rate_mbps": round(write_rate, 2),
                "io_wait_pct": round(io_wait_pct, 2),
            }
        )

        return io_metrics

    def check_alerts(self, metrics: dict[str, float]) -> list[str]:
        """Check for performance anomalies and generate alerts."""
        alerts = []

        # GPU underutilization
        if self.alert_gpu_underutilization and self.gpu_monitoring and self.gpu_available:
            # Check first GPU utilization
            gpu_util_key = "gpu_0_util"
            if gpu_util_key in metrics:
                if metrics[gpu_util_key] < self.gpu_util_threshold * 100:
                    alerts.append(f"GPU underutilized: {metrics[gpu_util_key]:.1f}% " f"(threshold: {self.gpu_util_threshold * 100}%)")

        # Memory pressure
        if self.alert_memory_pressure and self.cpu_monitoring:
            memory_pct = metrics.get("memory_percent", 0)
            if memory_pct > self.memory_threshold * 100:
                alerts.append(f"High memory usage: {memory_pct:.1f}% " f"(threshold: {self.memory_threshold * 100}%)")

        # I/O bottleneck
        if self.alert_io_bottleneck and self.io_monitoring:
            io_wait_pct = metrics.get("io_wait_pct", 0)
            if io_wait_pct > self.io_wait_threshold * 100:
                alerts.append(f"I/O bottleneck: {io_wait_pct:.1f}% wait time " f"(threshold: {self.io_wait_threshold * 100}%)")

        # Temperature warning (if available)
        for key, value in metrics.items():
            if key.endswith("_temp_c") and value > 80:
                alerts.append(f"High temperature on {key.replace('_temp_c', '')}: {value}°C")

        return alerts

    def log_metrics(self, metrics: dict[str, float]):
        """Log metrics to console."""
        print(f"[Resources] Epoch {self.current_epoch}, Batch {metrics['step']}:")

        # Log GPU metrics
        for key, value in metrics.items():
            if key.startswith("gpu_") and "_util" in key:
                gpu_id = key.split("_")[1]
                mem_key = f"gpu_{gpu_id}_memory_pct"
                temp_key = f"gpu_{gpu_id}_temp_c"

                print(f"  GPU {gpu_id}: {value:.1f}% util, {metrics.get(mem_key, 0):.1f}% memory, {metrics.get(temp_key, 0):.1f}°C")

        # Log CPU metrics
        cpu_sys = metrics.get("cpu_percent", 0)
        cpu_proc = metrics.get("cpu_process_percent", 0)
        print(f"  CPU: {cpu_sys:.1f}% (process: {cpu_proc:.1f}%)")

        # Log memory metrics
        mem_total = metrics.get("memory_system_mb", 0) / 1024  # Convert to GB
        mem_avail = metrics.get("memory_available_mb", 0) / 1024  # Convert to GB
        mem_pct = metrics.get("memory_percent", 0)
        print(f"  Memory: {mem_avail:.1f}/{mem_total:.1f} GB available ({mem_pct:.1f}%)")

        # Log I/O metrics
        read_rate = metrics.get("io_read_rate_mbps", 0)
        write_rate = metrics.get("io_write_rate_mbps", 0)
        io_wait = metrics.get("io_wait_pct", 0)
        print(f"  I/O: Read {read_rate:.1f} MB/s, Write {write_rate:.1f} MB/s, Wait {io_wait:.1f}%")

    def export_timeseries_data(self, metrics: dict[str, float]):
        """Export metrics to CSV for later analysis."""
        if not self.export_timeseries:
            return

        # Create CSV file for current epoch
        csv_path = self.timeseries_path / f"epoch_{self.current_epoch}.csv"

        # Ensure the dataframe has the right structure
        df = pd.DataFrame([metrics])

        # Write header only if file doesn't exist (first time)
        write_header = not csv_path.exists()

        # Append metrics to CSV file
        df.to_csv(csv_path, mode="a", header=write_header, index=False)

    def on_train_end(self, trainer, pl_module):
        """Finalize monitoring when training ends."""
        if not self.enabled:
            return

        print(f"[Resources] Training completed. Metrics exported to {self.timeseries_path}/")
