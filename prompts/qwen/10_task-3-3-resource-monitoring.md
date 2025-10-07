# Qwen Task 3.3: System Resource Monitoring & Alerting

## Mission
Implement comprehensive system resource monitoring (GPU, CPU, I/O, disk) with intelligent alerting for performance anomalies. This is part of Phase 3 performance monitoring.

## Context
- **Project:** OCR model training pipeline optimization
- **Previous Work:** Phase 1 & 2 completed (monitoring infrastructure + polygon cache)
- **Current Branch:** `07_refactor/performance`
- **Your Task:** Independent task - implement resource monitoring only

## What You're Building
A monitoring system that:
1. **Tracks GPU utilization** (usage %, memory, temperature)
2. **Tracks CPU/memory** (per-process and system-wide)
3. **Tracks disk I/O** (read/write rates, dataset access patterns)
4. **Alerts on anomalies** (OOM risk, GPU underutilization, I/O bottlenecks)
5. **Exports time-series data** for visualization

## Implementation Requirements

### 1. Create `ResourceMonitorCallback`
**Location:** `ocr/callbacks/resource_monitor.py`

```python
import psutil
import GPUtil
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
        pass
```

### 2. GPU Monitoring

Use `GPUtil` or `pynvml` for GPU metrics:
```python
def get_gpu_metrics(self) -> dict:
    """
    Collect GPU metrics for all available GPUs.

    Returns:
        {
            "gpu_0_util": 85.3,  # Utilization %
            "gpu_0_memory_used_mb": 8192,
            "gpu_0_memory_total_mb": 16384,
            "gpu_0_memory_pct": 50.0,
            "gpu_0_temp_c": 72.0,
            "gpu_0_power_w": 180.5,
        }
    """
    gpus = GPUtil.getGPUs()
    metrics = {}
    for i, gpu in enumerate(gpus):
        metrics[f"gpu_{i}_util"] = gpu.load * 100
        metrics[f"gpu_{i}_memory_used_mb"] = gpu.memoryUsed
        metrics[f"gpu_{i}_memory_total_mb"] = gpu.memoryTotal
        metrics[f"gpu_{i}_memory_pct"] = gpu.memoryUtil * 100
        metrics[f"gpu_{i}_temp_c"] = gpu.temperature
    return metrics
```

### 3. CPU & Memory Monitoring

Use `psutil` for system metrics:
```python
def get_cpu_memory_metrics(self) -> dict:
    """
    Collect CPU and memory metrics.

    Returns:
        {
            "cpu_percent": 45.2,  # System-wide
            "cpu_process_percent": 180.5,  # This process (can be >100%)
            "memory_system_mb": 16384,
            "memory_available_mb": 8192,
            "memory_percent": 50.0,
            "memory_process_mb": 2048,
        }
    """
    process = psutil.Process()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "cpu_process_percent": process.cpu_percent(),
        "memory_system_mb": psutil.virtual_memory().total / 1024 / 1024,
        "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
        "memory_percent": psutil.virtual_memory().percent,
        "memory_process_mb": process.memory_info().rss / 1024 / 1024,
    }
```

### 4. Disk I/O Monitoring

Track data loading patterns:
```python
def get_io_metrics(self) -> dict:
    """
    Collect disk I/O metrics.

    Returns:
        {
            "disk_read_mb": 150.3,  # MB read since last check
            "disk_write_mb": 12.5,
            "io_read_rate_mbps": 75.2,  # MB/s
            "io_write_rate_mbps": 6.3,
            "io_wait_pct": 15.3,  # % time waiting on I/O
        }
    """
    process = psutil.Process()
    io_counters = process.io_counters()
    # Calculate rates based on time delta
    return metrics
```

### 5. Alerting System

Detect and log performance anomalies:
```python
def check_alerts(self, metrics: dict) -> list[str]:
    """
    Check for performance anomalies and generate alerts.

    Returns:
        List of alert messages
    """
    alerts = []

    # GPU underutilization
    if self.alert_gpu_underutilization:
        if metrics.get("gpu_0_util", 100) < self.gpu_util_threshold * 100:
            alerts.append(
                f"⚠️ GPU underutilized: {metrics['gpu_0_util']:.1f}% "
                f"(threshold: {self.gpu_util_threshold * 100}%)"
            )

    # Memory pressure
    if self.alert_memory_pressure:
        if metrics["memory_percent"] > self.memory_threshold * 100:
            alerts.append(
                f"⚠️ High memory usage: {metrics['memory_percent']:.1f}% "
                f"(threshold: {self.memory_threshold * 100}%)"
            )

    # I/O bottleneck
    if self.alert_io_bottleneck:
        if metrics.get("io_wait_pct", 0) > self.io_wait_threshold * 100:
            alerts.append(
                f"⚠️ I/O bottleneck: {metrics['io_wait_pct']:.1f}% wait time "
                f"(threshold: {self.io_wait_threshold * 100}%)"
            )

    # Temperature warning
    temp = metrics.get("gpu_0_temp_c", 0)
    if temp > 80:
        alerts.append(f"⚠️ GPU temperature high: {temp}°C")

    return alerts
```

### 6. Time-Series Export

Save metrics for visualization:
```python
def export_timeseries(self, metrics: dict, step: int):
    """
    Export metrics to CSV for later analysis.

    Format: resource_logs/epoch_N.csv
    Columns: step, timestamp, metric1, metric2, ...
    """
    csv_path = Path(self.timeseries_path) / f"epoch_{self.current_epoch}.csv"
    # Append metrics with timestamp
```

### 7. Integration Points

#### Config File: `configs/callbacks/resource_monitor.yaml`
```yaml
resource_monitor:
  _target_: ocr.callbacks.resource_monitor.ResourceMonitorCallback
  enabled: true
  log_interval: 10  # Log every 10 batches
  gpu_monitoring: true
  cpu_monitoring: true
  io_monitoring: true
  alert_gpu_underutilization: true
  alert_memory_pressure: true
  alert_io_bottleneck: true
  gpu_util_threshold: 0.5  # Alert if GPU <50%
  memory_threshold: 0.9  # Alert if memory >90%
  io_wait_threshold: 0.3  # Alert if >30% I/O wait
  export_timeseries: true
  timeseries_path: "resource_logs"
```

#### In `callbacks/__init__.py`:
```python
from ocr.callbacks.resource_monitor import ResourceMonitorCallback

__all__ = [
    # ... existing ...
    "ResourceMonitorCallback",
]
```

### 8. Output Format

#### Console Output (every N batches):
```
[Resources] Batch 50:
  GPU 0: 87.3% util, 8192/16384 MB (50%), 72°C
  CPU: 45.2% (process: 180.5%)
  Memory: 8192/16384 MB available (50%)
  I/O: Read 75.2 MB/s, Write 6.3 MB/s, Wait 5.3%

[Resources] ⚠️ ALERTS:
  - GPU underutilized: 42.3% (threshold: 50%)
  - High memory usage: 92.1% (threshold: 90%)
```

#### Time-Series CSV:
```csv
step,timestamp,gpu_0_util,gpu_0_memory_pct,cpu_percent,memory_percent,io_read_rate_mbps
0,2025-10-07T10:00:00,85.3,50.0,45.2,48.5,75.2
10,2025-10-07T10:00:05,87.1,51.2,46.3,49.1,73.8
...
```

## Files to Modify/Create

### Create New:
1. `ocr/callbacks/resource_monitor.py` - Main callback (~300 lines)
2. `configs/callbacks/resource_monitor.yaml` - Configuration
3. `tests/test_resource_monitor.py` - Unit tests
4. `scripts/plot_resources.py` - Visualization script (optional)

### Modify Existing:
1. `ocr/callbacks/__init__.py` - Add import
2. `.gitignore` - Add `resource_logs/` to ignore list
3. `requirements.txt` - Add `GPUtil` if not present

### Dependencies:
```txt
# Add to requirements.txt if missing
psutil>=5.9.0
GPUtil>=1.4.0
```

## Testing Requirements

### Unit Tests (`tests/test_resource_monitor.py`):
```python
def test_gpu_metrics_collection():
    # Test GPU metric gathering (mock GPUtil)
    pass

def test_cpu_memory_metrics():
    # Test CPU/memory metrics
    pass

def test_alert_thresholds():
    # Test alert triggering logic
    pass

def test_timeseries_export():
    # Test CSV export format
    pass

def test_disabled_monitoring():
    # Test no overhead when disabled
    pass
```

### Integration Test:
```bash
# Monitor resources for 3 epochs
python ocr/train.py \
    experiment=synthetic_debug \
    trainer.max_epochs=3 \
    callbacks.resource_monitor.enabled=true \
    callbacks.resource_monitor.log_interval=5
```

**Expected:**
- Resource metrics logged every 5 batches
- Alerts triggered if thresholds exceeded
- `resource_logs/epoch_*.csv` files created
- No crashes or performance degradation

### Visualization Test (Optional):
```python
# scripts/plot_resources.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("resource_logs/epoch_1.csv")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot GPU util, memory, CPU, I/O over time
axes[0, 0].plot(df["step"], df["gpu_0_util"])
axes[0, 0].set_title("GPU Utilization")
# ... more plots

plt.savefig("resource_timeline.png")
```

## Success Criteria
- [ ] Callback tracks GPU, CPU, memory, I/O metrics
- [ ] Alerts trigger correctly based on thresholds
- [ ] Time-series data exported to CSV
- [ ] Configurable monitoring intervals and thresholds
- [ ] No performance impact when disabled
- [ ] Minimal overhead when enabled (<1% slowdown)
- [ ] Unit tests pass with >75% coverage
- [ ] Integration test shows expected metrics/alerts
- [ ] Code passes linting and type checking

## References

### Python Libraries:
- `psutil` docs: https://psutil.readthedocs.io/
- `GPUtil` docs: https://github.com/anderskm/gputil
- Alternative: `pynvml` for NVIDIA GPUs

### Existing Code:
- `ocr/callbacks/performance_monitor.py` - Memory tracking patterns
- `ocr/callbacks/polygon_cache_logger.py` - Callback structure

### NVIDIA Tools:
- `nvidia-smi` command for validation
- `nvtop` for interactive monitoring

## Important Notes

### Independence:
- This task is **independent** of Tasks 3.1 and 3.2
- Don't modify throughput or profiler code
- Focus only on system resource monitoring

### Performance Overhead:
- Monitoring should add <1% overhead
- Use `psutil` with `interval=0` for instant snapshots
- Cache `psutil.Process()` object (don't recreate each call)

### Cross-Platform Support:
- `GPUtil` may not work on CPU-only machines (handle gracefully)
- I/O metrics differ on Windows vs Linux (use platform checks)
- Temperature monitoring may not be available everywhere

### Error Handling:
- GPU monitoring can fail if no GPU or driver issues
- Wrap GPU calls in try/except, log warnings
- Continue training even if monitoring fails

## Advanced Features (Optional)

If time permits, add:
1. **Adaptive alerting:** Learn baseline metrics, alert on deviations
2. **Multi-GPU support:** Track all GPUs independently
3. **Network I/O:** Monitor if using distributed training
4. **Slack/email alerts:** Send notifications for critical issues
5. **Dashboard:** Real-time web dashboard (Streamlit/Gradio)

## Delivery Checklist
Before marking complete:
1. All files created/modified as specified
2. Unit tests written and passing
3. Integration test successful with alerts triggered
4. Linting and mypy clean
5. Resource metrics logged correctly
6. Time-series CSV files generated
7. Documentation in docstrings complete
8. Handles missing GPU gracefully

## Questions to Resolve Yourself
- Exact logging interval (balance verbosity vs insight)
- Which metrics to log vs which to only use for alerts
- CSV format and column names
- Default alert thresholds

## Questions to Ask if Blocked
- GPUtil not detecting GPUs
- psutil I/O counters not available on system
- Integration with Lightning trainer unclear
- Performance overhead concerns

---

## Start Here
1. Install and test `GPUtil` and `psutil` libraries
2. Write standalone scripts to test metric collection
3. Create `ocr/callbacks/resource_monitor.py` skeleton
4. Implement GPU monitoring (mock if no GPU available)
5. Implement CPU/memory monitoring
6. Implement I/O monitoring
7. Add alerting logic
8. Add time-series export
9. Write tests
10. Run integration test
11. Polish and document

**Estimated Time:** 5-7 hours
**Priority:** Medium-High (important for production monitoring)

## Quick Test Commands
```bash
# Test GPU monitoring
python -c "import GPUtil; print(GPUtil.getGPUs())"

# Test psutil
python -c "import psutil; print(psutil.virtual_memory())"

# Run with monitoring
python ocr/train.py \
    experiment=synthetic_debug \
    trainer.max_epochs=2 \
    callbacks.resource_monitor.enabled=true \
    callbacks.resource_monitor.log_interval=5

# Verify CSV output
ls resource_logs/
head resource_logs/epoch_1.csv
```

## Alert Testing Tips
To test alerts, temporarily lower thresholds:
```bash
python ocr/train.py \
    experiment=synthetic_debug \
    callbacks.resource_monitor.enabled=true \
    callbacks.resource_monitor.gpu_util_threshold=0.99 \
    callbacks.resource_monitor.memory_threshold=0.01
```

This should trigger alerts immediately, verifying the system works.
