# Performance Monitoring Callbacks Usage Guide

This guide explains how to use the performance monitoring callbacks for comprehensive training pipeline analysis and optimization.

## Overview

The OCR training pipeline includes three specialized callbacks for performance monitoring:

- **ThroughputMonitorCallback**: Monitors dataloader efficiency and training throughput
- **ProfilerCallback**: Provides detailed PyTorch profiling with Chrome trace export
- **ResourceMonitorCallback**: Tracks system resources (GPU, CPU, memory, I/O) with intelligent alerting

## Prerequisites

- PyTorch Lightning training pipeline
- Hydra configuration system
- WandB logging (optional but recommended)
- GPU access (recommended for GPU monitoring)

## ThroughputMonitorCallback

### Purpose
Monitors dataloader throughput, memory usage, and batch processing efficiency to identify performance bottlenecks in the training pipeline.

### Metrics Tracked
- **samples_per_second**: Average training throughput per epoch
- **batch_load_time_ms**: Time spent loading batches (avg, p50, p95, p99)
- **batch_transform_time_ms**: Time spent in data transformations
- **memory_dataset_mb**: Dataset memory usage
- **memory_cache_mb**: Cache memory usage
- **memory_peak_mb**: Peak memory consumption
- **batches_per_second**: Batch processing rate
- **throughput_efficiency**: Actual vs theoretical throughput

### Enabling the Callback

```bash
uv run python runners/train.py \
  data=canonical \
  callbacks=throughput_monitor \
  callbacks.throughput_monitor.enabled=true \
  callbacks.throughput_monitor.log_interval=1 \
  project_name=OCR_Training \
  exp_name=throughput_monitoring_test
```

### Configuration Options

```yaml
throughput_monitor:
  _target_: ocr.callbacks.throughput_monitor.ThroughputMonitorCallback
  enabled: true                    # Enable/disable monitoring
  log_interval: 1                  # Log every N epochs
  track_memory: true              # Monitor memory usage
  track_timing: true              # Monitor batch timing
```

### Output and Interpretation

The callback logs metrics to WandB and console:

```
ThroughputMonitor: Epoch 1 - samples/sec: 45.2, memory_peak: 2.1GB
ThroughputMonitor: Batch timing - load: 45ms (p95), transform: 12ms (p95)
```

**Interpretation:**
- **Low throughput (< 30 samples/sec)**: Check dataloader bottlenecks
- **High memory usage**: Consider data preprocessing optimization
- **Uneven batch timing**: Look for data augmentation inefficiencies

## ProfilerCallback

### Purpose
Integrates PyTorch Profiler for detailed performance analysis with Chrome trace export and automated bottleneck detection.

### Features
- CPU/GPU/Memory profiling
- Chrome trace export for visualization
- Configurable profiling windows
- Top-k operation analysis
- Automated bottleneck detection

### Enabling the Callback

```bash
uv run python runners/train.py \
  data=canonical \
  callbacks=profiler \
  callbacks.profiler.enabled=true \
  callbacks.profiler.profile_epochs=[1,5] \
  callbacks.profiler.profile_steps=50 \
  project_name=OCR_Training \
  exp_name=profiling_test
```

### Configuration Options

```yaml
profiler:
  _target_: ocr.callbacks.profiler.ProfilerCallback
  enabled: false                   # Disabled by default (performance overhead)
  profile_epochs: [1, 5, 10]      # Which epochs to profile
  profile_steps: 100              # Steps to profile per epoch
  warmup_steps: 5                 # Warmup steps before profiling
  activities: ["cpu", "cuda"]     # Profiling activities
  record_shapes: true             # Record tensor shapes
  with_stack: false               # Stack traces (slower but detailed)
  output_dir: "profiler_traces"   # Output directory
  export_chrome_trace: true       # Export Chrome traces
  log_top_k_ops: 15               # Top-k slowest operations to log
```

### Output and Interpretation

**Chrome Traces**: Open `profiler_traces/chrome_trace_epoch_1.json` in Chrome's `chrome://tracing/`

**Console Output:**
```
Profiler: Top 5 slowest operations:
1. conv2d (35.2% of time)
2. batch_norm (18.7% of time)
3. relu (12.3% of time)
4. max_pool2d (8.9% of time)
5. linear (6.1% of time)
```

**Interpretation:**
- Focus optimization efforts on top operations
- Look for memory bottlenecks in GPU traces
- Identify data loading vs computation imbalances

## ResourceMonitorCallback

### Purpose
Monitors system resources with intelligent alerting for GPU, CPU, memory, and I/O usage patterns.

### Metrics Tracked
- **GPU**: Utilization, memory, temperature
- **CPU**: Per-core usage, process CPU time
- **Memory**: System and process memory, OOM risk
- **Disk I/O**: Read/write rates, dataset access patterns

### Alerts
- GPU underutilization (<50% for extended periods)
- High memory pressure (>90% usage)
- I/O bottlenecks (>30% time waiting on disk)
- Temperature warnings (>80°C)

### Enabling the Callback

```bash
uv run python runners/train.py \
  data=canonical \
  callbacks=resource_monitor \
  callbacks.resource_monitor.enabled=true \
  callbacks.resource_monitor.log_interval=10 \
  project_name=OCR_Training \
  exp_name=resource_monitoring_test
```

### Configuration Options

```yaml
resource_monitor:
  _target_: ocr.callbacks.resource_monitor.ResourceMonitorCallback
  enabled: true                    # Enable/disable monitoring
  log_interval: 10                 # Log every N batches
  gpu_monitoring: true            # Monitor GPU resources
  cpu_monitoring: true            # Monitor CPU resources
  io_monitoring: true             # Monitor disk I/O
  alert_gpu_underutilization: true # Alert on GPU underuse
  alert_memory_pressure: true     # Alert on high memory usage
  alert_io_bottleneck: true       # Alert on I/O bottlenecks
  gpu_util_threshold: 0.5         # GPU utilization threshold (50%)
  memory_threshold: 0.9           # Memory usage threshold (90%)
  io_wait_threshold: 0.3          # I/O wait threshold (30%)
  export_timeseries: true         # Export time-series data
  timeseries_path: "resource_logs" # CSV export directory
```

### Output and Interpretation

**Console Logging:**
```
ResourceMonitor: GPU 0 - Util: 85%, Mem: 6.2/8.0GB, Temp: 72°C
ResourceMonitor: CPU - Usage: 45%, Memory: 12.3/32GB
ResourceMonitor: ALERT - GPU utilization below 50% for 30 batches
```

**CSV Export**: Time-series data saved to `resource_logs/epoch_0.csv`

**Interpretation:**
- **GPU underutilization**: Check batch size, model complexity, or data loading
- **High memory pressure**: Consider gradient accumulation or model sharding
- **I/O bottlenecks**: Optimize data preprocessing or use faster storage

## Combined Usage

### Enabling All Callbacks

```bash
uv run python runners/train.py \
  data=canonical \
  callbacks=throughput_monitor,profiler,resource_monitor \
  callbacks.throughput_monitor.enabled=true \
  callbacks.profiler.enabled=true \
  callbacks.profiler.profile_epochs=[1] \
  callbacks.resource_monitor.enabled=true \
  project_name=OCR_Training \
  exp_name=comprehensive_monitoring
```

### Performance Optimization Workflow

1. **Start with ThroughputMonitor**: Identify overall pipeline efficiency
2. **Enable ResourceMonitor**: Check for resource bottlenecks
3. **Use Profiler**: Deep-dive into specific performance issues
4. **Iterate**: Apply optimizations and re-measure

### Best Practices

- **Enable profiling selectively**: Only profile specific epochs to avoid overhead
- **Monitor resource alerts**: Address alerts promptly to prevent training issues
- **Use Chrome traces**: For detailed bottleneck analysis
- **Export time-series data**: For long-term performance tracking

## Troubleshooting

### Callbacks Not Working
- Verify callback names in Hydra config
- Check that `enabled=true` for each callback
- Ensure proper imports in `ocr/callbacks/__init__.py`

### Missing Metrics
- Confirm WandB logging is configured
- Check console output for error messages
- Verify hardware availability (GPU for GPU monitoring)

### Performance Impact
- Disable profiler in production (high overhead)
- Use appropriate log intervals for resource monitoring
- Consider selective enabling based on training phase

## Integration with Existing Tools

These callbacks complement the existing performance profiler usage guide:

- Use **baseline reports** for high-level performance assessment
- Use **monitoring callbacks** for detailed, real-time analysis
- Combine with **WandB logging** for comprehensive experiment tracking

## Related Files

- `ocr/callbacks/throughput_monitor.py`: Throughput monitoring implementation
- `ocr/callbacks/profiler.py`: PyTorch profiler integration
- `ocr/callbacks/resource_monitor.py`: Resource monitoring implementation
- `configs/callbacks/`: Callback configuration files
- `docs/performance/`: Performance report outputs</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/ai_handbook/03_references/performance_monitoring_callbacks_usage.md
