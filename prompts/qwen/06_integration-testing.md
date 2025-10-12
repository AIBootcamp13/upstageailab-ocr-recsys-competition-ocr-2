# Integration Testing: Phase 3 Complete System

## Mission
Validate that all three Phase 3 tasks work together correctly after parallel implementation by three Qwen agents.

## Context
After Tasks 3.1, 3.2, and 3.3 are completed independently, we need to:
1. Verify no conflicts between implementations
2. Test all features working together
3. Measure combined performance overhead
4. Create comprehensive test suite
5. Document the complete monitoring system

## Pre-Integration Checklist

Before starting integration testing, verify each task is complete:

### Task 3.1 - Throughput Metrics ✓
- [ ] `ocr/callbacks/throughput_monitor.py` exists
- [ ] `configs/callbacks/throughput_monitor.yaml` exists
- [ ] Unit tests pass: `pytest tests/test_throughput_monitor.py`
- [ ] Integration test successful (2-epoch run)
- [ ] Metrics logged to console and MLflow
- [ ] No linting or mypy errors

### Task 3.2 - Profiler Integration ✓
- [ ] `ocr/callbacks/profiler.py` exists
- [ ] `configs/callbacks/profiler.yaml` exists
- [ ] Unit tests pass: `pytest tests/test_profiler_callback.py`
- [ ] Chrome traces generated and viewable
- [ ] Bottleneck detection working
- [ ] No linting or mypy errors

### Task 3.3 - Resource Monitoring ✓
- [ ] `ocr/callbacks/resource_monitor.py` exists
- [ ] `configs/callbacks/resource_monitor.yaml` exists
- [ ] Unit tests pass: `pytest tests/test_resource_monitor.py`
- [ ] Metrics logged and CSV exported
- [ ] Alerts triggered correctly
- [ ] No linting or mypy errors

## Integration Testing Plan

### Test 1: Combined Callback Test (All Enabled)

Test all three callbacks running simultaneously:

```bash
python ocr/train.py \
    experiment=synthetic_debug \
    trainer.max_epochs=3 \
    callbacks.throughput_monitor.enabled=true \
    callbacks.profiler.enabled=true \
    callbacks.profiler.profile_epochs=[1,2] \
    callbacks.resource_monitor.enabled=true \
    callbacks.resource_monitor.log_interval=10
```

**Expected Output:**
```
[Throughput] Epoch 1: Samples/sec: 245.3, Memory: 1250 MB
[Resources] Batch 10: GPU: 87.3%, CPU: 45.2%, Memory: 8192 MB
[Profiler] Epoch 1: Profiling steps 0-100...
[Resources] Batch 20: GPU: 89.1%, CPU: 46.5%, Memory: 8250 MB
...
[Throughput] Epoch 1 complete: Avg 242.1 samples/sec
[Profiler] Top 10 operations: aten::conv2d - 125ms, ...
[Resources] Epoch 1 summary: Avg GPU: 88.2%, Alerts: 0
```

**Validation:**
- [ ] All three callbacks log output
- [ ] No interleaved/corrupted logs
- [ ] No callback conflicts or crashes
- [ ] MLflow shows all metrics
- [ ] Chrome trace exists: `profiler_traces/epoch_1.json`
- [ ] Resource CSV exists: `resource_logs/epoch_1.csv`

### Test 2: Performance Overhead Test

Measure combined overhead of all monitoring:

```bash
# Baseline (no monitoring)
time python ocr/train.py \
    experiment=synthetic_debug \
    trainer.max_epochs=1 \
    callbacks.throughput_monitor.enabled=false \
    callbacks.profiler.enabled=false \
    callbacks.resource_monitor.enabled=false

# With all monitoring
time python ocr/train.py \
    experiment=synthetic_debug \
    trainer.max_epochs=1 \
    callbacks.throughput_monitor.enabled=true \
    callbacks.profiler.enabled=true \
    callbacks.profiler.profile_steps=50 \
    callbacks.resource_monitor.enabled=true \
    callbacks.resource_monitor.log_interval=10
```

**Expected:**
- Overhead <10% (acceptable for monitoring)
- No memory leaks (memory stable across epochs)
- No GPU memory increase

**Validation:**
- [ ] Overhead within acceptable range
- [ ] Memory usage stable
- [ ] No warnings or errors

### Test 3: Selective Monitoring Test

Test enabling callbacks individually:

```bash
# Only throughput
python ocr/train.py experiment=synthetic_debug trainer.max_epochs=1 \
    callbacks.throughput_monitor.enabled=true

# Only profiler
python ocr/train.py experiment=synthetic_debug trainer.max_epochs=1 \
    callbacks.profiler.enabled=true callbacks.profiler.profile_epochs=[1]

# Only resource monitor
python ocr/train.py experiment=synthetic_debug trainer.max_epochs=1 \
    callbacks.resource_monitor.enabled=true
```

**Validation:**
- [ ] Each callback works independently
- [ ] No errors when others disabled
- [ ] Correct metrics logged for enabled callback

### Test 4: Configuration Override Test

Test Hydra config overrides work correctly:

```bash
# Override thresholds
python ocr/train.py \
    experiment=synthetic_debug \
    trainer.max_epochs=1 \
    callbacks.resource_monitor.enabled=true \
    callbacks.resource_monitor.gpu_util_threshold=0.99 \
    callbacks.resource_monitor.memory_threshold=0.01
```

**Validation:**
- [ ] Alerts triggered immediately (low thresholds)
- [ ] Config overrides applied correctly

### Test 5: Full Training Run Test

Test on real training (not synthetic):

```bash
# 10 epochs with monitoring
python ocr/train.py \
    experiment=base \
    trainer.max_epochs=10 \
    callbacks.throughput_monitor.enabled=true \
    callbacks.profiler.enabled=true \
    callbacks.profiler.profile_epochs=[1,5,10] \
    callbacks.resource_monitor.enabled=true \
    callbacks.resource_monitor.log_interval=20
```

**Validation:**
- [ ] Training completes successfully
- [ ] Monitoring data collected for all epochs
- [ ] Profiler traces for epochs 1, 5, 10
- [ ] Resource CSVs for all epochs
- [ ] MLflow metrics complete

### Test 6: Error Handling Test

Test graceful degradation:

```bash
# Test with no GPU (if applicable)
CUDA_VISIBLE_DEVICES="" python ocr/train.py \
    experiment=synthetic_debug \
    trainer.max_epochs=1 \
    callbacks.resource_monitor.enabled=true

# Test with read-only filesystem (if applicable)
# Should log warning but not crash
```

**Validation:**
- [ ] No GPU: CPU-only monitoring works
- [ ] No write permissions: Logs warning, continues
- [ ] Missing dependencies: Logs warning, disables feature

## Conflict Resolution

If conflicts arise between implementations:

### Common Conflict Types:

1. **Import conflicts** - Check `ocr/callbacks/__init__.py`
   - Ensure all three callbacks imported
   - No duplicate imports

2. **Config conflicts** - Check `configs/train.yaml`
   - Ensure all callbacks in default config
   - No conflicting parameter names

3. **Logging conflicts** - Check log output
   - Ensure loggers use unique prefixes: `[Throughput]`, `[Profiler]`, `[Resources]`
   - No interleaved output (use proper logging, not print)

4. **Resource conflicts** - Check memory/GPU usage
   - Profiler memory overhead acceptable
   - No GPU memory leaks

### Resolution Steps:
1. Identify conflicting files
2. Review both implementations
3. Merge carefully, preserving both features
4. Add tests for conflict areas
5. Rerun integration tests

## Post-Integration Tasks

### 1. Create Master Config

Create `configs/experiment/full_monitoring.yaml`:
```yaml
# @package _global_

defaults:
  - override /callbacks: default
  - override /callbacks/throughput_monitor: default
  - override /callbacks/profiler: default
  - override /callbacks/resource_monitor: default

# Full monitoring profile
trainer:
  max_epochs: 20

callbacks:
  throughput_monitor:
    enabled: true
    track_memory: true
    track_timing: true

  profiler:
    enabled: true
    profile_epochs: [1, 5, 10, 15, 20]
    profile_steps: 100
    log_top_k_ops: 15

  resource_monitor:
    enabled: true
    log_interval: 20
    alert_gpu_underutilization: true
    alert_memory_pressure: true
```

Usage:
```bash
python ocr/train.py experiment=full_monitoring
```

### 2. Update Documentation

Update `docs/monitoring.md` (create if missing):
```markdown
# Performance Monitoring System

## Overview
This project includes comprehensive performance monitoring:

1. **Throughput Metrics** (Task 3.1)
   - Samples/second tracking
   - Batch timing (p50, p95, p99)
   - Memory footprint

2. **PyTorch Profiler** (Task 3.2)
   - Chrome trace export
   - Automated bottleneck detection
   - Operation-level timing

3. **Resource Monitoring** (Task 3.3)
   - GPU/CPU/Memory tracking
   - I/O patterns
   - Intelligent alerting

## Quick Start

### Enable All Monitoring
```bash
python ocr/train.py experiment=full_monitoring
```

### Enable Selectively
```bash
# Only throughput
python ocr/train.py callbacks.throughput_monitor.enabled=true

# Only profiler
python ocr/train.py callbacks.profiler.enabled=true

# Only resources
python ocr/train.py callbacks.resource_monitor.enabled=true
```

## Outputs

- Console logs with metrics
- MLflow metrics tracking
- Chrome traces: `profiler_traces/epoch_*.json`
- Resource time-series: `resource_logs/epoch_*.csv`

## Visualization

### Chrome Traces
1. Open `chrome://tracing`
2. Load `profiler_traces/epoch_1.json`
3. Analyze operation timeline

### Resource Plots
```bash
python scripts/plot_resources.py resource_logs/epoch_1.csv
```

## Performance Impact

- **Throughput monitoring:** <1% overhead
- **Resource monitoring:** <1% overhead
- **Profiler:** 5-10% overhead (only when profiling)
- **Combined:** <10% overhead

## Configuration

See:
- `configs/callbacks/throughput_monitor.yaml`
- `configs/callbacks/profiler.yaml`
- `configs/callbacks/resource_monitor.yaml`
```

### 3. Create Visualization Script

Create `scripts/plot_monitoring_summary.py`:
```python
#!/usr/bin/env python
"""
Generate monitoring summary plots from Phase 3 outputs.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_throughput(mlflow_dir: Path):
    """Plot throughput over epochs from MLflow metrics."""
    pass


def plot_resources(resource_dir: Path):
    """Plot GPU/CPU/Memory over time from CSV files."""
    pass


def summarize_profiler(profiler_dir: Path):
    """Summarize bottlenecks from Chrome traces."""
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-dir", type=Path, default="mlruns")
    parser.add_argument("--resource-dir", type=Path, default="resource_logs")
    parser.add_argument("--profiler-dir", type=Path, default="profiler_traces")
    parser.add_argument("--output", type=Path, default="monitoring_summary.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Throughput over epochs
    plot_throughput(args.mlflow_dir)

    # Plot 2: Resource utilization
    plot_resources(args.resource_dir)

    # Plot 3: Top bottlenecks
    summarize_profiler(args.profiler_dir)

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Summary saved to {args.output}")


if __name__ == "__main__":
    main()
```

### 4. Update README

Add to main README.md:
```markdown
## Performance Monitoring

This project includes comprehensive performance monitoring. See [docs/monitoring.md](docs/monitoring.md) for details.

Quick start:
```bash
# Enable all monitoring
python ocr/train.py experiment=full_monitoring
```
```

## Success Criteria

Integration complete when:
- [ ] All three callbacks work together without conflicts
- [ ] Combined overhead <10%
- [ ] All integration tests pass
- [ ] Documentation updated
- [ ] Visualization tools created
- [ ] No linting or mypy errors in merged code
- [ ] Full training run (10+ epochs) successful with monitoring

## Rollback Plan

If integration fails critically:
1. Revert to Phase 2 checkpoint: `git reset --hard 9ea89da`
2. Analyze failure cause
3. Fix conflicts individually
4. Retry integration

## Final Deliverables

After integration complete:
1. ✅ Three monitoring callbacks fully integrated
2. ✅ Comprehensive test suite
3. ✅ Documentation complete
4. ✅ Example configs created
5. ✅ Visualization tools available
6. ✅ Performance overhead measured and acceptable

## Timeline

- **Day 1:** Three Qwen agents implement tasks in parallel (4-8 hours each)
- **Day 2:** Integration testing, conflict resolution, documentation
- **Day 3:** Final testing, visualization, polish

**Total:** 2-3 days

---

## Next Steps After Integration

Once Phase 3 is complete and integrated:
1. Run baseline vs optimized comparison
2. Generate performance report
3. Identify remaining bottlenecks from profiler data
4. Plan Phase 4 (if needed) based on findings
5. Document lessons learned
