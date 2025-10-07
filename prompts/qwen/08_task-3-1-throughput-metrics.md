# Qwen Task 3.1: Dataloader Throughput Monitoring

## Mission
Implement comprehensive dataloader throughput metrics to measure training pipeline efficiency. This is part of Phase 3 performance monitoring.

## Context
- **Project:** OCR model training pipeline optimization
- **Previous Work:** Phase 1 & 2 completed (monitoring infrastructure + polygon cache)
- **Current Branch:** `07_refactor/performance`
- **Your Task:** Independent task - implement throughput monitoring only

## What You're Building
Add metrics to track:
1. **Samples/second** throughput per epoch
2. **Memory usage** (dataset, cache, peak memory)
3. **Batch timing** (load time, transform time)
4. **Bottleneck detection** (identify slow operations)

## Implementation Requirements

### 1. Create `ThroughputMonitorCallback`
**Location:** `ocr/callbacks/throughput_monitor.py`

```python
# Required metrics to track:
# - samples_per_second (avg per epoch)
# - batch_load_time_ms (avg, p50, p95, p99)
# - batch_transform_time_ms (avg, p50, p95, p99)
# - memory_dataset_mb
# - memory_cache_mb
# - memory_peak_mb
# - batches_per_second
# - throughput_efficiency (actual vs theoretical)
```

**Key Features:**
- Use `time.perf_counter()` for accurate timing
- Track per-batch timing in `on_train_batch_start/end`
- Compute statistics in `on_train_epoch_end`
- Log to both console and MLflow
- Support enabling/disabling via config

### 2. Memory Tracking
Use existing monitoring infrastructure:
```python
# Reference: ocr/callbacks/performance_monitor.py (lines 50-70)
process = psutil.Process()
memory_info = process.memory_info()
memory_mb = memory_info.rss / 1024 / 1024
```

Track:
- Dataset memory footprint
- PolygonCache memory usage
- Peak memory during epoch

### 3. Integration Points

#### In `lightning_modules/ocr_pl.py`:
```python
def train_dataloader(self):
    # Add timing wrapper if throughput monitoring enabled
    return self._wrap_with_timing(dataloader)
```

#### In `callbacks/__init__.py`:
Add `ThroughputMonitorCallback` to exports

#### In config (`configs/callbacks/throughput_monitor.yaml`):
```yaml
throughput_monitor:
  _target_: ocr.callbacks.throughput_monitor.ThroughputMonitorCallback
  enabled: true
  log_interval: 1  # epochs
  track_memory: true
  track_timing: true
```

### 4. Output Format
Log metrics like:
```
[Throughput] Epoch 1:
  Samples/sec: 245.3
  Batches/sec: 15.3
  Avg batch load: 12.5ms (p95: 18.2ms)
  Avg batch transform: 8.3ms (p95: 14.1ms)
  Memory (dataset): 1250 MB
  Memory (cache): 340 MB
  Memory (peak): 2100 MB
  Efficiency: 87.3% (of theoretical max)
```

## Files to Modify/Create

### Create New:
1. `ocr/callbacks/throughput_monitor.py` - Main callback class
2. `configs/callbacks/throughput_monitor.yaml` - Configuration
3. `tests/test_throughput_monitor.py` - Unit tests

### Modify Existing:
1. `ocr/callbacks/__init__.py` - Add import
2. `ocr/lightning_modules/ocr_pl.py` - Optional timing wrapper
3. `configs/train.yaml` - Add callback to default config

## Testing Requirements

### Unit Tests
Create `tests/test_throughput_monitor.py`:
```python
def test_throughput_calculation():
    # Test samples/sec computation
    pass

def test_memory_tracking():
    # Test memory measurement accuracy
    pass

def test_timing_percentiles():
    # Test p50, p95, p99 calculations
    pass
```

### Integration Test
```bash
# Quick 2-epoch test
python ocr/train.py \
    experiment=synthetic_debug \
    trainer.max_epochs=2 \
    callbacks.throughput_monitor.enabled=true
```

Expected output: Throughput metrics logged after each epoch

## Success Criteria
- [ ] Callback successfully tracks all required metrics
- [ ] Memory tracking accurate (compare with `nvidia-smi`)
- [ ] Timing measurements have low overhead (<1% impact)
- [ ] Metrics logged to console and MLflow
- [ ] Unit tests pass with >80% coverage
- [ ] Integration test shows expected output
- [ ] Code passes linting (`ruff check`) and type checking (`mypy`)

## References

### Existing Code to Study:
- `ocr/callbacks/performance_monitor.py` - Memory tracking patterns
- `ocr/callbacks/polygon_cache_logger.py` - Callback structure
- `ocr/datasets/polygon_cache.py:240-260` - Cache memory tracking

### Documentation:
- PyTorch Lightning callbacks: https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html
- MLflow logging: Check existing callbacks for patterns

## Important Notes

### Independence:
- This task is **independent** of Tasks 3.2 and 3.3
- Don't modify profiler or resource monitoring code
- Focus only on throughput metrics

### Performance:
- Timing overhead should be minimal (<1ms per batch)
- Use `time.perf_counter()` not `time.time()`
- Only track detailed timing when enabled

### Error Handling:
- Gracefully handle missing memory info
- Don't crash training if monitoring fails
- Log warnings for tracking errors

## Delivery Checklist
Before marking complete:
1. All files created/modified as specified
2. Unit tests written and passing
3. Integration test successful (2-epoch run)
4. Linting and mypy clean
5. Throughput metrics visible in logs
6. MLflow metrics recorded
7. Documentation updated (if needed)

## Questions to Resolve Yourself
- Exact timing points for batch measurement
- How to calculate "theoretical max" throughput
- Whether to use separate config file or embed in train.yaml
- Percentile calculation method (numpy or custom)

## Questions to Ask if Blocked
- Issues accessing dataset/cache memory info
- Conflicts with existing callback infrastructure
- MLflow logging patterns unclear
- Performance overhead concerns

---

## Start Here
1. Read `ocr/callbacks/performance_monitor.py` to understand callback patterns
2. Create `ocr/callbacks/throughput_monitor.py` skeleton
3. Implement basic samples/sec tracking
4. Add memory tracking
5. Add batch timing
6. Add percentile calculations
7. Write tests
8. Run integration test
9. Polish and document

**Estimated Time:** 4-6 hours
**Priority:** High (blocks comprehensive performance analysis)
