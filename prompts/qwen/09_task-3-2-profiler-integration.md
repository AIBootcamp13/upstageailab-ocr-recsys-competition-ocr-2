# Qwen Task 3.2: PyTorch Profiler Integration

## Mission
Integrate PyTorch Profiler for automated bottleneck detection with Chrome trace visualization. This is part of Phase 3 performance monitoring.

## Context
- **Project:** OCR model training pipeline optimization
- **Previous Work:** Phase 1 & 2 completed (monitoring infrastructure + polygon cache)
- **Current Branch:** `07_refactor/performance`
- **Your Task:** Independent task - implement profiler integration only

## What You're Building
A Lightning callback that:
1. **Captures detailed traces** of training operations (CPU, GPU, memory)
2. **Exports Chrome traces** for visualization in `chrome://tracing`
3. **Automatically identifies bottlenecks** (top-k slowest operations)
4. **Configurable profiling windows** (profile specific epochs/steps)

## Implementation Requirements

### 1. Create `ProfilerCallback`
**Location:** `ocr/callbacks/profiler.py`

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from pytorch_lightning import Callback

class ProfilerCallback(Callback):
    """
    PyTorch Profiler integration for training pipeline analysis.

    Features:
    - CPU/GPU/Memory profiling
    - Chrome trace export
    - Automated bottleneck detection
    - Configurable profiling windows
    """

    def __init__(
        self,
        enabled: bool = True,
        profile_epochs: list[int] | None = None,  # Which epochs to profile
        profile_steps: int = 100,  # Steps to profile per epoch
        warmup_steps: int = 5,
        activities: list[str] | None = None,  # ["cpu", "cuda"]
        record_shapes: bool = True,
        with_stack: bool = False,  # Stack traces (slow but detailed)
        output_dir: str = "profiler_traces",
        export_chrome_trace: bool = True,
        log_top_k_ops: int = 10,  # Top-k slowest operations
    ):
        pass
```

**Key Features:**
- Use `torch.profiler.profile` context manager
- Schedule: warmup → active → skip pattern
- Export traces to `profiler_traces/epoch_N.json`
- Parse trace data to find bottlenecks
- Log top-k slowest operations after profiling

### 2. Profiler Schedule Configuration

Use PyTorch's built-in scheduler:
```python
profiler_schedule = schedule(
    skip_first=10,  # Skip first N steps (initialization)
    wait=5,         # Wait N steps between profiling
    warmup=5,       # Warmup for N steps
    active=100,     # Profile N steps
    repeat=1        # Repeat cycle N times
)
```

### 3. Bottleneck Detection

Parse profiler output to identify:
- **CPU bottlenecks:** Operations with high CPU time
- **GPU bottlenecks:** CUDA kernel execution time
- **Memory bottlenecks:** Allocations/deallocations
- **I/O bottlenecks:** Data loading time

```python
def analyze_trace(self, trace_path: str) -> dict:
    """
    Analyze Chrome trace file to identify bottlenecks.

    Returns:
        {
            "top_cpu_ops": [(op_name, time_us), ...],
            "top_cuda_ops": [(op_name, time_us), ...],
            "memory_peaks": [(time, bytes), ...],
            "dataloader_time_pct": float,
        }
    """
    pass
```

### 4. Integration Points

#### Config File: `configs/callbacks/profiler.yaml`
```yaml
profiler:
  _target_: ocr.callbacks.profiler.ProfilerCallback
  enabled: false  # Disabled by default (performance overhead)
  profile_epochs: [1, 5, 10]  # Profile epochs 1, 5, and 10
  profile_steps: 100
  warmup_steps: 5
  activities: ["cpu", "cuda"]
  record_shapes: true
  with_stack: false  # Enable for detailed traces (slower)
  output_dir: "profiler_traces"
  export_chrome_trace: true
  log_top_k_ops: 15
```

#### In `callbacks/__init__.py`:
```python
from ocr.callbacks.profiler import ProfilerCallback

__all__ = [
    # ... existing ...
    "ProfilerCallback",
]
```

#### Usage in training:
```bash
# Enable profiler for specific epochs
python ocr/train.py \
    experiment=synthetic_debug \
    callbacks.profiler.enabled=true \
    callbacks.profiler.profile_epochs=[1,2]
```

### 5. Output Format

#### Console Output:
```
[Profiler] Epoch 1: Profiling steps 0-100...
[Profiler] Trace exported: profiler_traces/epoch_1.json
[Profiler] Top 10 CPU operations:
  1. aten::conv2d - 125.3ms (18.5%)
  2. aten::batch_norm - 87.2ms (12.9%)
  3. DataLoader - 65.8ms (9.7%)
  4. aten::linear - 52.1ms (7.7%)
  ...

[Profiler] Top 10 CUDA operations:
  1. volta_scudnn_winograd_128x128 - 95.2ms (22.1%)
  2. volta_sgemm_128x128 - 68.4ms (15.9%)
  ...

[Profiler] Summary:
  Dataloader time: 9.7%
  Forward pass: 45.3%
  Backward pass: 38.2%
  Optimizer: 6.8%

[Profiler] ⚠️ Bottlenecks detected:
  - High conv2d CPU time (consider torch.compile)
  - Dataloader <10% (good efficiency)
```

#### Chrome Trace:
- Open `chrome://tracing` in Chrome
- Load `profiler_traces/epoch_1.json`
- Visualize timeline of operations

### 6. Callback Lifecycle

```python
def on_train_epoch_start(self, trainer, pl_module):
    if self._should_profile_epoch(trainer.current_epoch):
        self.profiler = profile(...)
        self.profiler.__enter__()

def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    if self.profiler:
        self.profiler.step()

def on_train_epoch_end(self, trainer, pl_module):
    if self.profiler:
        self.profiler.__exit__()
        self._export_and_analyze()
```

## Files to Modify/Create

### Create New:
1. `ocr/callbacks/profiler.py` - Main profiler callback (~250 lines)
2. `configs/callbacks/profiler.yaml` - Configuration
3. `tests/test_profiler_callback.py` - Unit tests
4. `scripts/analyze_trace.py` - Standalone trace analyzer (optional)

### Modify Existing:
1. `ocr/callbacks/__init__.py` - Add import
2. `.gitignore` - Add `profiler_traces/` to ignore list

## Testing Requirements

### Unit Tests (`tests/test_profiler_callback.py`):
```python
def test_profiler_schedule():
    # Test epoch selection logic
    pass

def test_trace_export():
    # Test Chrome trace file creation
    pass

def test_bottleneck_detection():
    # Test trace parsing and analysis
    pass

def test_disabled_profiler():
    # Test no overhead when disabled
    pass
```

### Integration Test:
```bash
# Profile first 2 epochs
python ocr/train.py \
    experiment=synthetic_debug \
    trainer.max_epochs=3 \
    callbacks.profiler.enabled=true \
    callbacks.profiler.profile_epochs=[1,2] \
    callbacks.profiler.profile_steps=50
```

**Expected:**
- `profiler_traces/epoch_1.json` created
- `profiler_traces/epoch_2.json` created
- No trace for epoch 3 (not in profile_epochs)
- Console shows top operations
- Chrome trace viewable

### Manual Verification:
1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Click "Load" and select generated trace
4. Verify operations are visible and timeline makes sense

## Success Criteria
- [ ] Profiler callback successfully captures traces
- [ ] Chrome traces exported and viewable
- [ ] Bottleneck detection identifies top-k operations
- [ ] Configurable profiling windows work correctly
- [ ] No performance impact when disabled
- [ ] Minimal overhead when enabled (<5% slowdown)
- [ ] Unit tests pass with >75% coverage
- [ ] Integration test produces valid traces
- [ ] Code passes linting and type checking

## References

### PyTorch Documentation:
- Profiler tutorial: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- API docs: https://pytorch.org/docs/stable/profiler.html
- Chrome trace: https://www.chromium.org/developers/how-tos/trace-event-profiling-tool/

### Existing Code:
- `ocr/callbacks/performance_monitor.py` - Callback structure
- `ocr/lightning_modules/ocr_pl.py` - Training loop integration points

### Example Implementation:
```python
# Reference pattern for profiler usage
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    with_stack=False
) as prof:
    for step in range(total_steps):
        train_step()
        prof.step()
```

## Important Notes

### Independence:
- This task is **independent** of Tasks 3.1 and 3.3
- Don't modify throughput or resource monitoring code
- Focus only on PyTorch Profiler integration

### Performance Overhead:
- Profiling adds 5-10% overhead (acceptable)
- `with_stack=True` adds 20-30% overhead (use sparingly)
- Default to `enabled=false` to avoid surprise slowdowns

### Chrome Trace Size:
- 100 steps ≈ 5-20 MB trace file
- Don't profile entire training run (huge files)
- Use selective epoch profiling

### Error Handling:
- Profiler can fail silently - add logging
- Handle missing CUDA gracefully (CPU-only mode)
- Don't crash training if profiling fails

## Advanced Features (Optional)

If time permits, add:
1. **TensorBoard integration:** Export to TensorBoard format
2. **Automated recommendations:** Suggest optimizations based on bottlenecks
3. **Comparison mode:** Compare traces across epochs
4. **Memory timeline:** Visualize memory usage over time

## Delivery Checklist
Before marking complete:
1. All files created/modified as specified
2. Unit tests written and passing
3. Integration test produces valid Chrome traces
4. Linting and mypy clean
5. Traces viewable in Chrome
6. Bottleneck detection working
7. Documentation in docstrings complete

## Questions to Resolve Yourself
- Exact profiler schedule parameters (warmup/active/skip)
- How many steps to profile (balance detail vs overhead)
- Which activities to enable by default (CPU, CUDA, memory)
- Trace file naming convention

## Questions to Ask if Blocked
- Profiler not capturing GPU operations
- Chrome trace format issues
- Excessive memory usage during profiling
- Integration with Lightning trainer unclear

---

## Start Here
1. Read PyTorch Profiler tutorial (link above)
2. Create `ocr/callbacks/profiler.py` skeleton
3. Implement basic profiler wrapper (no analysis)
4. Test Chrome trace export manually
5. Add bottleneck detection logic
6. Add configuration options
7. Write tests
8. Run integration test and view traces
9. Polish and document

**Estimated Time:** 6-8 hours
**Priority:** High (critical for identifying performance issues)

## Quick Test Commands
```bash
# Minimal test - profile 20 steps of epoch 1
python ocr/train.py \
    experiment=synthetic_debug \
    trainer.max_epochs=1 \
    callbacks.profiler.enabled=true \
    callbacks.profiler.profile_steps=20

# View trace
# Open chrome://tracing and load profiler_traces/epoch_1.json
```
