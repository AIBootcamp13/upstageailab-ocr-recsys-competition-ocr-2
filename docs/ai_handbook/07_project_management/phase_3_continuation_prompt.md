# Phase 3 Continuation Prompt: Monitoring and Profiling

**Date:** 2025-10-07
**Context:** Performance Optimization Project
**Status:** Phase 1 & 2 Complete, Moving to Phase 3

---

## 🎯 Quick Context

You are continuing the **Performance Optimization Project** for a Receipt OCR system. Phases 1 and 2 are complete:
- ✅ Phase 1: Monitoring infrastructure (profiler, baseline, regression tests)
- ✅ Phase 2.1: PolygonCache implementation (ready, 0% hit rate explained)
- ✅ Phase 2.2: Cache integration complete

**Current Challenge:** Cache works correctly but shows 0% hit rate because validation images have unique polygons (expected behavior). This is fine - the infrastructure is ready for when it's needed (repeated evaluations, training augmentation, inference).

---

## 📋 Task: Start Phase 3 - Monitoring and Profiling

### Objective
Establish comprehensive performance monitoring and automated profiling to identify bottlenecks beyond PyClipper.

### Your Mission
Implement Phase 3 from the execution plan, focusing on:
1. **Task 3.1:** Performance metrics collection (dataloader throughput, memory per epoch)
2. **Task 3.2:** Automated profiling integration (PyTorch profiler, bottleneck detection)
3. **Task 3.3:** Resource monitoring (GPU/CPU tracking, I/O patterns, alerting)

---

## 📚 Essential References

### Primary Documents (Read These First)

1. **Execution Plan** - Your roadmap
   - Path: `docs/ai_handbook/07_project_management/performance_optimization_execution_plan.md`
   - Sections: Phase 3 (lines 83-117)
   - Focus: Tasks 3.1, 3.2, 3.3 specifications

2. **Phase 1 & 2 Completion Summary** - What's already done
   - Path: `docs/ai_handbook/07_project_management/phase_1_2_completion_summary.md`
   - Shows: All completed infrastructure
   - Key: PerformanceProfilerCallback already exists

3. **Task 2.2 Summary** - Cache integration details
   - Path: `docs/ai_handbook/07_project_management/task_2.2_completion_summary.md`
   - Shows: How cache was integrated (pattern to follow)

4. **Baseline Report** - Current performance metrics
   - Path: `docs/performance/baseline_2025-10-07_final.md`
   - Shows: Current bottlenecks and measurements

### Code References

5. **Performance Profiler Callback** - Existing monitoring
   - Path: `ocr/lightning_modules/callbacks/performance_profiler.py`
   - Shows: How to create Lightning callbacks
   - Pattern: Follow this for new callbacks

6. **Lightning Module** - Where callbacks integrate
   - Path: `ocr/lightning_modules/ocr_pl.py`
   - Shows: Training loop, dataloader creation
   - Key: Lines 587-605 (dataloader methods)

7. **DBCollateFN** - Data processing pipeline
   - Path: `ocr/datasets/db_collate_fn.py`
   - Shows: Where PyClipper bottleneck was
   - Note: Now has cache integration

### Configuration Examples

8. **Hydra Config Structure**
   - Path: `configs/data/base.yaml`
   - Shows: How to structure configs
   - Pattern: Follow for new monitoring configs

9. **Callback Config**
   - Path: `configs/callbacks/performance_profiler.yaml`
   - Shows: How to configure callbacks
   - Pattern: Follow for new profiling callbacks

### Testing Patterns

10. **Integration Tests**
    - Path: `tests/integration/test_performance_profiler.py`
    - Shows: How to test callbacks
    - Pattern: Follow for new monitoring tests

11. **Performance Tests**
    - Path: `tests/performance/test_regression.py`
    - Shows: Performance assertion patterns
    - Pattern: Follow for new profiling tests

---

## 🔧 Existing Infrastructure

### What You Have Available

**Callbacks:**
- ✅ `PerformanceProfilerCallback` - Validation timing and memory
  - Tracks batch times, epoch times
  - Monitors GPU/CPU memory
  - Logs to WandB and Lightning logger

**Metrics:**
- ✅ Validation time per epoch
- ✅ Batch time statistics (mean, median, p95, p99)
- ✅ GPU memory usage
- ✅ CPU memory usage

**Configuration:**
- ✅ Hydra-based config system
- ✅ Callback composition via configs
- ✅ WandB integration

**Testing:**
- ✅ 16/16 performance tests passing
- ✅ Regression test framework
- ✅ Integration test patterns

---

## 📊 Current Performance Baseline

From `docs/performance/baseline_2025-10-07_final.md`:

```
Validation Time: 16.29s (34 batches)
Mean Batch Time: 436.2ms
P95 Batch Time: 617.1ms
H-Mean Accuracy: 0.9561
GPU Memory: 0.06 GB
CPU Memory: 7.8%
```

**Observation:** High variance (P95 is 1.4x mean) suggests potential bottlenecks.

---

## 🎯 Phase 3 Implementation Guide

### Task 3.1: Enhanced Metrics Collection

**What to Build:**
- Dataloader throughput callback (samples/sec, batches/sec)
- Per-epoch memory tracking (peak, average, delta)
- PyClipper operation timing (if possible to instrument)

**Pattern to Follow:**
```python
# Similar to PerformanceProfilerCallback
class DataLoaderThroughputCallback(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Track batch start time

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Calculate throughput
        # Log to WandB
```

**Config Pattern:**
```yaml
# configs/callbacks/throughput_monitor.yaml
throughput_monitor:
  _target_: ocr.lightning_modules.callbacks.DataLoaderThroughputCallback
  log_interval: 10
  track_samples_per_sec: true
```

### Task 3.2: PyTorch Profiler Integration

**What to Build:**
- PyTorch profiler callback wrapper
- Automated bottleneck detection
- Profile report generation

**Resources:**
- PyTorch Profiler docs: https://pytorch.org/docs/stable/profiler.html
- Lightning profiler: https://lightning.ai/docs/pytorch/stable/tuning/profiler.html

**Pattern:**
```python
from torch.profiler import profile, ProfilerActivity

class ProfilerCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        # Setup profiler

    def on_train_epoch_end(self, trainer, pl_module):
        # Generate report
```

### Task 3.3: Resource Monitoring

**What to Build:**
- GPU utilization tracking (not just memory)
- I/O monitoring (disk reads, network if applicable)
- Alerting for performance regressions

**Tools Available:**
- `nvidia-ml-py` (already in dependencies)
- `psutil` (already in dependencies)
- WandB alerts

---

## 🚀 Suggested Approach

### Option 1: Sequential Implementation (Safest)
1. Implement Task 3.1 (throughput metrics)
2. Test and validate
3. Implement Task 3.2 (profiler)
4. Test and validate
5. Implement Task 3.3 (resource monitoring)
6. Test and validate

**Time:** ~1 week
**Risk:** Low
**Thoroughness:** High

### Option 2: Parallel with Qwen (Faster)
1. Create Qwen prompts for all 3 tasks
2. Delegate in parallel
3. Validate and integrate all
4. Run comprehensive tests

**Time:** ~2-3 days
**Risk:** Medium (need careful integration)
**Thoroughness:** High

### Option 3: Minimal Viable (Quick Win)
1. Just implement Task 3.1 (throughput)
2. Use existing PyTorch profiler (no custom callback)
3. Manual resource monitoring

**Time:** ~1 day
**Risk:** Low
**Thoroughness:** Medium

---

## 📁 File Structure

### Files to Create

```
ocr/lightning_modules/callbacks/
├── throughput_monitor.py         # Task 3.1
├── profiler_callback.py           # Task 3.2
└── resource_monitor.py            # Task 3.3

configs/callbacks/
├── throughput_monitor.yaml        # Task 3.1 config
├── profiler.yaml                  # Task 3.2 config
└── resource_monitor.yaml          # Task 3.3 config

tests/integration/
├── test_throughput_monitor.py     # Task 3.1 tests
├── test_profiler.py               # Task 3.2 tests
└── test_resource_monitor.py       # Task 3.3 tests

scripts/performance/
├── analyze_profile.py             # Profile analysis
└── generate_performance_report.py # Comprehensive report
```

---

## 🧪 Testing Strategy

### Integration Tests
For each callback:
```python
def test_callback_enabled():
    """Test callback works when enabled"""

def test_callback_disabled():
    """Test callback does nothing when disabled"""

def test_metrics_logged():
    """Test metrics appear in logs"""

def test_callback_verbose_mode():
    """Test console output when verbose"""
```

### Performance Tests
```python
def test_callback_overhead():
    """Ensure callback adds <5% overhead"""

def test_throughput_calculation():
    """Verify samples/sec calculation accurate"""

def test_memory_tracking():
    """Verify memory measurements accurate"""
```

---

## 💡 Key Patterns from Previous Work

### 1. Callback Pattern (from PerformanceProfilerCallback)
```python
class NewCallback(Callback):
    def __init__(self, enabled=True, verbose=False):
        super().__init__()
        self.enabled = enabled
        self.verbose = verbose

    def on_train_epoch_start(self, trainer, pl_module):
        if not self.enabled:
            return
        # Your logic here
```

### 2. Configuration Pattern (from cache integration)
```python
# In Lightning module
callback_cfg = getattr(self.config, "new_callback", None)
if callback_cfg and callback_cfg.get("enabled", False):
    callback = NewCallback(**callback_cfg)
```

### 3. Testing Pattern (from performance tests)
```python
@pytest.fixture
def dummy_model():
    """Minimal model for testing"""

def test_feature(dummy_model, dummy_dataloader):
    """Test with minimal setup"""
```

---

## 📊 Success Criteria

### Task 3.1 Success
- [ ] Throughput metrics logged (samples/sec, batches/sec)
- [ ] Per-epoch memory tracking working
- [ ] Metrics appear in WandB
- [ ] <5% overhead on training

### Task 3.2 Success
- [ ] PyTorch profiler integrated
- [ ] Profile reports generated automatically
- [ ] Bottlenecks identified
- [ ] Can run on-demand or periodic

### Task 3.3 Success
- [ ] GPU utilization tracked
- [ ] I/O patterns monitored
- [ ] Alerting system working
- [ ] Dashboard available (WandB)

---

## 🤝 Delegation Strategy

### Qwen Prompt Template (if delegating)

```markdown
# Task: Implement [Callback Name]

## Context
- Receipt OCR project with PyTorch Lightning + Hydra
- PerformanceProfilerCallback exists as reference
- Follow patterns from `ocr/lightning_modules/callbacks/performance_profiler.py`

## Objective
[Specific callback goal]

## Requirements
1. [Requirement 1]
2. [Requirement 2]

## Reference Files
- ocr/lightning_modules/callbacks/performance_profiler.py
- configs/callbacks/performance_profiler.yaml
- tests/integration/test_performance_profiler.py

## Implementation
[Detailed specs]

## Validation
```bash
uv run mypy [file]
uv run ruff check [file]
uv run pytest [test_file]
```
```

---

## 🔗 Quick Links

**Documentation:**
- Execution Plan: `docs/ai_handbook/07_project_management/performance_optimization_execution_plan.md`
- Completion Summary: `docs/ai_handbook/07_project_management/phase_1_2_completion_summary.md`
- Baseline Report: `docs/performance/baseline_2025-10-07_final.md`

**Code:**
- Callbacks: `ocr/lightning_modules/callbacks/`
- Lightning Module: `ocr/lightning_modules/ocr_pl.py`
- Configs: `configs/callbacks/`

**Tests:**
- Integration: `tests/integration/`
- Performance: `tests/performance/`

**Tools:**
- Qwen Prompts: `prompts/qwen/`
- Delegation Script: `scripts/agent_tools/delegate_to_qwen.py`

---

## ✨ Starting Point

**First Steps:**
1. Read the execution plan (Phase 3 section)
2. Review PerformanceProfilerCallback for patterns
3. Decide on implementation approach (sequential, parallel, or minimal)
4. Start with Task 3.1 (throughput monitoring) - easiest win

**Suggested First Command:**
```bash
# Read the execution plan
cat docs/ai_handbook/07_project_management/performance_optimization_execution_plan.md | grep -A 50 "Phase 3:"

# Review existing callback
cat ocr/lightning_modules/callbacks/performance_profiler.py

# Check current metrics
cat docs/performance/baseline_2025-10-07_final.md
```

---

## 📞 Status Update

**What's Complete:**
- ✅ Performance profiler callback (validation timing, memory)
- ✅ Baseline report (current metrics documented)
- ✅ Regression tests (CI safety net)
- ✅ PolygonCache (implemented, integrated, tested)
- ✅ 100% test pass rate (16/16 tests)

**What's Next:**
- 🔄 Task 3.1: Throughput metrics
- 🔄 Task 3.2: Automated profiling
- 🔄 Task 3.3: Resource monitoring

**Project Health:**
- ✅ All infrastructure working
- ✅ Clear baseline established
- ✅ Zero breaking changes
- ✅ Good test coverage

---

## 🎯 Your Goal

Implement Phase 3 monitoring and profiling to:
1. Track dataloader throughput
2. Automate profiling and bottleneck detection
3. Monitor system resources and alert on issues

**Expected Outcome:**
- Comprehensive performance visibility
- Automated bottleneck identification
- Proactive regression detection

**Time Estimate:** 1 week (sequential) or 2-3 days (parallel with Qwen)

---

**Good luck! You have all the infrastructure and patterns you need. Phase 3 is about expanding the monitoring we started in Phase 1.** 🚀

---

**Last Updated:** 2025-10-07
**Created by:** Claude Code (AI Agent)
**For:** Fresh context continuation
