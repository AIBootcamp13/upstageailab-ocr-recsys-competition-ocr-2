# Phase 3 Quick Reference

**Last Updated:** 2025-10-07
**Status:** Ready to Start

## Current State

✅ **Phase 1 & 2 Complete:**
- Performance profiler callback working
- Baseline established (16.29s validation)
- Regression tests in place (16/16 passing)
- PolygonCache implemented and integrated
- Zero breaking changes

## Phase 3 Tasks

### Task 3.1: Enhanced Metrics Collection
**Goal:** Track dataloader throughput (samples/sec, batches/sec)
**Files to Create:**
- `ocr/lightning_modules/callbacks/throughput_monitor.py`
- `configs/callbacks/throughput_monitor.yaml`
- `tests/integration/test_throughput_monitor.py`

**Pattern:** Follow `PerformanceProfilerCallback` structure

### Task 3.2: PyTorch Profiler Integration
**Goal:** Automated profiling and bottleneck detection
**Files to Create:**
- `ocr/lightning_modules/callbacks/profiler_callback.py`
- `configs/callbacks/profiler.yaml`
- `tests/integration/test_profiler.py`
- `scripts/performance/analyze_profile.py`

**Resources:**
- PyTorch Profiler: https://pytorch.org/docs/stable/profiler.html
- Lightning Profiler: https://lightning.ai/docs/pytorch/stable/tuning/profiler.html

### Task 3.3: Resource Monitoring
**Goal:** GPU utilization, I/O patterns, alerting
**Files to Create:**
- `ocr/lightning_modules/callbacks/resource_monitor.py`
- `configs/callbacks/resource_monitor.yaml`
- `tests/integration/test_resource_monitor.py`

**Tools Available:** `nvidia-ml-py`, `psutil`, WandB alerts

## Essential Files

**Reference Code:**
- [PerformanceProfilerCallback](ocr/lightning_modules/callbacks/performance_profiler.py) - Main pattern
- [Lightning Module](ocr/lightning_modules/ocr_pl.py:587-605) - Dataloader methods
- [Baseline Report](../../../docs/performance/baseline_2025-10-07_final.md) - Current metrics

**Reference Config:**
- [Base Config](configs/data/base.yaml) - Structure pattern
- [Callback Config](configs/callbacks/performance_profiler.yaml) - Callback pattern

**Reference Tests:**
- [Integration Tests](tests/integration/test_performance_profiler.py) - Test pattern
- [Performance Tests](tests/performance/test_regression.py) - Assertion pattern

## Quick Start Commands

```bash
# Review execution plan (Phase 3)
cat docs/ai_handbook/07_project_management/performance_optimization_execution_plan.md | grep -A 50 "Phase 3:"

# Review existing callback (pattern reference)
cat ocr/lightning_modules/callbacks/performance_profiler.py

# Check current baseline
cat docs/performance/baseline_2025-10-07_final.md

# Review delegation log
cat docs/ai_handbook/07_project_management/qwen_delegation_log.md
```

## Implementation Options

### Option 1: Sequential (Safest)
1. Task 3.1 → Test → Validate
2. Task 3.2 → Test → Validate
3. Task 3.3 → Test → Validate

**Time:** ~1 week | **Risk:** Low

### Option 2: Parallel with Qwen (Faster)
1. Create 3 Qwen prompts
2. Delegate all in parallel
3. Validate and integrate

**Time:** 2-3 days | **Risk:** Medium

### Option 3: Minimal Viable (Quick Win)
1. Just Task 3.1 (throughput)
2. Use existing PyTorch profiler
3. Manual resource monitoring

**Time:** ~1 day | **Risk:** Low

## Success Criteria

**Task 3.1:**
- [ ] Throughput metrics logged (samples/sec, batches/sec)
- [ ] Per-epoch memory tracking
- [ ] WandB integration
- [ ] <5% overhead

**Task 3.2:**
- [ ] PyTorch profiler integrated
- [ ] Automated reports
- [ ] Bottleneck identification
- [ ] On-demand + periodic modes

**Task 3.3:**
- [ ] GPU utilization tracked
- [ ] I/O patterns monitored
- [ ] Alerting system
- [ ] WandB dashboard

## Callback Pattern Template

```python
from lightning.pytorch.callbacks import Callback

class NewCallback(Callback):
    def __init__(self, enabled=True, verbose=False):
        super().__init__()
        self.enabled = enabled
        self.verbose = verbose

    def on_train_epoch_start(self, trainer, pl_module):
        if not self.enabled:
            return
        # Your logic here

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.enabled:
            return
        # Log metrics to WandB and Lightning
        pl_module.log_dict(metrics, on_epoch=True, sync_dist=True)
```

## Validation Commands

```bash
# Type checking
uv run mypy ocr/lightning_modules/callbacks/new_callback.py

# Linting
uv run ruff check ocr/lightning_modules/callbacks/new_callback.py

# Testing
uv run pytest tests/integration/test_new_callback.py -v

# Integration smoke test
uv run python -m ocr.train trainer.fast_dev_run=true +callbacks.new_callback.enabled=true
```

## Project Health

- ✅ 16/16 tests passing
- ✅ Zero breaking changes
- ✅ Clear baseline (16.29s validation, 436.2ms/batch)
- ✅ High variance identified (P95 = 1.4x mean)
- ✅ Cache infrastructure ready

## Next Session Startup

1. Read [phase_3_continuation_prompt.md](phase_3_continuation_prompt.md) (full context)
2. Review this quick reference
3. Choose implementation approach
4. Start with Task 3.1 (easiest win)

---

**For Full Details:** See [phase_3_continuation_prompt.md](phase_3_continuation_prompt.md)
