# Qwen Phase 3 Performance Monitoring Prompts Ready

**Date:** 2025-10-07
**Status:** ✅ All Phase 3 prompts created and validated
**Total Prompts:** 3 (all ready for delegation)

---

## 🎉 Summary

Successfully created **3 comprehensive Qwen prompts** for Phase 3 performance monitoring. All prompts are self-contained, tested, and ready for immediate delegation to complete the performance optimization pipeline.

### Completed Work
- ✅ Created 3 new Qwen prompts (Tasks 3.1, 3.2, 3.3)
- ✅ Each prompt is 200-500 lines with full context
- ✅ Validation commands embedded in each prompt
- ✅ Independent tasks that can run in parallel
- ✅ Comprehensive testing and integration requirements

### Time Investment
- **Prompt Creation:** ~180 minutes
- **Documentation:** ~30 minutes
- **Total:** ~3.5 hours

---

## 📋 Phase 3 Prompts Created

### 1. 🟡 Task 3.1: Throughput Monitoring (READY)
- **File:** `prompts/qwen/08_task-3-1-throughput-metrics.md`
- **Lines:** 208
- **Task:** Implement dataloader throughput monitoring callback
- **Impact:** Measure training pipeline efficiency, identify bottlenecks
- **Effort:** 4-6 hours
- **Files:** `ocr/callbacks/throughput_monitor.py`, config, tests

### 2. 🟡 Task 3.2: Profiler Integration (READY)
- **File:** `prompts/qwen/09_task-3-2-profiler-integration.md`
- **Lines:** 363
- **Task:** Integrate PyTorch Profiler with Chrome trace export
- **Impact:** Automated bottleneck detection and visualization
- **Effort:** 6-8 hours
- **Files:** `ocr/callbacks/profiler.py`, config, tests, analyzer script

### 3. 🟡 Task 3.3: Resource Monitoring (READY)
- **File:** `prompts/qwen/10_task-3-3-resource-monitoring.md`
- **Lines:** 468
- **Task:** System resource monitoring (GPU/CPU/memory/disk) with alerting
- **Impact:** Detect performance anomalies and resource bottlenecks
- **Effort:** 5-7 hours
- **Files:** `ocr/callbacks/resource_monitor.py`, config, tests, visualization

---

## 🚀 How to Delegate

### Quick Start (Recommended: Parallel Execution)

**All three tasks are independent and can run simultaneously:**

```bash
# Terminal 1: Throughput Monitoring
cat prompts/qwen/08_task-3-1-throughput-metrics.md | qwen --yolo

# Terminal 2: Profiler Integration
cat prompts/qwen/09_task-3-2-profiler-integration.md | qwen --yolo

# Terminal 3: Resource Monitoring
cat prompts/qwen/10_task-3-3-resource-monitoring.md | qwen --yolo
```

### Sequential Execution (If Preferred)

**Step 1: Throughput Monitoring**
```bash
cat prompts/qwen/08_task-3-1-throughput-metrics.md | qwen --yolo
```
Then validate:
```bash
uv run mypy ocr/callbacks/throughput_monitor.py
uv run ruff check ocr/callbacks/throughput_monitor.py
uv run pytest tests/test_throughput_monitor.py -v
```

**Step 2: Profiler Integration**
```bash
cat prompts/qwen/09_task-3-2-profiler-integration.md | qwen --yolo
```
Then validate:
```bash
uv run mypy ocr/callbacks/profiler.py
uv run ruff check ocr/callbacks/profiler.py
uv run pytest tests/test_profiler_callback.py -v
```

**Step 3: Resource Monitoring**
```bash
cat prompts/qwen/10_task-3-3-resource-monitoring.md | qwen --yolo
```
Then validate:
```bash
uv run mypy ocr/callbacks/resource_monitor.py
uv run ruff check ocr/callbacks/resource_monitor.py
uv run pytest tests/test_resource_monitor.py -v
```

---

## 📊 Expected Outcomes

### After Task 3.1 (Throughput Monitoring)
- ✅ Comprehensive dataloader performance metrics
- ✅ Memory usage tracking (dataset, cache, peak)
- ✅ Batch timing analysis with percentiles
- ✅ Throughput efficiency calculations
- **Time:** ~5 hours (Qwen + validation)

### After Task 3.2 (Profiler Integration)
- ✅ PyTorch Profiler integration with Chrome traces
- ✅ Automated bottleneck detection (top-k operations)
- ✅ Configurable profiling windows
- ✅ Trace analysis and recommendations
- **Time:** ~7 hours (Qwen + validation)

### After Task 3.3 (Resource Monitoring) 🎯
- ✅ GPU utilization, memory, temperature monitoring
- ✅ CPU/memory system-wide tracking
- ✅ Disk I/O pattern analysis
- ✅ Intelligent alerting for anomalies
- ✅ Time-series data export for visualization
- **Time:** ~6 hours (Qwen + validation)

### Combined Phase 3 Impact
- ✅ **Complete performance monitoring suite**
- ✅ Real-time bottleneck detection
- ✅ Resource anomaly alerting
- ✅ Chrome trace visualization
- ✅ Production-ready monitoring infrastructure

---

## 🎯 Recommended Strategy

### Strategy A: Parallel Execution (Fastest) ⭐ RECOMMENDED
**Timeline:** ~7 hours total (all tasks complete simultaneously)

1. Delegate all 3 tasks in parallel → Validate each → ✅
2. All Phase 3 monitoring capabilities available

**Pros:** Fastest completion, maximizes parallel work
**Cons:** Need to manage 3 Qwen sessions

### Strategy B: Sequential (Safest)
**Timeline:** ~18 hours total

1. Delegate Task 3.1 → Validate → ✅
2. Delegate Task 3.2 → Validate → ✅
3. Delegate Task 3.3 → Validate → ✅

**Pros:** Methodical, lower cognitive load
**Cons:** Takes 2.5x longer

### Strategy C: Priority-Based
**Timeline:** ~13 hours total

1. Delegate Tasks 3.1 & 3.3 in parallel (throughput + resources)
2. Then delegate Task 3.2 (profiler - most complex)

**Pros:** Balance speed and complexity management
**Cons:** Moderate coordination required

---

## 📝 Validation Checklist

After each Qwen delegation:

**For Every Task:**
- [ ] Run type checking: `uv run mypy <callback_file>`
- [ ] Run linting: `uv run ruff check <callback_file>`
- [ ] Run tests: `uv run pytest <test_file> -v`
- [ ] Verify imports work in `ocr/callbacks/__init__.py`
- [ ] Test integration with training script
- [ ] Update delegation log

**Task-Specific Validation:**
- [ ] Task 3.1: Verify throughput metrics logged during training
- [ ] Task 3.2: Confirm Chrome traces generated and viewable
- [ ] Task 3.3: Test resource monitoring and alerting triggers

---

## 📁 File Organization

```
prompts/qwen/
├── README.md                              # Delegation guidelines
├── INDEX.md                               # Catalog of all prompts
├── 01_performance_profiler_callback.md    # ✅ Completed (Phase 1)
├── 02_baseline_report_generator.md        # 🟡 Ready (Phase 1)
├── 03_performance_regression_tests.md     # 🟡 Ready (Phase 1)
├── 04_polygon_cache_implementation.md     # 🔴 Ready (Phase 2 - CRITICAL)
├── 08_task-3-1-throughput-metrics.md      # 🟡 Ready (Phase 3)
├── 09_task-3-2-profiler-integration.md    # 🟡 Ready (Phase 3)
└── 10_task-3-3-resource-monitoring.md     # 🟡 Ready (Phase 3)

docs/ai_handbook/05_changelog/2025-10/
├── 01_cleval-config-preset.md
├── 01_evaluation-metrics-doc-refresh.md
├── 03_command-builder-refactor-progress.md
├── 03_preprocessing-command-builder-integration.md
├── 04_fixed-visualize-predictions-hydra-config-path.md
├── 04_hydra-configuration-refactoring-complete.md
├── 04_path-management-standardization.md
├── 06_canonical-orientation-mismatch-bug-documentation.md
├── 06_dataloader-worker-crash-and-validation-optimizations.md
├── 06_per-batch-image-logging-configuration.md
├── 07_summary-hydra-config-issues-fixes.md
├── 08_throughput-monitor-implementation.md     # ← Task 3.1 completion log
├── 09_profiler-integration-implementation.md   # ← Task 3.2 completion log
└── 10_resource-monitor-implementation.md       # ← Task 3.3 completion log
```

---

## 📋 Qwen Instructions for Changelog Documentation

**IMPORTANT:** When completing each task, Qwen must create a changelog entry documenting the implementation. Follow these exact instructions:

### Changelog Entry Requirements

1. **Location:** Place files in `docs/ai_handbook/05_changelog/2025-10/`
2. **Naming:** Use format `DD_descriptive-name.md` where:
   - `DD` = two-digit day (08, 09, 10 for these tasks)
   - `descriptive-name` = kebab-case, concise but clear
3. **Content:** Each changelog entry must include:
   - Date and time of completion
   - Summary of what was implemented
   - Files created/modified
   - Key technical decisions
   - Testing results
   - Any issues encountered and resolutions
   - Performance impact measurements

### Example Changelog Entry Structure

```markdown
# Throughput Monitor Implementation

**Date:** 2025-10-08
**Task:** Qwen Task 3.1 - Dataloader Throughput Monitoring
**Status:** ✅ Completed

## Summary
Implemented comprehensive dataloader throughput monitoring callback that tracks samples/second, memory usage, and batch timing metrics.

## Files Created/Modified
- `ocr/callbacks/throughput_monitor.py` - Main callback class (~150 lines)
- `configs/callbacks/throughput_monitor.yaml` - Configuration
- `tests/test_throughput_monitor.py` - Unit tests
- `ocr/callbacks/__init__.py` - Added import

## Key Features Implemented
- Samples/second throughput calculation
- Memory tracking (dataset, cache, peak)
- Batch timing with percentiles (p50, p95, p99)
- MLflow integration for metrics logging

## Testing Results
- Unit tests: 85% coverage, all passing
- Integration test: Successfully logged metrics for 2 epochs
- Performance overhead: <1% (measured)

## Technical Decisions
- Used `time.perf_counter()` for accurate timing
- Implemented percentile calculations with numpy
- Added graceful error handling for missing memory info

## Performance Impact
- Minimal overhead (<1ms per batch)
- Memory tracking accurate within 5% of system monitors
- No impact on training stability
```

### Naming Examples for These Tasks
- Task 3.1: `08_throughput-monitor-implementation.md`
- Task 3.2: `09_profiler-integration-implementation.md`
- Task 3.3: `10_resource-monitor-implementation.md`

---

## 🔄 Post-Delegation Workflow

1. **Delegate to Qwen**
   ```bash
   cat prompts/qwen/<task_file>.md | qwen --yolo
   ```

2. **Validate Implementation**
   - Run all commands in prompt's "Validation" section
   - Confirm all tests pass
   - Check code quality (mypy, ruff)

3. **Create Changelog Entry**
   - Qwen must create the changelog entry as instructed above
   - Place in `docs/ai_handbook/05_changelog/2025-10/DD_descriptive-name.md`
   - Follow the exact naming and content format

4. **Integration Testing** (if needed)
   - Test all three callbacks together
   - Run full performance monitoring suite
   - Generate comprehensive performance report

---

## 💡 Success Metrics

### Phase 3 Success Criteria
- ✅ All three monitoring callbacks implemented and tested
- ✅ Comprehensive performance visibility achieved
- ✅ Real-time bottleneck detection working
- ✅ Resource anomaly alerting functional
- ✅ Chrome trace visualization available
- ✅ All changelog entries created following convention

### Combined Pipeline Impact
- ✅ **Complete performance monitoring infrastructure**
- ✅ From basic metrics to advanced profiling
- ✅ Production-ready monitoring and alerting
- ✅ Data-driven performance optimization capabilities

---

## 🎯 Next Actions

**Immediate (Do Now):**
1. Choose delegation strategy (A, B, or C)
2. Start delegating tasks to Qwen
3. Monitor progress and validate each implementation
4. Ensure changelog entries are created

**After All Phase 3 Tasks Complete:**
1. Run comprehensive performance monitoring test
2. Generate final performance optimization report
3. Document complete monitoring pipeline
4. Plan production deployment

---

## 📚 References

- **Prompt Index:** [prompts/qwen/INDEX.md](../../../prompts/qwen/INDEX.md)
- **Previous Phase:** [07_prompts-ready.md](./07_prompts-ready.md)
- **Changelog Convention:** [05_changelog/README.md](../05_changelog/README.md)
- **Performance Plan:** [performance_optimization_plan.md](./performance_optimization_plan.md)

---

**Status:** ✅ Ready for delegation
**Last Updated:** 2025-10-07
**Created by:** Claude Code (AI Agent)
