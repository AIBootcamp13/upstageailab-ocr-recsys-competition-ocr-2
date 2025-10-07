# Qwen Prompts Ready for Delegation

**Date:** 2025-10-07
**Status:** ✅ All prompts created and validated
**Total Prompts:** 4 (1 completed, 3 ready)

---

## 🎉 Summary

Successfully created **3 comprehensive Qwen prompts** for the performance optimization project. All prompts are self-contained, tested, and ready for immediate delegation.

### Completed Work
- ✅ Created 3 new Qwen prompts (Tasks 1.2, 1.3, 2.1)
- ✅ Each prompt is 200-300 lines with full context
- ✅ Validation commands embedded in each prompt
- ✅ Created prompt index and delegation guide
- ✅ Updated delegation log
- ✅ Documented expected outcomes

### Time Investment
- **Prompt Creation:** ~90 minutes
- **Documentation:** ~30 minutes
- **Total:** ~2 hours

---

## 📋 Prompts Created

### 1. ✅ Performance Profiler Callback (COMPLETED)
- **File:** `prompts/qwen/01_performance_profiler_callback.md`
- **Status:** Done ✅
- **Outcome:** Production-ready, 5/5 tests passed

### 2. 🟡 Baseline Report Generator (READY)
- **File:** `prompts/qwen/02_baseline_report_generator.md`
- **Lines:** 250+
- **Task:** Generate performance baseline report from WandB
- **Impact:** Visibility into bottlenecks
- **Effort:** 2-3 hours

### 3. 🟡 Performance Regression Tests (READY)
- **File:** `prompts/qwen/03_performance_regression_tests.md`
- **Lines:** 280+
- **Task:** Create pytest suite for performance regression
- **Impact:** Safety net against regressions
- **Effort:** 2-3 hours

### 4. 🔴 PolygonCache Implementation (READY - CRITICAL)
- **File:** `prompts/qwen/04_polygon_cache_implementation.md`
- **Lines:** 300+
- **Task:** Implement LRU cache for PyClipper (TDD)
- **Impact:** **5-8x validation speedup** 🚀
- **Effort:** 3-4 hours

---

## 🚀 How to Delegate

### Quick Start (Recommended Order)

**Step 1: Baseline Report Generator**
```bash
cat prompts/qwen/02_baseline_report_generator.md | qwen --yolo
```
Then validate:
```bash
uv run mypy scripts/performance/generate_baseline_report.py
uv run ruff check scripts/performance/generate_baseline_report.py
uv run python scripts/performance/generate_baseline_report.py --help
```

**Step 2: Performance Regression Tests**
```bash
cat prompts/qwen/03_performance_regression_tests.md | qwen --yolo
```
Then validate:
```bash
uv run pytest tests/performance/test_regression.py -v
```

**Step 3: PolygonCache (CRITICAL - Biggest Impact)**
```bash
cat prompts/qwen/04_polygon_cache_implementation.md | qwen --yolo
```
Then validate:
```bash
uv run pytest tests/performance/test_polygon_caching.py -v
```

### Alternative: Parallel Delegation

Tasks 2 and 3 can run in parallel (no dependencies):
```bash
# Terminal 1
cat prompts/qwen/02_baseline_report_generator.md | qwen --yolo

# Terminal 2
cat prompts/qwen/03_performance_regression_tests.md | qwen --yolo
```

### Alternative: Focus on Critical Path

Skip straight to the highest impact task:
```bash
# This delivers the 5-8x speedup immediately
cat prompts/qwen/04_polygon_cache_implementation.md | qwen --yolo
```

---

## 📊 Expected Outcomes

### After Task 1.2 (Baseline Report)
- ✅ Script to generate performance reports
- ✅ Markdown report documenting bottlenecks
- ✅ JSON export of metrics
- **Time:** ~3 hours (Qwen + validation)

### After Task 1.3 (Regression Tests)
- ✅ pytest suite for performance validation
- ✅ CI workflow to catch regressions
- ✅ Configurable SLO thresholds
- **Time:** ~3 hours (Qwen + validation)

### After Task 2.1 (PolygonCache) 🎯
- ✅ **5-8x validation speedup**
- ✅ Validation time <2x training
- ✅ Cache hit rate >80%
- ✅ Zero accuracy loss
- **Time:** ~4 hours (Qwen + validation + integration)
- **Impact:** MASSIVE performance improvement

---

## 🎯 Recommended Strategy

### Strategy A: Sequential (Safest)
**Timeline:** ~10 hours total

1. Delegate Task 1.2 → Validate → ✅
2. Delegate Task 1.3 → Validate → ✅
3. Delegate Task 2.1 → Validate → ✅ (5-8x speedup!)

**Pros:** Methodical, lower risk
**Cons:** Takes longer

### Strategy B: Parallel + Critical (Faster) ⭐ RECOMMENDED
**Timeline:** ~6 hours total

1. Delegate Tasks 1.2 & 1.3 in parallel → Validate → ✅
2. Delegate Task 2.1 → Validate → ✅ (5-8x speedup!)

**Pros:** Faster, still safe
**Cons:** Need to manage two Qwen sessions

### Strategy C: Critical Path First (Highest Impact)
**Timeline:** ~4 hours to first win

1. Delegate Task 2.1 (PolygonCache) → Validate → ✅ (5-8x speedup!)
2. Then delegate 1.2 & 1.3 for monitoring/safety

**Pros:** Fastest time to value
**Cons:** No baseline first (but can create later)

---

## 📝 Validation Checklist

After each Qwen delegation:

**For Every Task:**
- [ ] Run type checking: `uv run mypy <files>`
- [ ] Run linting: `uv run ruff check <files>`
- [ ] Run tests: `uv run pytest <test_files> -v`
- [ ] Verify imports work
- [ ] Update delegation log

**Task-Specific:**
- [ ] Task 1.2: Test with real WandB run
- [ ] Task 1.3: Verify threshold enforcement
- [ ] Task 2.1: Confirm >10x performance improvement

---

## 📁 File Organization

```
prompts/qwen/
├── README.md                              # Delegation guidelines
├── INDEX.md                               # Catalog of all prompts
├── 01_performance_profiler_callback.md    # ✅ Completed
├── 02_baseline_report_generator.md        # 🟡 Ready
├── 03_performance_regression_tests.md     # 🟡 Ready
└── 04_polygon_cache_implementation.md     # 🔴 Ready (CRITICAL)

docs/ai_handbook/07_project_management/
├── performance_optimization_plan.md       # Original plan
├── performance_optimization_execution_plan.md  # Detailed execution
├── qwen_delegation_log.md                 # Delegation tracking
├── qwen_prompts_ready.md                  # This file
└── task_1.1_completion_summary.md         # Completed task summary
```

---

## 🔄 Post-Delegation Workflow

1. **Delegate to Qwen**
   ```bash
   cat prompts/qwen/<task>.md | qwen --yolo
   ```

2. **Validate Implementation**
   - Run all commands in prompt's "Validation" section
   - Confirm all tests pass
   - Check code quality (mypy, ruff)

3. **Update Documentation**
   - Mark task complete in `qwen_delegation_log.md`
   - Update `INDEX.md` status
   - Create completion summary (optional)

4. **Integration** (if needed)
   - Test with real data
   - Create Hydra configs
   - Run end-to-end tests

---

## 💡 Success Metrics

### Phase 1 Success (Tasks 1.2-1.3)
- ✅ Performance baseline documented
- ✅ Regression tests in CI
- ✅ Clear visibility into bottlenecks

### Phase 2 Success (Task 2.1)
- ✅ **5-8x validation speedup** 🚀
- ✅ Validation time <2x training
- ✅ Cache hit rate >80%
- ✅ Zero accuracy loss

---

## 🎯 Next Actions

**Immediate (Do Now):**
1. Choose delegation strategy (A, B, or C)
2. Start with first task delegation
3. Validate and move to next

**After All Delegations Complete:**
1. Run full performance comparison
2. Generate final performance report
3. Document improvements
4. Plan Phase 3 optimizations

---

## 📚 References

- **Prompt Index:** [prompts/qwen/INDEX.md](../../../prompts/qwen/INDEX.md)
- **Delegation Log:** [qwen_delegation_log.md](./qwen_delegation_log.md)
- **Execution Plan:** [performance_optimization_execution_plan.md](./performance_optimization_execution_plan.md)
- **Original Plan:** [performance_optimization_plan.md](./performance_optimization_plan.md)

---

**Status:** ✅ Ready for delegation
**Last Updated:** 2025-10-07
**Created by:** Claude Code (AI Agent)
