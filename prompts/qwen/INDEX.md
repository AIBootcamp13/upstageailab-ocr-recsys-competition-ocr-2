# Qwen Coder Prompt Index

This index catalogs all Qwen prompts created for the performance optimization project. Use this to quickly find and delegate tasks.

---

## 📋 Prompt Catalog

### Phase 1: Foundation & Monitoring

#### ✅ 01. Performance Profiler Callback
- **File:** `01_performance_profiler_callback.md`
- **Status:** ✅ COMPLETED (2025-10-07)
- **Task:** Task 1.1
- **Priority:** HIGH
- **Estimated Effort:** 2 hours
- **Actual Effort:** 5 min (Qwen) + 15 min (validation)
- **Outcome:** Production-ready callback, 5/5 tests passed
- **Deliverables:**
  - `ocr/lightning_modules/callbacks/performance_profiler.py` ✅
  - `configs/callbacks/performance_profiler.yaml` ✅ (Claude)
  - `tests/integration/test_performance_profiler.py` ✅ (Claude)

---

#### 🟡 02. Baseline Report Generator
- **File:** `02_baseline_report_generator.md`
- **Status:** 🟡 READY TO DELEGATE
- **Task:** Task 1.2
- **Priority:** HIGH
- **Estimated Effort:** 2-3 hours
- **Dependencies:** None (can run independently)
- **Description:** Script to fetch WandB metrics and generate baseline performance report
- **Deliverables:**
  - `scripts/performance/generate_baseline_report.py`
  - Markdown report with bottleneck analysis
  - JSON export of raw metrics

**How to Delegate:**
```bash
cat prompts/qwen/02_baseline_report_generator.md | qwen --yolo
```

**Post-Delegation:**
1. Run validation commands
2. Test with actual WandB run
3. Verify markdown report format
4. Update delegation log

---

#### 🟡 03. Performance Regression Tests
- **File:** `03_performance_regression_tests.md`
- **Status:** 🟡 READY TO DELEGATE
- **Task:** Task 1.3
- **Priority:** MEDIUM
- **Estimated Effort:** 2-3 hours
- **Dependencies:** Baseline report (Task 1.2) for full functionality
- **Description:** pytest suite to catch performance regressions in CI
- **Deliverables:**
  - `tests/performance/test_regression.py`
  - `tests/performance/baselines/thresholds.yaml`
  - `.github/workflows/performance-regression.yml`

**How to Delegate:**
```bash
cat prompts/qwen/03_performance_regression_tests.md | qwen --yolo
```

**Post-Delegation:**
1. Run regression tests locally
2. Verify threshold enforcement
3. Test CI workflow (optional)
4. Update delegation log

---

### Phase 2: PyClipper Caching (Critical Path)

#### 🔴 04. PolygonCache Implementation
- **File:** `04_polygon_cache_implementation.md`
- **Status:** 🔴 READY TO DELEGATE (HIGH PRIORITY)
- **Task:** Task 2.1
- **Priority:** **CRITICAL** (This is the main bottleneck)
- **Estimated Effort:** 3-4 hours
- **Expected Impact:** **5-8x validation speedup**
- **Approach:** Test-Driven Development (TDD)
- **Description:** Implement LRU cache for PyClipper polygon processing
- **Deliverables:**
  - `ocr/datasets/polygon_cache.py`
  - Updated `tests/performance/test_polygon_caching.py` (remove skips)

**How to Delegate:**
```bash
cat prompts/qwen/04_polygon_cache_implementation.md | qwen --yolo
```

**Critical Success Criteria:**
- ✅ All TDD tests pass
- ✅ Performance test shows >10x speedup
- ✅ Cache hit rate >80% after warmup
- ✅ Zero accuracy loss (bit-exact results)

**Post-Delegation:**
1. Run TDD tests: `uv run pytest tests/performance/test_polygon_caching.py -v`
2. Verify performance improvement
3. Check memory usage
4. Prepare for integration (Task 2.2)

---

## 🎯 Delegation Strategy

### Recommended Order

**Week 1: Foundation (Tasks 1.1-1.3)**
1. ✅ Task 1.1: Performance Profiler - **DONE**
2. 🟡 Task 1.2: Baseline Report - **Delegate next**
3. 🟡 Task 1.3: Regression Tests - **Delegate after 1.2**

**Week 2: Critical Path (Task 2.1)**
4. 🔴 Task 2.1: PolygonCache - **High priority, biggest impact**

### Batch Delegation

To delegate multiple tasks in sequence:

```bash
# Delegate all Week 1 tasks
for task in 02 03; do
    echo "🤖 Delegating Task ${task}..."
    cat prompts/qwen/${task}_*.md | qwen --yolo
    echo "✅ Task ${task} complete\n"
done

# Then delegate the critical PolygonCache
cat prompts/qwen/04_polygon_cache_implementation.md | qwen --yolo
```

### Parallel Delegation

Tasks 02 and 03 can be delegated in parallel (no dependencies):

```bash
# Terminal 1
cat prompts/qwen/02_baseline_report_generator.md | qwen --yolo

# Terminal 2
cat prompts/qwen/03_performance_regression_tests.md | qwen --yolo
```

---

## 📊 Progress Tracking

| Task | Prompt | Status | Priority | Impact | Effort |
|------|--------|--------|----------|--------|--------|
| 1.1 | 01_performance_profiler_callback.md | ✅ Done | HIGH | Monitoring | 20 min |
| 1.2 | 02_baseline_report_generator.md | 🟡 Ready | HIGH | Analysis | 2-3 hrs |
| 1.3 | 03_performance_regression_tests.md | 🟡 Ready | MED | Safety | 2-3 hrs |
| 2.1 | 04_polygon_cache_implementation.md | 🔴 Ready | **CRITICAL** | **5-8x speedup** | 3-4 hrs |

**Legend:**
- ✅ Done - Completed and validated
- 🟡 Ready - Prompt complete, ready to delegate
- 🔴 Ready - High priority, ready to delegate

---

## 🔄 Workflow

### For Each Prompt:

**1. Delegation**
```bash
cat prompts/qwen/<prompt_file>.md | qwen --yolo
```

**2. Validation**
```bash
# Type checking
uv run mypy <output_files>

# Linting
uv run ruff check <output_files>

# Tests
uv run pytest <test_files> -v
```

**3. Documentation**
- Update `qwen_delegation_log.md`
- Mark task complete in this index
- Create completion summary (optional)

**4. Integration** (if needed)
- Test with real data
- Create Hydra configs
- Update imports

---

## 📈 Expected Outcomes

### Phase 1 Complete (Tasks 1.1-1.3)
- ✅ Performance monitoring infrastructure
- ✅ Baseline report documenting bottlenecks
- ✅ Regression tests in CI
- **Time Investment:** ~6-8 hours total
- **Value:** Visibility and safety

### Phase 2.1 Complete (PolygonCache)
- ✅ 5-8x validation speedup
- ✅ Validation time <2x training time
- ✅ Cache hit rate >80%
- **Time Investment:** ~4-5 hours
- **Value:** Massive performance improvement

---

## 🚀 Quick Start

**Option 1: Sequential Delegation** (Safest)
```bash
# Task 1.2
cat prompts/qwen/02_baseline_report_generator.md | qwen --yolo
# Validate, then:

# Task 1.3
cat prompts/qwen/03_performance_regression_tests.md | qwen --yolo
# Validate, then:

# Task 2.1 (Critical)
cat prompts/qwen/04_polygon_cache_implementation.md | qwen --yolo
# Validate and integrate
```

**Option 2: Parallel Delegation** (Faster)
```bash
# Delegate 1.2 and 1.3 in parallel
# Then delegate 2.1 after validation
```

**Option 3: Focus on Critical Path**
```bash
# Skip ahead to the highest impact task
cat prompts/qwen/04_polygon_cache_implementation.md | qwen --yolo
# This delivers the 5-8x speedup immediately
```

---

## 📝 Notes

- All prompts are self-contained with full context
- Validation commands embedded in each prompt
- Expected effort estimates are for Qwen execution + validation
- Post-delegation integration may require additional time
- Track all completions in `qwen_delegation_log.md`

---

**Last Updated:** 2025-10-07
**Total Prompts:** 4
**Completed:** 1
**Ready to Delegate:** 3
**Success Rate:** 100% (1/1)
