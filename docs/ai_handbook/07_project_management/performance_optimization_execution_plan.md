# Performance Optimization Execution Plan
**Date:** 2025-10-07
**Status:** In Planning
**Owner:** AI Agent Collaboration
**Related:** [performance_optimization_plan.md](./performance_optimization_plan.md)

## Executive Summary

This document provides a **concrete, actionable execution plan** for implementing the performance optimizations outlined in the Performance Optimization Plan. It breaks down each phase into specific tasks, dependencies, validation criteria, and rollback procedures.

## Planning Context

### Current State Analysis
- **Codebase Size:** 200,000+ lines of code
- **Documentation:** 200,000+ lines
- **Known Bottleneck:** PyClipper polygon processing in validation (10x slower than training)
- **Current Recall:** 0.90 (improved from 0.75)
- **Performance Tests:** Already exist at `tests/performance/test_polygon_caching.py`
- **TDD Approach:** Tests written, implementation pending

### Strategic Priorities
1. **Validation Speed:** Critical path - blocks experimentation velocity
2. **Monitoring First:** Need visibility before optimization
3. **Incremental Changes:** Avoid big-bang refactors
4. **Maintain Accuracy:** No regressions in model performance

## Phase 1: Foundation & Monitoring (Week 1)

### Objective
Establish performance baselines and monitoring infrastructure before making any optimizations.

### Task 1.1: Performance Profiling Infrastructure
**Priority:** HIGH
**Effort:** 2 days
**Assignee:** AI Agent + Human Review

#### Implementation Steps
1. Create profiling callback for PyTorch Lightning
   ```bash
   # File to create: ocr/lightning_modules/callbacks/performance_profiler.py
   ```

2. Add profiling configuration
   ```yaml
   # File to create: configs/callbacks/performance_profiler.yaml
   ```

3. Integrate with training pipeline
   ```python
   # Modify: ocr/lightning_modules/ocr_pl.py
   # Add performance profiler callback
   ```

#### Validation Criteria
- [ ] Profiler captures validation batch timing
- [ ] PyClipper operation times logged to WandB
- [ ] Memory usage tracked per epoch
- [ ] Baseline report generated for current validation speed

#### Rollback Procedure
```bash
git revert <commit-hash>
# Profiling is additive only - no risk to existing functionality
```

### Task 1.2: Benchmark Current Performance
**Priority:** HIGH
**Effort:** 1 day
**Assignee:** AI Agent

#### Implementation Steps
1. Run profiling on validation set
   ```bash
   uv run python runners/test.py preset=example \
     checkpoint_path="outputs/ocr_training/checkpoints/best.ckpt" \
     callbacks.performance_profiler.enabled=true
   ```

2. Generate performance report
   ```bash
   uv run python scripts/performance/generate_baseline_report.py \
     --run-id <wandb_run_id> \
     --output docs/performance/baseline_2025-10-07.md
   ```

3. Document bottlenecks
   - Identify top 5 slowest operations
   - Measure time spent in PyClipper
   - Profile memory allocation patterns

#### Success Metrics
- [ ] Baseline validation time documented (target: measure 10x slowdown)
- [ ] PyClipper time % of total identified
- [ ] Memory high-water mark recorded
- [ ] Report includes actionable insights

### Task 1.3: Set Up Regression Testing
**Priority:** MEDIUM
**Effort:** 1 day
**Assignee:** AI Agent

#### Implementation Steps
1. Create performance regression test suite
   ```bash
   # File: tests/performance/test_regression.py
   ```

2. Add CI integration
   ```yaml
   # File: .github/workflows/performance-regression.yml
   ```

3. Define performance SLOs (Service Level Objectives)
   ```python
   # Validation time must not exceed baseline by >10%
   # Memory usage must stay under 80% of available
   # Cache hit rate must be >80% after warmup
   ```

#### Validation Criteria
- [ ] Regression tests pass with current baseline
- [ ] CI fails if validation time increases >10%
- [ ] Performance metrics automatically reported in PRs

---

## Phase 2: PyClipper Caching Implementation (Week 2)

### Objective
Implement polygon processing cache to reduce validation time from 10x to <2x training time.

### Task 2.1: Implement PolygonCache Class
**Priority:** HIGH
**Effort:** 2 days
**Assignee:** AI Agent (following TDD)

#### Implementation Steps
1. Implement `PolygonCache` to pass existing tests
   ```bash
   # File to create: ocr/datasets/polygon_cache.py
   ```

2. Follow TDD workflow:
   ```bash
   # Run failing tests
   uv run pytest tests/performance/test_polygon_caching.py -v

   # Implement minimal code to pass tests
   # Iterate until all tests pass
   ```

3. Add cache key generation
   - Hash polygon geometry + image dimensions
   - Use deterministic hashing (not object id)
   - Include collate_fn parameters (shrink_ratio, etc.)

#### Implementation Design
```python
class PolygonCache:
    """LRU cache for polygon processing results."""

    def __init__(self, max_size=1000, persist_to_disk=False):
        self.cache = {}  # Start simple, optimize later
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0

    def _generate_key(self, polygons, image_shape, params):
        """Generate deterministic hash from polygon geometry."""
        # Normalize polygons to canonical form
        # Hash using xxhash or hashlib for speed
        pass

    def get(self, key):
        """Retrieve from cache, track hit/miss."""
        pass

    def set(self, key, value):
        """Store with LRU eviction."""
        pass
```

#### Validation Criteria
- [ ] All tests in `test_polygon_caching.py` pass
- [ ] Cache hit rate >80% on validation set (second epoch)
- [ ] Memory usage <200MB for cache with max_size=1000
- [ ] No accuracy regression (prob_maps identical)

#### Rollback Procedure
```bash
# Caching is opt-in via config
configs/callbacks/polygon_cache.yaml:
  enabled: false  # Disable if issues occur
```

### Task 2.2: Integrate Cache with DBCollateFN
**Priority:** HIGH
**Effort:** 1 day
**Assignee:** AI Agent

#### Implementation Steps
1. Modify `DBCollateFN` to use cache
   ```python
   # File: ocr/datasets/db_collate_fn.py

   def __init__(self, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7,
                cache=None):
       self.cache = cache
       # ...

   def make_prob_thresh_map(self, image, polygons, filename):
       if self.cache is not None:
           key = self.cache._generate_key(polygons, image.shape,
                                          (self.shrink_ratio, self.thresh_min, self.thresh_max))
           cached = self.cache.get(key)
           if cached is not None:
               return cached

       # Original processing logic
       result = self._process_polygons(image, polygons, filename)

       if self.cache is not None:
           self.cache.set(key, result)

       return result
   ```

2. Add cache instantiation to dataset
   ```python
   # File: ocr/datasets/base.py
   from ocr.datasets.polygon_cache import PolygonCache

   def __init__(self, ...):
       self.polygon_cache = PolygonCache(max_size=1000) if use_cache else None
       self.collate_fn = DBCollateFN(cache=self.polygon_cache)
   ```

3. Add Hydra configuration
   ```yaml
   # File: configs/data/base.yaml
   polygon_cache:
     enabled: true
     max_size: 1000
     persist_to_disk: false
   ```

#### Validation Criteria
- [ ] Validation time reduced by 5-8x
- [ ] Cache hit rate logged to WandB
- [ ] No changes to prob_map outputs (bit-exact)
- [ ] Memory usage within budget

### Task 2.3: Performance Validation
**Priority:** HIGH
**Effort:** 1 day
**Assignee:** AI Agent

#### Implementation Steps
1. Run A/B comparison test
   ```bash
   # Without cache
   uv run python runners/test.py preset=example \
     data.polygon_cache.enabled=false \
     experiment_tag=no_cache

   # With cache
   uv run python runners/test.py preset=example \
     data.polygon_cache.enabled=true \
     experiment_tag=with_cache
   ```

2. Compare metrics
   ```bash
   uv run python scripts/performance/compare_runs.py \
     --baseline <no_cache_run_id> \
     --optimized <with_cache_run_id> \
     --output docs/performance/cache_comparison.md
   ```

3. Verify accuracy preservation
   ```bash
   uv run pytest tests/integration/test_validation_accuracy.py
   ```

#### Success Criteria
- [ ] Validation time: <2x training time (target: 5-8x speedup)
- [ ] H-mean: No change from baseline
- [ ] Precision/Recall: Within 0.001 of baseline
- [ ] Memory increase: <10% of baseline

---

## Phase 3: Memory Optimization (Week 3-4)

### Objective
Reduce memory footprint to enable larger batch sizes.

### Task 3.1: Optimize Cache Memory Usage
**Priority:** MEDIUM
**Effort:** 2 days
**Assignee:** AI Agent

#### Implementation Steps
1. Implement disk-backed cache (optional)
   ```python
   # Use memory-mapped files for large caches
   import mmap
   import pickle
   ```

2. Add cache compression
   ```python
   # Compress prob_maps using numpy's savez_compressed
   ```

3. Implement smart eviction policy
   ```python
   # Prioritize keeping frequent validation samples
   # Evict large, infrequent items first
   ```

#### Validation Criteria
- [ ] Cache size reduced by 30-50%
- [ ] Cache hit rate maintained >75%
- [ ] Access latency <10ms

### Task 3.2: Dataset Memory Profiling
**Priority:** MEDIUM
**Effort:** 2 days
**Assignee:** AI Agent

#### Implementation Steps
1. Profile memory usage with memory_profiler
   ```bash
   uv run python -m memory_profiler runners/train.py preset=example trainer.max_epochs=1
   ```

2. Identify top memory consumers
3. Implement lazy loading for large datasets
4. Optimize polygon storage format

#### Success Metrics
- [ ] Memory usage reduced by 20%
- [ ] No performance degradation

---

## Phase 4: Parallel Processing (Week 5-6)

### Objective
Improve throughput through parallel polygon preprocessing.

### Task 4.1: Multiprocess Polygon Preprocessing
**Priority:** MEDIUM
**Effort:** 3 days
**Assignee:** AI Agent

#### Implementation Steps
1. Implement worker pool for polygon processing
   ```python
   from multiprocessing import Pool

   def preprocess_batch(polygons_batch):
       with Pool(processes=4) as pool:
           results = pool.map(process_single_polygon, polygons_batch)
       return results
   ```

2. Add async loading
3. Benchmark parallelization overhead

#### Validation Criteria
- [ ] 2-3x speedup in preprocessing
- [ ] No deadlocks or race conditions
- [ ] Clean worker shutdown

---

## Risk Mitigation Strategy

### High-Risk Changes
1. **PyClipper Caching:** Could introduce subtle bugs in prob_map generation
   - **Mitigation:** Extensive testing, bit-exact validation

2. **Memory Optimization:** Could cause OOM errors
   - **Mitigation:** Gradual rollout, monitoring

3. **Parallel Processing:** Potential for deadlocks
   - **Mitigation:** Thorough testing, timeout mechanisms

### Rollback Plan
All optimizations are **feature-flagged via Hydra configs**:
```yaml
# Disable all optimizations
data:
  polygon_cache:
    enabled: false
  parallel_preprocessing:
    enabled: false
```

### Quality Gates
Before merging any phase:
- [ ] All existing tests pass
- [ ] Performance regression tests pass
- [ ] Memory usage within budget
- [ ] Model accuracy unchanged (H-mean within 0.001)
- [ ] Code review completed
- [ ] Documentation updated

---

## Success Metrics (Overall)

### Performance Targets
| Metric | Baseline | Target | Stretch Goal |
|--------|----------|--------|--------------|
| Validation Time | 10x training | <2x training | <1.5x training |
| Training Throughput | TBD | >100 samples/sec | >150 samples/sec |
| Memory Usage | TBD | <80% available | <70% available |
| Cache Hit Rate | N/A | >80% | >90% |

### Timeline
- **Week 1:** Foundation & Monitoring (Phase 1)
- **Week 2:** PyClipper Caching (Phase 2)
- **Week 3-4:** Memory Optimization (Phase 3)
- **Week 5-6:** Parallel Processing (Phase 4)

### Review Points
- **End of Week 1:** Baseline report review
- **End of Week 2:** Cache performance review
- **End of Week 4:** Memory optimization review
- **End of Week 6:** Final performance audit

---

## Next Steps

### Immediate Actions (Today)
1. **Review this plan** with human collaborator
2. **Start Task 1.1:** Create performance profiler callback
3. **Run baseline profiling:** Document current performance

### This Week
1. Complete Phase 1 (Monitoring infrastructure)
2. Generate baseline performance report
3. Begin Phase 2 (Cache implementation)

### Resources Needed
- Access to GPU for profiling
- WandB project for metrics
- Disk space for cache storage (optional)

---

## References

- **Main Plan:** [performance_optimization_plan.md](./performance_optimization_plan.md)
- **Modular Refactor Protocol:** [02_protocols/05_modular_refactor.md](../02_protocols/05_modular_refactor.md)
- **Training Protocol:** [02_protocols/13_training_protocol.md](../02_protocols/13_training_protocol.md)
- **Existing Tests:** [tests/performance/test_polygon_caching.py](../../tests/performance/test_polygon_caching.py)

---

## Appendix: Command Cheat Sheet

```bash
# Profiling
uv run python runners/test.py preset=example callbacks.performance_profiler.enabled=true

# Baseline benchmarking
uv run pytest tests/performance/ -v --benchmark

# With cache enabled
uv run python runners/test.py preset=example data.polygon_cache.enabled=true

# A/B comparison
uv run python scripts/performance/compare_runs.py --baseline <id1> --optimized <id2>

# Memory profiling
uv run python -m memory_profiler runners/train.py preset=example trainer.max_epochs=1

# Regression testing
uv run pytest tests/performance/test_regression.py -v
```
