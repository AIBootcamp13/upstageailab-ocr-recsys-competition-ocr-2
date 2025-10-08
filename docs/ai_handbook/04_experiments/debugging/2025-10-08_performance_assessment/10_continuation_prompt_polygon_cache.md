# Polygon Cache Performance Optimization - Continuation Prompt

**Date:** 2025-10-08
**Priority:** HIGH - Performance Issue Requires Resolution
**Context:** See `09_session_handover_polygon_cache.md` for complete background

## Problem Statement

The polygon cache is functional but provides **no performance benefit** and may actually slow down training. Expected 5-8x validation speedup, currently achieves ~10% hit rate with degrading performance.

**Current Metrics:**
- Hit Rate: 9.99% (target: 50-80% for meaningful speedup)
- Performance: Validation speeds drop from 21it/s to 4it/s over epochs
- Cache Size: 426 entries (858MB) with minimal benefit

## Investigation Required

### Phase 1: Performance Profiling (IMMEDIATE NEXT STEP)
**Goal:** Quantify exactly where time is spent in cache operations vs. pyclipper computations

**Tasks:**
1. Add detailed timing instrumentation to `db_collate_fn.py`:
   - Time cache key generation
   - Time cache lookup operations
   - Time pyclipper computations
   - Time cache storage operations

2. Create profiling script to measure overhead:
   - Run validation batches with/without cache
   - Compare per-batch processing times
   - Identify bottleneck operations

3. Analyze cache hit/miss patterns:
   - Which polygons are being cached/retrieved
   - Frequency of cache hits for different polygon types
   - Cache key collision analysis

### Phase 2: Dataset Analysis
**Goal:** Determine if dataset has sufficient polygon repetition for effective caching

**Tasks:**
1. Analyze polygon patterns in training/validation data:
   - Count unique polygon configurations
   - Measure repetition frequency
   - Identify common polygon shapes/sizes

2. Statistical analysis:
   - Cache hit potential estimation
   - Optimal cache size calculation
   - Polygon similarity analysis

### Phase 3: Cache Strategy Optimization
**Goal:** Reduce overhead while maintaining/maximizing hit rate

**Tasks:**
1. Alternative cache key strategies:
   - Polygon clustering/grouping
   - Approximate matching for similar polygons
   - Hierarchical caching (coarse → fine)

2. Cache policy optimization:
   - Different eviction strategies (LFU vs LRU)
   - Cache size tuning based on dataset analysis
   - Memory vs disk persistence trade-offs

3. Implementation alternatives:
   - Per-image caching instead of per-polygon
   - Lazy loading strategies
   - Background cache warming

## Technical Requirements

### Code Access Needed
- `ocr/datasets/polygon_cache.py` - Core cache implementation
- `ocr/datasets/db_collate_fn.py` - Cache integration point
- `configs/cache_performance_test.yaml` - Test configuration
- `.cache/polygon_cache/polygon_cache.pkl` - Active cache file (858MB)

### Testing Framework
- Use `cache_performance_test.yaml` config for consistent testing
- Run multi-epoch training (3-5 epochs) to see cache buildup
- Compare with/without cache using same random seed
- Measure both throughput and cache statistics

### Success Criteria
1. **Performance Gain:** Cache provides measurable speedup (>10% improvement)
2. **No Regression:** Cache doesn't slow down training when disabled
3. **Scalability:** Performance benefits increase with dataset size
4. **Maintainability:** Cache logic is well-documented and testable

## Investigation Guidelines

### Profiling Approach
```python
# Add to db_collate_fn.py for timing
import time

start_time = time.time()
# ... cache operation ...
cache_time = time.time() - start_time

start_time = time.time()
# ... pyclipper operation ...
pyclipper_time = time.time() - start_time

# Log comparison
logger.info(f"Cache: {cache_time:.4f}s, Pyclipper: {pyclipper_time:.4f}s")
```

### Data Analysis Approach
```python
# Analyze cache keys and hit patterns
cache_keys = list(cache._cache.keys())
polygon_patterns = [key.split('|')[0] for key in cache_keys]  # Extract polygon hash
pattern_counts = Counter(polygon_patterns)
print(f"Unique polygon patterns: {len(pattern_counts)}")
```

### Optimization Strategy
1. **Measure First:** Establish baseline performance metrics
2. **Identify Bottlenecks:** Find where time is actually spent
3. **Hypothesis Testing:** Try changes one at a time
4. **Validate Impact:** Measure performance impact of each change
5. **Iterate:** Use data to guide next optimization steps

## Expected Deliverables

### Immediate (Phase 1)
- Profiling instrumentation added to collate function
- Performance comparison script
- Detailed timing breakdown report

### Short-term (Phase 2)
- Dataset polygon pattern analysis
- Cache hit potential assessment
- Recommendations for cache strategy

### Long-term (Phase 3)
- Optimized cache implementation
- Performance validation with multiple datasets
- Documentation of findings and recommendations

## Risk Mitigation

### Fallback Options
1. **Disable Cache:** Remove overhead if no benefit found
2. **Conditional Caching:** Enable only for large datasets
3. **Alternative Optimization:** Focus on other performance bottlenecks

### Validation Checks
- **Correctness:** Ensure cached results match computed results
- **Memory:** Monitor memory usage with cache enabled
- **Disk I/O:** Assess disk persistence impact on training

## Communication Requirements

### Progress Updates
- Daily status updates on investigation progress
- Key findings and insights gained
- Blockers or resource needs identified

### Documentation
- All code changes documented with rationale
- Performance measurements recorded
- Analysis results summarized for future reference

---

**Next Action:** Begin with Phase 1 profiling. Add timing instrumentation and run comparative performance tests.
