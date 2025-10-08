# Continuation Prompt: Polygon Cache Debugging

**Date:** 2025-10-08
**Context:** Performance Assessment Session Complete
**Next Task:** Debug 100% cache miss rate in polygon caching

## Session Context

### Previous Work Completed ✅
- **Performance Assessment:** Determined polygon cache adds 10.6% overhead with 0% hit rate
- **Configuration Fixes:** Resolved DataLoader params and coordinate mismatches
- **Testing Infrastructure:** Created comprehensive performance testing framework
- **Documentation:** Complete session summary and debug results

### Current Problem Statement
**Issue:** Polygon caching shows 100% cache misses (0% hits) despite being expected to provide 5-8x validation speedup.

**Impact:** Performance optimization that should accelerate training is instead adding overhead.

**Expected Behavior:**
- Cache stores computed polygon transformations
- Repeated validation runs should hit cache
- Major performance improvement observed

**Actual Behavior:**
- Cache appears functional but never accessed
- 100% miss rate on all cache lookups
- 10.6% performance overhead instead of speedup

## Technical Context

### Cache Implementation
**Location:** `ocr/data/collate_fn.py` (DBCollateFN class)
**Configuration:** `configs/data/base.yaml` (currently `polygon_cache.enabled: false`)
**Purpose:** Cache polygon computations to speed up validation

### Code Structure
```python
class DBCollateFN:
    def __init__(self, config):
        self.polygon_cache = PolygonCache(config.polygon_cache)

    def __call__(self, batch):
        for sample in batch:
            cache_key = self._generate_cache_key(sample['image_path'], sample['polygons'])
            if cache_key in self.polygon_cache:
                # CACHE HIT - Expected but not happening
                result = self.polygon_cache[cache_key]
            else:
                # CACHE MISS - Always happening
                result = self._compute_polygons(sample)
                self.polygon_cache[cache_key] = result
```

### Performance Data
```
Test Results (1 epoch, limited batches):
- Baseline:           64.11s
- Polygon Cache:      70.89s (+10.6% overhead)
- Cache Hit Rate:     0% (100% misses)
- Expected Hit Rate:  >50% for repeated runs
```

## Investigation Framework

### Primary Hypothesis
**Cache keys are not matching between storage and retrieval.**

### Debug Steps Required

1. **Enable Detailed Logging**
   ```python
   # Add to DBCollateFN
   logging.info(f"Cache key generated: {cache_key}")
   logging.info(f"Cache contents: {list(self.polygon_cache.keys())}")
   logging.info(f"Cache hit: {cache_key in self.polygon_cache}")
   ```

2. **Inspect Cache Keys**
   - What keys are being generated?
   - What keys are stored in cache?
   - Do they match exactly?

3. **Verify Key Generation**
   ```python
   def _generate_cache_key(self, image_path, polygons):
       # Current implementation - verify deterministic
       key_data = (image_path, tuple(polygons.flatten()))
       return hashlib.md5(str(key_data).encode()).hexdigest()
   ```

4. **Test Cache Operations**
   - Manually test cache storage/retrieval
   - Verify cache persistence across batches
   - Check for race conditions or threading issues

### Expected Debug Findings

**Possible Root Causes:**
- **Non-deterministic keys:** Cache keys not consistent between runs
- **Path differences:** Image paths not normalized (relative vs absolute)
- **Data format changes:** Polygon data format changes between storage/retrieval
- **Cache invalidation:** Cache being cleared or not persisting
- **Threading issues:** Multi-worker DataLoader interfering with cache

**Success Indicators:**
- Cache hit rate > 50%
- Performance improvement > 5%
- Consistent cache key generation

## Available Resources

### Testing Tools
- **Enable Cache:** `uv run python runners/train.py --config-name=performance_test data.polygon_cache.enabled=true`
- **Performance Test:** `python scripts/performance_measurement.py`
- **Quick Validation:** `python scripts/quick_performance_validation.py`

### Code Locations
- **Cache Implementation:** `ocr/data/collate_fn.py`
- **Cache Config:** `configs/data/base.yaml`
- **Test Config:** `configs/performance_test.yaml`

### Documentation
- **Session Summary:** `docs/ai_handbook/04_experiments/2025-10-08_performance_assessment/00_session_summary.md`
- **Debug Results:** `docs/ai_handbook/04_experiments/2025-10-08_performance_assessment/05_debug_results.md`

## Continuation Instructions

### Immediate Next Steps
1. **Enable cache with logging** to see what's happening
2. **Inspect cache keys** generated vs stored
3. **Verify key generation determinism**
4. **Test manual cache operations**

### Debug Approach
- Start with detailed logging to understand cache behavior
- Compare cache keys between storage and retrieval attempts
- Use single-threaded execution to eliminate threading complications
- Focus on one sample case to understand the issue completely

## Debugging Artifacts & Logging Organization

### Artifact Creation Guidelines

**1. Create Debug Session Folder**
```
docs/ai_handbook/04_experiments/YYYY-MM-DD_debug_session_name/
├── 00_debug_session_summary.md
├── 01_debug_findings.md
├── 02_code_changes.md
├── 03_test_results.md
├── artifacts/
│   ├── cache_keys_sample.json
│   ├── performance_logs_YYYY-MM-DD_HH-MM-SS.log
│   ├── memory_dumps/
│   └── cache_inspection_data/
└── scripts/
    ├── debug_cache_keys.py
    ├── reproduce_issue.py
    └── validate_fix.py
```

**2. Naming Convention for Artifacts**
- **Logs:** `performance_logs_YYYY-MM-DD_HH-MM-SS.log`
- **Cache Data:** `cache_keys_sample_batch_N.json`
- **Memory Dumps:** `memory_snapshot_epoch_N_batch_M.pkl`
- **Test Results:** `validation_results_config_X.json`

**3. Rolling Log Management**
```bash
# Create timestamped log directory
LOG_DIR="artifacts/logs/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$LOG_DIR"

# Enable detailed logging
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LOG_LEVEL=DEBUG
export LOG_FILE="$LOG_DIR/debug_session.log"

# Run with logging
uv run python runners/train.py ... 2>&1 | tee "$LOG_DIR/console_output.log"
```

**4. Cache Inspection Artifacts**
```python
# Save cache state for analysis
def save_cache_inspection_data(cache, batch_idx, epoch):
    inspection_data = {
        'timestamp': datetime.now().isoformat(),
        'epoch': epoch,
        'batch_idx': batch_idx,
        'cache_keys': list(cache.keys())[:10],  # Sample first 10 keys
        'cache_size': len(cache),
        'memory_usage': get_memory_usage()
    }

    filename = f"artifacts/cache_inspection_epoch_{epoch}_batch_{batch_idx}.json"
    with open(filename, 'w') as f:
        json.dump(inspection_data, f, indent=2, default=str)
```

**5. Performance Profiling Data**
```python
# Save profiling data
def save_performance_profile(profiler, epoch, batch_idx):
    profile_data = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'total_time': profiler.total_time,
        'cache_hits': profiler.cache_hits,
        'cache_misses': profiler.cache_misses,
        'memory_peak': profiler.memory_peak
    }

    filename = f"artifacts/performance_profile_epoch_{epoch}_batch_{batch_idx}.json"
    with open(filename, 'w') as f:
        json.dump(profile_data, f, indent=2)
```

### Artifact Organization Workflow

**During Debugging:**
1. Create timestamped artifact folders immediately when starting debug session
2. Save cache keys, memory snapshots, and performance data at regular intervals
3. Maintain rolling logs with timestamps for all test runs
4. Document each artifact's purpose and contents in session summary

**Post-Debug Cleanup:**
1. Move relevant artifacts to permanent storage locations
2. Archive debug logs older than 7 days
3. Update artifact index in session documentation
4. Remove temporary debug files that are no longer needed

### Success Criteria
- **Functional:** Cache hit rate > 50% for repeated validation
- **Performance:** Measurable speedup (>5% improvement)
- **Correctness:** No change in validation metrics accuracy

### Communication
- Document all findings and attempted solutions
- Update cache implementation with fixes
- Provide clear summary of root cause and resolution

---

**Continuation Prepared By:** AI Assistant
**Debug Priority:** High (Major expected performance improvement)
**Estimated Debug Time:** 2-4 hours
**Expected Outcome:** Functional polygon cache with performance benefits</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/ai_handbook/04_experiments/2025-10-08_performance_assessment/07_continuation_prompt_polygon_cache.md
