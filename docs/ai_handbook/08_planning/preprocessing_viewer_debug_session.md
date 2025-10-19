# Preprocessing Viewer Debug Session Handover

**Date**: 2025-10-18
**Status**: 🔴 CRITICAL - App Hanging on Pipeline Execution
**PID**: 3687430 (134% CPU usage - blocking operation)
**Port**: 8501

## Problem Summary

The Streamlit Preprocessing Viewer app **hangs indefinitely** when running the full pipeline, showing only a spinner with no progress indication or error messages.

### Observable Symptoms

1. **UI Behavior**:
   - App starts successfully
   - Image uploads correctly
   - Full Pipeline tab shows: "Running preprocessing pipeline..." with spinner
   - **Hangs indefinitely** - no timeout, no error, no progress

2. **System Behavior**:
   - Process consuming **134% CPU** (stuck in compute-heavy operation)
   - Memory usage: ~1.1GB (normal for image processing)
   - No crash, no error logs

3. **Last Visible Log**:
   ```
   INFO:ocr.datasets.preprocessing.intelligent_brightness:Initialized IntelligentBrightnessAdjuster with method: auto
   ```

## Root Cause Analysis

### Primary Issue: Document Flattening Performance

**Location**: [pipeline.py:183-197](ui/preprocessing_viewer/pipeline.py:183-197)

```python
# Stage 3: Document flattening
corners_for_processing = results.get("selected_quadrilateral") or results.get("detected_corners")
if config.get("enable_document_flattening", True) and isinstance(corners_for_processing, np.ndarray):
    try:
        flattened_result = self.document_flattener.flatten_document(current_image, corners_for_processing)
        # ...
    except Exception:
        pass
```

**Problem**: Document flattening is **enabled by default** (`True`) and takes **3-15 seconds** per image according to Phase 2 validation results.

**Evidence from Phase 2**:
- Processing time: 0.01s for simple cases, **3-15s for complex cases**
- The operation is CPU-intensive (thin plate spline warping, RBF interpolation)
- No progress indication during processing
- See: [15_phase2_complete.md:101-111](docs/ai_handbook/05_changelog/2025-10/15_phase2_complete.md:101-111)

### Secondary Issues

1. **No Progress Indication**:
   - Streamlit spinner doesn't show granular progress
   - No stage-by-stage progress updates
   - User has no idea which stage is running

2. **No Timeout Mechanism**:
   - Pipeline can run indefinitely
   - No maximum processing time limit
   - No cancellation option

3. **Exception Swallowing**:
   - `except Exception: pass` silently ignores errors
   - No logging of failures
   - Makes debugging nearly impossible

4. **Missing Performance Optimizations**:
   - No caching of intermediate results
   - Reprocesses entire image on each run
   - No resolution downsampling for preview

## Impact Assessment

**Severity**: 🔴 CRITICAL
- App is unusable for full pipeline processing
- Step-by-step visualizer may have same issue
- Blocks user testing and validation
- Degrades user experience significantly

**Affected Components**:
- Full Pipeline tab (completely broken)
- Step-by-Step Visualizer (potentially affected)
- Any workflow using document flattening

## Recommended Fixes (Priority Order)

### 1. **IMMEDIATE FIX: Disable Document Flattening by Default** ⚡

**Action**: Change default config to disable expensive features

**File**: `ui/preprocessing_viewer/preset_manager.py` or config defaults

```python
# Change from:
"enable_document_flattening": True

# To:
"enable_document_flattening": False  # Too slow for interactive use
```

**Rationale**:
- Makes app immediately usable
- User can opt-in for flattening when needed
- Matches "fast preprocessor" pattern from Phase 3

### 2. **SHORT-TERM: Add Progress Indicators** 📊

**Action**: Add stage-by-stage progress updates

**Implementation**:
```python
# In pipeline.py process_with_intermediates()
progress_placeholder = st.empty()
progress_placeholder.text("Stage 1/8: Document detection...")

# Update after each stage:
progress_placeholder.text("Stage 2/8: Perspective correction...")
# etc.
```

**Benefit**: User knows processing is happening and which stage

### 3. **SHORT-TERM: Add Timeout Protection** ⏱️

**Action**: Implement per-stage and total pipeline timeouts

**Implementation**:
```python
import signal
from contextlib import contextmanager

@contextmanager
def timeout_context(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Stage exceeded {seconds}s timeout")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Usage:
try:
    with timeout_context(5):  # 5 second timeout per stage
        flattened_result = self.document_flattener.flatten_document(...)
except TimeoutError as e:
    self.logger.warning(f"Flattening timeout: {e}")
    # Skip this stage
```

### 4. **MEDIUM-TERM: Performance Optimizations** 🚀

**Actions**:
a. **Downsample for Preview**:
```python
# Resize large images for faster processing
MAX_PREVIEW_SIZE = 800
if max(image.shape[:2]) > MAX_PREVIEW_SIZE:
    scale = MAX_PREVIEW_SIZE / max(image.shape[:2])
    preview_image = cv2.resize(image, None, fx=scale, fy=scale)
    # Process preview_image instead
```

b. **Caching**:
```python
# Cache intermediate results
@st.cache_data
def process_with_intermediates_cached(image_hash, config_hash):
    # Process and return results
```

c. **Async Processing**:
```python
# Run expensive operations in background
import asyncio
async def process_stage_async(stage_func, image):
    return await asyncio.to_thread(stage_func, image)
```

### 5. **LONG-TERM: GPU Acceleration** 🎮

**Action**: Implement GPU-accelerated flattening (documented in Phase 3)

**Note**: This is future work, not required for immediate fix

## Implementation Plan

### Phase 1: Emergency Fix (30 minutes)
- [ ] Disable document flattening by default
- [ ] Add warning message when enabling it
- [ ] Test full pipeline runs quickly

### Phase 2: User Experience (2 hours)
- [ ] Add progress indicators for each stage
- [ ] Improve error messages
- [ ] Add stage timing display
- [ ] Show which stage is running

### Phase 3: Robustness (4 hours)
- [ ] Implement timeout protection
- [ ] Better exception handling with logging
- [ ] Add "Cancel processing" button
- [ ] Validate all stages before running

### Phase 4: Performance (1 day)
- [ ] Image downsampling for preview
- [ ] Intermediate result caching
- [ ] Async processing for expensive stages
- [ ] Performance monitoring dashboard

## Testing Checklist

After implementing fixes:

- [ ] Full pipeline completes in <5 seconds with default settings
- [ ] Progress indicator shows current stage
- [ ] Timeout triggers after reasonable time (10s?)
- [ ] Error messages are informative
- [ ] User can cancel long-running operations
- [ ] Document flattening works when explicitly enabled
- [ ] Step-by-step visualizer doesn't hang
- [ ] Memory usage stays reasonable (<2GB)

## Code Locations

**Key Files**:
1. [preprocessing_viewer_app.py:138-143](ui/preprocessing_viewer_app.py:138-143) - Main app pipeline execution
2. [pipeline.py:183-197](ui/preprocessing_viewer/pipeline.py:183-197) - Document flattening stage
3. [pipeline.py:84-357](ui/preprocessing_viewer/pipeline.py:84-357) - Full pipeline implementation
4. [document_flattening.py](ocr/datasets/preprocessing/document_flattening.py:1) - Slow operation implementation

**Related Documentation**:
- [Phase 2 Completion](docs/ai_handbook/05_changelog/2025-10/15_phase2_complete.md:101-111) - Performance characteristics
- [Phase 3 Integration](docs/ai_handbook/05_changelog/2025-10/15_phase3_complete.md:1) - Enhanced pipeline
- [Enhanced Preprocessing Usage](docs/ai_handbook/03_references/guides/enhanced_preprocessing_usage.md:1) - Best practices

## Quick Commands

```bash
# Check app status
ps aux | grep streamlit | grep preprocessing_viewer

# Kill hanging process
kill -9 3687430

# Restart app
uv run streamlit run ui/preprocessing_viewer_app.py --server.port 8501 --server.headless true &

# Monitor CPU usage
top -p 3687430

# Check logs
tail -f <streamlit-log-file>
```

## Expected Behavior After Fix

1. **Default Configuration (Fast)**:
   - Full pipeline completes in **<5 seconds**
   - Stages: detection, perspective, brightness, enhancement
   - Skips: flattening (expensive), orientation (optional)

2. **With Flattening Enabled**:
   - Clear warning: "Document flattening may take 3-15 seconds"
   - Progress indicator: "Stage 3/8: Flattening document..."
   - Timeout after 20 seconds with error message

3. **User Experience**:
   - Always shows which stage is running
   - Displays elapsed time
   - Can cancel processing
   - Informative error messages

## Success Criteria

✅ App doesn't hang on full pipeline
✅ Processing completes in <5 seconds (default config)
✅ User sees progress indication
✅ Errors are logged and displayed
✅ Can enable flattening with awareness of performance impact

---

**Next Steps**: Implement Phase 1 emergency fix, test, then proceed with Phase 2-4 as needed.

**Reference**: This debug session addresses issues identified in the Enhanced Preprocessing Pipeline (Phase 3) integration with the Streamlit UI.
