# Preprocessing Viewer - Emergency Fixes

**Status**: 🔴 IMMEDIATE ACTION REQUIRED
**Apply Time**: ~30 minutes
**Impact**: Restore basic functionality while planning full refactor

---

## Quick Diagnosis Summary

Your observations reveal three critical bugs:

1. **Resource Leak**: Constant loading = infinite rerun loop or memory leak
2. **Dropdown Freeze**: Selecting "noise_eliminated" freezes = accessing missing/corrupted data
3. **Quality Problem**: Blurred text = aggressive preprocessing destroying details

---

## Emergency Fix 1: Stop the Bleeding (5 minutes)

### Disable All Broken Features

**File**: `ui/preprocessing_viewer/preset_manager.py`

Find the `get_default_config()` method and change these defaults:

```python
def get_default_config(self) -> dict[str, Any]:
    return {
        # === SAFE FEATURES (keep enabled) ===
        "enable_document_detection": True,
        "enable_perspective_correction": True,
        "enable_color_preprocessing": True,
        "convert_to_grayscale": False,
        "color_inversion": False,

        # === BROKEN/SLOW FEATURES (disable!) ===
        "enable_document_flattening": False,      # RBF hang issue
        "enable_orientation_correction": False,    # Rarely needed, slow
        "enable_noise_elimination": False,         # Blurs text
        "enable_brightness_adjustment": False,     # Quality issues
        "enable_enhancement": False,              # Over-sharpens

        # === OTHER SETTINGS ===
        "enhancement_method": "conservative",
        "enable_final_resize": False,
        "target_size": [1024, 1024],

        # ... rest of config
    }
```

**Why this works**: Eliminates all the stages causing freezes and quality issues.

---

## Emergency Fix 2: Fix Dropdown Freeze (10 minutes)

### Problem: Accessing results that don't exist

**File**: `ui/preprocessing_viewer/side_by_side_viewer.py`

Add defensive checks:

```python
def render_comparison(self, results: dict, available_stages: list[str]):
    col1, col2 = st.columns(2)

    # Get only stages that actually exist in results
    valid_stages = [s for s in available_stages if s in results and self.is_displayable_image(results.get(s))]

    if not valid_stages:
        st.warning("No displayable images available. Enable preprocessing stages in the sidebar.")
        return

    with col1:
        st.subheader("Left Image")
        left_stage = st.selectbox(
            "Select stage",
            options=valid_stages,  # Only show valid options!
            index=0,
            key="left_image_stage"
        )
        if left_stage in results:  # Double-check before accessing
            self._display_image(results[left_stage], left_stage)

    with col2:
        st.subheader("Right Image")
        right_stage = st.selectbox(
            "Select stage",
            options=valid_stages,  # Only show valid options!
            index=min(1, len(valid_stages) - 1),
            key="right_image_stage"
        )
        if right_stage in results:  # Double-check before accessing
            self._display_image(results[right_stage], right_stage)
```

**Why this fixes freeze**: Prevents selecting stages that don't exist in results dict.

---

## Emergency Fix 3: Add Result Caching (15 minutes)

### Problem: Reprocessing entire pipeline on every interaction

**File**: `ui/preprocessing_viewer_app.py`

Add caching decorator:

```python
import streamlit as st
import hashlib
import json

@st.cache_data(show_spinner=False, max_entries=10)
def process_image_cached(image_bytes: bytes, config_json: str):
    """Cache processed results by image + config hash."""
    import cv2
    import numpy as np
    from ui.preprocessing_viewer.pipeline import PreprocessingViewerPipeline

    # Decode image
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Parse config
    config = json.loads(config_json)

    # Process
    pipeline = PreprocessingViewerPipeline()
    return pipeline.process_with_intermediates(image, config)

# In main(), replace:
# pipeline_results = pipeline.process_with_intermediates(image, st.session_state.viewer_config)

# With:
if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    config_json = json.dumps(st.session_state.viewer_config, sort_keys=True)

    try:
        pipeline_results = process_image_cached(image_bytes, config_json)
    except Exception as exc:
        st.error(f"Processing error: {exc}")
        pipeline_results = None
```

**Why this fixes loading**: Caches results, only reprocesses when image or config changes.

---

## Emergency Fix 4: Quality - Reduce Aggressive Processing

### Fix Noise Elimination

**File**: `ocr/datasets/preprocessing/advanced_noise_elimination.py`

Find the `eliminate_noise` method and add a "gentle" mode:

```python
def eliminate_noise(self, image: np.ndarray, gentle: bool = True) -> NoiseEliminationResult:
    """
    Eliminate noise while preserving text details.

    Args:
        image: Input image
        gentle: If True, use text-preserving filters (recommended)
    """
    if gentle:
        # Use bilateral filter (edge-preserving) instead of morphological ops
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=50, sigmaSpace=50)
        method_used = "bilateral_gentle"
    else:
        # Original aggressive method
        denoised = self._aggressive_denoise(image)
        method_used = "morphological"

    return NoiseEliminationResult(
        cleaned_image=denoised,
        method_used=method_used,
        # ... rest
    )
```

### Fix RBF Smoothing

**File**: `ocr/datasets/preprocessing/document_flattening.py`

In `FlatteningConfig`, change default:

```python
class FlatteningConfig(BaseModel):
    # ... other fields
    smoothing_factor: float = Field(
        default=0.01,  # Changed from 0.1 - 10x less smoothing!
        ge=0.0,
        le=1.0,
        description="Smoothing factor for RBF interpolation (0.0-1.0)"
    )
```

---

## Testing the Fixes

### Test 1: App Loads Without Freezing
```bash
# Start app
uv run streamlit run ui/preprocessing_viewer_app.py --server.port 8501

# Upload image
# Expected: Pipeline completes in <3 seconds
# Expected: Results appear, no spinner stuck
```

### Test 2: Dropdown Works
```bash
# After processing completes
# Change "Right Image" dropdown to different stages
# Expected: Instant switch, no freeze
```

### Test 3: Text Quality
```bash
# Upload receipt image with small text
# View "final" result
# Expected: Text still readable, not blurred
```

---

## Rollback Plan

If these fixes break something:

```bash
# Revert changes
git diff HEAD > emergency_fixes.patch
git checkout -- ui/preprocessing_viewer/preset_manager.py
git checkout -- ui/preprocessing_viewer_app.py
git checkout -- ui/preprocessing_viewer/side_by_side_viewer.py
git checkout -- ocr/datasets/preprocessing/advanced_noise_elimination.py
git checkout -- ocr/datasets/preprocessing/document_flattening.py

# To re-apply later:
git apply emergency_fixes.patch
```

---

## Monitoring After Fixes

Watch these metrics:

1. **CPU Usage**: Should drop from 130%+ to <50% when idle
2. **Memory**: Should stay <500MB per session
3. **Response Time**: Dropdown changes <100ms
4. **Quality**: Text legibility maintained

---

## What's Next?

These emergency fixes make the app **barely functional** but it's still not production-ready.

**Proceed to**: [`preprocessing_viewer_refactor_plan.md`](preprocessing_viewer_refactor_plan.md) for the complete redesign strategy.

**Timeline**:
- **Today**: Apply emergency fixes, verify basic functionality
- **This week**: Decide on refactor approach (A, B, or C)
- **Next 2-4 weeks**: Execute chosen refactor plan
- **Month end**: Production-ready viewer with proper architecture

---

## Summary

| Fix | Time | Impact | Risk |
|-----|------|--------|------|
| Disable broken features | 5 min | High (immediate stability) | None |
| Fix dropdown freeze | 10 min | High (usability restored) | Low |
| Add caching | 15 min | Medium (performance boost) | Low |
| Quality fixes | 10 min | Medium (text preservation) | Low |

**Total Time**: ~40 minutes
**Expected Outcome**: App becomes minimally functional for basic use cases
**Next Step**: Plan and execute full refactor
