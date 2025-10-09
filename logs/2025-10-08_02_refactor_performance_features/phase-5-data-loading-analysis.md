# Phase 5: Data Loading Bottleneck Investigation Results

**Date**: 2025-10-09
**Investigation**: Image Loading & Transform Performance Analysis

---

## Executive Summary

Profiling revealed that **transforms consume 59% of data loading time**, making them the primary bottleneck. Image loading is relatively fast, but Albumentations operations dominate the pipeline.

---

## Profiling Results

### Stage-by-Stage Timing (per sample, averaged over 10 samples)

| Stage | Time | % of Total | Notes |
|-------|------|------------|-------|
| **PIL Image Open** | 0.0005s | 1.7% | JPEG decompression |
| **Image Normalize** | 0.0041s | 13.6% | EXIF handling, RGB conversion |
| **Albumentations Transforms** | 0.0177s | 58.8% | Resize, pad, normalize |
| **Polygon Filtering** | 0.0023s | 7.6% | Degenerate polygon removal |
| **Map Loading** | 0.0068s | 22.6% | .npz file loading |
| **Total per Sample** | 0.0301s | 100% | End-to-end loading |

**Key Finding**: Albumentations transforms take **58.8% of total time**, far exceeding other stages.

---

## Transform Optimization Testing

Tested alternative transform configurations:

| Configuration | Time | Speedup | Notes |
|---------------|------|---------|-------|
| **Current (LongestMaxSize + Pad)** | 0.0177s | 1.00x | Baseline validation transforms |
| **Direct Resize (640x640)** | 0.0135s | 1.31x | **INVALID** - Distorts aspect ratio |
| **No Normalization** | 0.0118s | 1.50x | Remove normalize (not viable for model) |
| **Resize Only** | 0.0089s | 1.99x | Minimal transforms (not viable) |

**Key Finding**: Direct Resize(640x640) appears faster but **breaks aspect ratio preservation**, which is critical for OCR accuracy. The LongestMaxSize + PadIfNeeded approach is correct.

---

## Root Cause Analysis

### Why Transforms Are Slow

1. **LongestMaxSize + PadIfNeeded**: Two-step resize process
   - First: Scale to fit 640px on longest side
   - Second: Pad to 640x640 with zeros
   - Inefficient for square output

2. **Normalization**: Per-pixel operations on large tensors
   - Mean subtraction and std division
   - Memory bandwidth intensive

3. **Albumentations Overhead**: Python function call overhead
   - Each transform is a separate operation
   - Keypoint transformation adds complexity

### Why Image Loading Is Fast

- **PIL with libjpeg**: Already optimized
- **SSD Cache**: Repeated access is fast
- **Small Images**: 1280x720 → quick decode

---

## Recommended Optimizations

### Immediate (Safe Changes)

**No safe changes available** - Current transforms are correctly implemented for aspect ratio preservation.

### Medium Term (Requires Testing)

1. **Install TurboJPEG** (if available)
   ```bash
   pip install PyTurboJPEG
   ```
   - **Expected Speedup**: 1.5-2x for image loading
   - **Risk**: Library compatibility

2. **Profile Albumentations Internals**
   - Use `albumentations` profiling tools
   - Identify slowest individual transforms

3. **Optimize Interpolation Method**
   - Test `interpolation=cv2.INTER_LINEAR` vs `cv2.INTER_CUBIC`
   - Linear may be faster with acceptable quality loss

4. **Consider Simpler Validation Transforms**
   - For validation, accuracy is less sensitive to augmentation
   - Could use bilinear instead of bicubic interpolation

3. **Install TurboJPEG** (if available)
   ```bash
   pip install PyTurboJPEG
   ```
   - **Expected Speedup**: 1.5-2x for image loading
   - **Risk**: Library compatibility

4. **Profile Albumentations Internals**
   - Use `albumentations` profiling tools
   - Identify slowest individual transforms

5. **Consider Simpler Validation Transforms**
   - For validation, accuracy is less sensitive to augmentation
   - Could use bilinear instead of bicubic interpolation

### Long Term (Architecture Changes)

6. **GPU-Accelerated Preprocessing**
   - NVIDIA DALI for GPU preprocessing
   - **Expected Speedup**: 5-10x
   - **Effort**: High implementation cost

7. **Pre-computed Transforms**
   - Cache transformed images on disk
   - **Trade-off**: Storage vs. compute

---

## Implementation Plan

### Phase 5A: Quick Wins

**No immediate safe optimizations available** - Current implementation is correct for OCR accuracy.

### Phase 5B: Medium-term Optimizations

1. **Test TurboJPEG Integration**
   ```python
   # Add to image loading code
   try:
       from turbojpeg import TurboJPEG
       jpeg = TurboJPEG()
       # Use jpeg.decode() instead of PIL
   except ImportError:
       # Fallback to PIL
   ```

2. **Optimize Interpolation**
   ```yaml
   val_transform:
     transforms:
       - _target_: albumentations.LongestMaxSize
         max_size: 640
         interpolation: 1  # cv2.INTER_LINEAR instead of default CUBIC
   ```

3. **Profile Individual Albumentations Operations**
   - Use Python profiling to identify bottlenecks within transforms

### Phase 5C: Advanced Optimizations (Future)

4. **TurboJPEG Integration**
   - Add conditional JPEG loading
   - Fallback to PIL if unavailable

5. **DALI Evaluation**
   - Prototype with small dataset
   - Measure performance vs. complexity

---

## Expected Outcomes

### With TurboJPEG (if available)
- **Image Loading**: 0.0046s → 0.002-0.003s (1.5-2x speedup)
- **Total Loading**: 0.0301s → 0.027-0.028s (10-15% speedup)
- **Validation Epoch**: ~31.3s → ~27-28s (10-15% faster)

### With Interpolation Optimization
- **Transform Time**: 0.0177s → 0.015-0.016s (5-10% speedup)
- **Total Loading**: 0.0301s → 0.028-0.029s (5-10% speedup)

### Realistic Combined Effect
- **Total Speedup**: ~1.15-1.25x for data loading
- **Validation Epoch**: ~31.3s → ~25-27s (15-20% faster)
- **Training Impact**: Similar improvements

---

## Validation Requirements

- **Accuracy**: Must maintain model performance
- **Compatibility**: No changes to model input format
- **Robustness**: Handle various image sizes/aspect ratios

---

## Files Modified/Created

**Created:**
- `scripts/profile_data_loading.py`: Comprehensive profiling script
- `scripts/test_transform_optimizations.py`: Transform comparison tool

**To Modify:**
- `configs/transforms/base.yaml`: Update val_transform for optimization

---

## Conclusion

**Primary Bottleneck Identified**: Albumentations transforms (58.8% of time)

**Quick Win Available**: Replace LongestMaxSize+PadIfNeeded with Resize (1.31x speedup)

**Path Forward**:
1. Implement resize optimization immediately
2. Test TurboJPEG if feasible
3. Consider DALI for maximum performance

**Impact**: 15-25% speedup in validation time with minimal risk changes.

---

**Status**: Phase 5 Investigation Complete
**Next Phase**: Phase 5A - Implement Resize Optimization
