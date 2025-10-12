# Empty Predictions Resolution Plan

## Overview

This document outlines a systematic approach to resolve the empty predictions issue identified in [BUG-2025-010_empty_predictions_inference.md](../bug_reports/BUG-2025-010_empty_predictions_inference.md).

## Root Cause Summary

**Statistical profiling revealed systematic differences** between problematic images and working baseline images:

- **Problematic Images**: 6 images with identical reddish-beige synthetic tile floor backgrounds
- **Key Differences**:
  - **Brightness**: 119.04 vs 129.16 (p=0.037, statistically significant)
  - **Contrast**: 47.98 vs 68.11 (32% lower)
  - **Edge Density**: 0.031 vs 0.09 (40% fewer edges)
  - **Resolution**: All exactly 1280×1280 pixels (0.0 std deviation)

**Hypothesis**: Over-aggressive polygon filtering removes valid text regions on low-contrast backgrounds, causing complete prediction failure.

## Resolution Plan

### Phase 1: Polygon Filtering Adjustment

**Objective**: Reduce false-positive polygon removal while maintaining data quality.

**Steps**:
1. Locate `_filter_degenerate_polygons` function in `ocr/datasets/base.py`
2. Modify filtering logic to allow near-degenerate polygons:
   ```python
   # Current: removes polygons with any zero span
   if np.any(np.ptp(polygon, axis=0) == 0):
       return False

   # Proposed: allow polygons with minimal span (1-2 pixels)
   if np.any(np.ptp(polygon, axis=0) <= 1):  # or <= 2
       return False
   ```
3. Test with sample problematic images
4. Validate polygon retention vs. quality

**Expected Outcome**: Problematic images retain minimal valid polygons instead of being completely filtered.

### Phase 2: Background-Specific Preprocessing

**Objective**: Enhance text detection on low-contrast backgrounds.

**Steps**:
1. Implement background type detection in preprocessing pipeline
2. Add contrast enhancement for uniform backgrounds:
   ```python
   def enhance_low_contrast_background(image):
       # Detect uniform backgrounds using edge density threshold
       if calculate_edge_density(image) < 0.05:
           # Apply CLAHE or histogram equalization
           lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
           clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
           lab[:,:,0] = clahe.apply(lab[:,:,0])
           return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
       return image
   ```
3. Integrate into dataset preprocessing pipeline
4. Test on problematic vs. baseline images

**Expected Outcome**: Improved text region detection on synthetic tile floor backgrounds.

### Phase 3: Annotation Quality Validation

**Objective**: Confirm problematic images actually contain detectable text.

**Steps**:
1. Manually inspect sample problematic images for text content
2. Cross-reference with original annotations in dataset
3. Check if text is present but too subtle for current detection thresholds
4. Document findings and update bug report

**Expected Outcome**: Clear understanding of whether issue is preprocessing vs. data quality.

### Phase 4: Model Retraining Enhancement

**Objective**: Improve model robustness across diverse backgrounds.

**Steps**:
1. Augment training dataset with more uniform background examples
2. Include synthetic tile floor patterns in data augmentation
3. Retrain model with adjusted preprocessing
4. Validate on held-out problematic images

**Expected Outcome**: Model learns to handle low-contrast background scenarios.

## Testing Plan

### Unit Testing
- [ ] Test polygon filtering with various degenerate polygon scenarios
- [ ] Validate background enhancement preserves image quality
- [ ] Unit tests for edge density calculations

### Integration Testing
- [ ] End-to-end pipeline testing with problematic images
- [ ] Compare predictions before/after fixes
- [ ] Performance regression testing (maintain preprocessing speed)

### Validation Testing
- [ ] Inference testing on all 6 problematic images
- [ ] Statistical comparison of prediction quality metrics
- [ ] Cross-validation with baseline working images

**Success Criteria**:
- Problematic images produce non-empty predictions
- Prediction quality (IoU, precision, recall) meets baseline standards
- No performance degradation on existing working images

## Prevention Plan

### Code Quality Improvements
1. **Type Safety**: Add polygon validation with descriptive error messages
2. **Logging**: Enhanced logging for filtered polygons with reasoning
3. **Configuration**: Make filtering thresholds configurable per dataset

### Process Improvements
1. **Data Profiling**: Automated statistical profiling of new datasets
2. **Background Analysis**: Include background type distribution in dataset reports
3. **Quality Gates**: Automated checks for empty prediction scenarios

### Documentation Updates
1. **Preprocessing Guidelines**: Document background-specific preprocessing requirements
2. **Dataset Standards**: Include minimum contrast and edge density requirements
3. **Troubleshooting Guide**: Add empty predictions debugging workflow

## Implementation Timeline

| Phase | Duration | Dependencies | Risk Level |
|-------|----------|--------------|------------|
| Polygon Filtering | 2-3 days | Dataset access | Low |
| Background Preprocessing | 3-5 days | Image processing libraries | Medium |
| Annotation Validation | 1-2 days | Manual inspection | Low |
| Model Retraining | 5-7 days | GPU resources | High |

## Risk Assessment

**High Risk**: Model retraining could introduce regression on existing performance
**Mitigation**: Maintain separate model versions, extensive validation testing

**Medium Risk**: Background enhancement could affect other image types
**Mitigation**: Conditional application based on background detection

**Low Risk**: Polygon filtering adjustments
**Mitigation**: Gradual threshold increases with rollback capability

## Monitoring & Metrics

**Key Metrics to Track**:
- Prediction completeness rate (non-empty predictions)
- Polygon retention rate during preprocessing
- Background type distribution in datasets
- Edge density and contrast statistics

**Alert Thresholds**:
- Empty predictions > 5% of batch → Investigate
- Polygon retention < 80% → Review filtering logic
- Background uniformity > 90% → Flag for preprocessing

## Rollback Plan

1. **Immediate Rollback**: Revert polygon filtering changes
2. **Gradual Rollback**: Restore original preprocessing pipeline
3. **Data Rollback**: Use previous dataset cache if needed
4. **Model Rollback**: Switch to previous model checkpoint

## Success Validation

**Quantitative Metrics**:
- All 6 problematic images produce ≥1 prediction
- Prediction count within 1σ of baseline images
- IoU scores maintained above 0.7 threshold

**Qualitative Validation**:
- Visual inspection of predictions on problematic images
- Expert review of text detection accuracy
- User acceptance testing in inference pipeline

---

**Document Version**: 1.0
**Last Updated**: October 13, 2025
**Related Documents**:
- [Bug Report](../bug_reports/BUG-2025-010_empty_predictions_inference.md)
- [EDA Analysis](../debug_empty_predictions/eda_results/)</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/bug_reports/BUG-2025-010_empty_predictions_resolution_plan.md
