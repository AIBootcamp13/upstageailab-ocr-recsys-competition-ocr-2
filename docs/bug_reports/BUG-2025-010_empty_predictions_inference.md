## 🐛 Bug Report: Empty Predictions During Inference After Degenerate Polygon Filtering

**Bug ID:** BUG-2025-010
**Date:** October 13, 2025
**Reporter:** Development Team
**Severity:** Critical
**Status:** Open

### Summary
After implementing preprocessing to filter degenerate polygons, certain images produce completely empty predictions during inference. The preprocessing pipeline removes all polygons from these images, resulting in no training data for those samples and subsequent inference failures where the model outputs no detections.

### Environment
- **Pipeline Version:** Post-degenerate polygon filtering implementation
- **Components:** Dataset preprocessing, Polygon filtering (`_filter_degenerate_polygons`), Inference pipeline
- **Configuration:** Polygon filtering enabled, convex hull processing active
- **Affected Images:** Multiple images from test set including:
  - `drp.en_ko.in_house.selectstar_000732.jpg`
  - `drp.en_ko.in_house.selectstar_001012.jpg`
  - `drp.en_ko.in_house.selectstar_001161.jpg`
  - `drp.en_ko.in_house.selectstar_000699.jpg`
  - `drp.en_ko.in_house.selectstar_000712.jpg`
  - `drp.en_ko.in_house.selectstar_001007.jpg`

### Steps to Reproduce
1. Enable polygon preprocessing with degenerate filtering
2. Process dataset containing images with near-degenerate polygons (horizontal/vertical lines, tiny regions)
3. Train model on filtered dataset
4. Run inference on affected images
5. Observe empty predictions (no polygon coordinates in output CSV)

### Expected Behavior
Images should retain at least minimal polygon annotations after preprocessing, allowing the model to make predictions during inference. Even if polygons are repaired or inflated, they should not be completely removed.

### Actual Behavior
```csv
filename,polygons
drp.en_ko.in_house.selectstar_000732.jpg,
drp.en_ko.in_house.selectstar_001012.jpg,
```
Empty polygons field indicates complete absence of predictions. Logs show convex hull failures for degenerate polygons:
```json
{"timestamp": "2025-10-11T17:54:12.557246Z", "exception": "ValueError", "message": "degenerate polygon after rounding", "points": [[380, 960], [390, 960], ...]}
```

### Root Cause Analysis
**Over-aggressive Filtering:** The `_filter_degenerate_polygons` function in `ocr/datasets/base.py` removes polygons that collapse to zero-area in integer space after rounding. Many polygons are horizontal/vertical lines or tiny regions that become degenerate.

**Pipeline Corruption:** The preprocessing pipeline is corrupt, not the datasource. Valid annotations exist but are over-filtered, leaving images with no polygons for training/inference.

**Impact on Training:** Images with no polygons contribute no loss during training, leading to poor model generalization.

**Impact on Inference:** Model fails to detect text in similar images, producing empty predictions.

**Code Path:**
```
Dataset Loading → Polygon Filtering (_filter_degenerate_polygons) → All polygons removed → Empty annotations → Training with no loss → Inference produces no detections
```

### Resolution
1. **Adjust Filtering Thresholds:** Modify `_filter_degenerate_polygons` to allow near-degenerate polygons (minimum side length 1-2 pixels instead of 0)
2. **Implement Polygon Repair:** Add repair logic in `ocr/utils/polygon_utils.py` to inflate tiny polygons
3. **Review Preprocessing Config:** Check `configs/preset/datasets/preprocessing.yaml` for overly strict settings
4. **Add Fallback in Inference:** Implement fallback predictions in `ui/apps/inference/services/inference_runner.py` for empty results
5. **Re-train and Re-validate:** Regenerate dataset cache and re-run training/inference after fixes

### Testing
- [ ] Validate polygon filtering retains minimal annotations
- [ ] Test inference on affected images produces non-empty predictions
- [ ] Performance regression test maintains preprocessing speed
- [ ] Integration test for polygon repair functionality

### Prevention
- Add minimum polygon count validation per image
- Implement polygon repair instead of discarding
- Add logging for filtered polygon statistics
- Create integration tests for preprocessing pipeline
- Document polygon quality requirements in dataset contracts</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/bug_reports/BUG-2025-010_empty_predictions_inference.md
