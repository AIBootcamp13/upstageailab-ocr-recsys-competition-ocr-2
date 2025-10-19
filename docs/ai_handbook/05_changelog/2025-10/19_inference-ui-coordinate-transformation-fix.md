# 2025-10-19: Inference UI Coordinate Transformation Bug Fix

## Summary
Fixed a critical bug in the inference UI where OCR text annotations were misaligned for EXIF-oriented images. The issue caused predictions to appear rotated 90° clockwise relative to the correctly displayed image, rendering inference results unusable for oriented images.

## Root Cause
The `InferenceEngine._remap_predictions_if_needed()` method was incorrectly applying inverse orientation transformations to prediction coordinates. Since predictions are already generated in the normalized coordinate system (after image rotation), applying inverse transformations moved them to the wrong positions.

## Changes Made
- **File**: `ui/utils/inference/engine.py`
  - Removed incorrect `_remap_predictions_if_needed()` calls that applied inverse orientation transformations
  - Predictions now remain in the normalized coordinate system for correct display alignment

## Impact
- **Fixed**: OCR annotations now correctly align with displayed images for all EXIF orientations
- **Restored**: Full functionality of inference UI for oriented images
- **Maintained**: Backward compatibility with no breaking changes

## Testing
- Verified fix with test image `drp.en_ko.in_house.selectstar_000699.jpg` (EXIF orientation 6)
- Confirmed prediction coordinates remain within image boundaries
- Validated no regression for non-oriented images

## Related Documentation
- Bug Report: `docs/ai_handbook/05_changelog/2025-10/19_inference-ui-coordinate-transformation-bug.md`
- EXIF Orientation Handling: `ocr/utils/orientation.py`
- Inference Engine: `ui/utils/inference/engine.py`</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/ai_handbook/05_changelog/2025-10/19_inference-ui-coordinate-transformation-fix.md
