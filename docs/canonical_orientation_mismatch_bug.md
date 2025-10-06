# Canonical Orientation Mismatch Bug

## Discovery

**Date Discovered:** October 2025
**Reporter:** Development team during GT overlay validation
**Environment:** OCR competition pipeline, validation dataset

The bug was identified during routine validation of ground truth (GT) overlays. Model predictions were misaligned with annotated polygons on approximately 2% of validation images, despite the pipeline applying EXIF-based rotation corrections. Initial investigation revealed that overlays showed polygons rotated twice, suggesting a double-application of orientation transforms.

## Root Cause

The issue stemmed from inconsistent annotation practices in the dataset:

- **EXIF Orientation Tags:** Images contained non-trivial EXIF orientation values (e.g., 6 for 90° clockwise rotation), indicating the camera captured them in a rotated orientation.
- **Annotation Frame Mismatch:** For ~2% of images, the polygon coordinates were already authored in the "canonical" (rotation-corrected) coordinate system, even though the EXIF tag signaled a rotation was needed.
- **Pipeline Behavior:** The OCR dataset loader (`OCRDataset.__getitem__`) blindly applied `remap_polygons` to all EXIF-rotated images, causing double-rotation on these canonical-frame annotations.

**Affected Samples:** Primarily images with orientation 6 (90° clockwise), where annotations were drawn post-manual rotation but EXIF tags were not cleared. The mismatch detector (`scripts/report_orientation_mismatches.py`) identified 93 such samples in the validation set.

## Impact

- **Validation Accuracy:** Misaligned GT overlays led to incorrect evaluation of model performance, potentially masking or exaggerating errors.
- **Training Inconsistency:** If present in training data, could cause the model to learn incorrect spatial relationships.
- **Debugging Overhead:** Required manual inspection of overlays and coordinate dumps to isolate the issue.
- **Reproducibility:** Affected downstream analyses relying on consistent polygon-image alignment.

## Resolution Steps

1. **Detection:** Implemented `polygons_in_canonical_frame` function in `ocr/utils/orientation.py` to detect when annotations already match the canonical frame.
2. **Guard Logic:** Modified `OCRDataset.__getitem__` to skip `remap_polygons` when polygons are canonical, adding `polygon_frame` metadata.
3. **Physical Correction Option:** Created `scripts/fix_canonical_orientation_images.py` to generate corrected image copies with EXIF tags cleared.
4. **Logging Adjustments:** Switched WandB validation logging from raw images to tables to avoid oversized logs.

## Code Changes

### `ocr/utils/orientation.py`
- Added `polygons_in_canonical_frame` with tolerance-based boundary checks.
- Annotated `FLIP_LEFT_RIGHT` for mypy compatibility.

### `ocr/datasets/base.py`
- Integrated canonical-frame detection in `__getitem__`.
- Added debug logging for skipped remaps.
- Introduced `polygon_frame` item metadata.

### `ocr/datasets/craft_collate_fn.py`
- Tightened NumPy typing for consistency.

### `ocr/lightning_modules/ocr_pl.py`
- Changed validation logging to WandB tables.

### `scripts/fix_canonical_orientation_images.py` (New)
- Utility for physical dataset correction.

## Validation

- **Type Checking:** `uv run mypy ocr/utils/orientation.py ocr/datasets/base.py ocr/datasets/craft_collate_fn.py` passes.
- **Mismatch Detection:** `scripts/report_orientation_mismatches.py` confirms 93 canonical-frame samples; guard prevents double-rotation without hiding them.
- **Dry-Run Correction:** `scripts/fix_canonical_orientation_images.py --dry-run` validates correction logic.
- **Overlay Verification:** Post-guard overlays show aligned polygons for previously affected samples.

## Lessons Learned

- **Annotation Hygiene:** Always clear EXIF orientation tags after manual rotation to avoid frame mismatches.
- **Defensive Coding:** Add guards for edge cases in data pipelines, especially with user-generated annotations.
- **Testing:** Include coordinate boundary checks in dataset unit tests.
- **Documentation:** Maintain clear records of data preprocessing assumptions and fixes.
- **Tooling:** Build utilities for detecting and correcting common annotation issues early.

## References

- `docs/session_handover_rotation_debug.md`: Continuation prompts and workflow.
- `scripts/report_orientation_mismatches.py`: Detection script.
- `root-cause.md`: Initial investigation notes.
