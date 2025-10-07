# 2025-10-06 Canonical Orientation Mismatch Bug Documentation

## Summary
Created comprehensive documentation for the canonical orientation mismatch bug discovered during GT overlay validation. The bug affected ~2% of validation images where polygon annotations were already in canonical frame despite EXIF rotation tags, causing double-rotation in the pipeline. Documentation covers discovery, root cause, resolution, and lessons learned for future reference.

## Changes Made

### 1. Created Bug Documentation
- **Added**: `docs/canonical_orientation_mismatch_bug.md` with complete bug report including:
  - Discovery details and environment context
  - Root cause analysis of annotation frame mismatches
  - Impact assessment on validation accuracy and debugging
  - Step-by-step resolution process
  - Code changes summary across affected modules
  - Validation procedures and results
  - Lessons learned and best practices

### 2. Enhanced Session Handover Documentation
- **Updated**: `docs/session_handover_rotation_debug.md` to include:
  - Reference to new physical correction script
  - Updated command examples with correct dataset paths
  - Detailed workflow for physical dataset correction
  - Validation results from dry-run testing

### 3. Code Fixes Recap (Previously Implemented)
- **Guard Logic**: Added `polygons_in_canonical_frame` detection in `ocr/utils/orientation.py`
- **Dataset Protection**: Modified `OCRDataset.__getitem__` to skip remapping when polygons are canonical
- **Physical Correction Tool**: Created `scripts/fix_canonical_orientation_images.py` for dataset surgery
- **Logging Improvements**: Switched WandB validation logging to tables to avoid oversized logs
- **Type Safety**: Fixed mypy issues in orientation utilities

## Benefits Achieved

1. **Knowledge Preservation**: Complete record of the bug for future developers and auditors
2. **Process Documentation**: Clear workflow for handling similar annotation inconsistencies
3. **Tool Availability**: Physical correction script enables dataset fixes when runtime guards aren't sufficient
4. **Validation Framework**: Established procedures for detecting and resolving orientation mismatches
5. **Best Practices**: Lessons learned guide future annotation and pipeline development

## Files Changed

- `docs/canonical_orientation_mismatch_bug.md` (new)
- `docs/session_handover_rotation_debug.md` (updated)
- `ocr/utils/orientation.py` (previously updated)
- `ocr/datasets/base.py` (previously updated)
- `ocr/datasets/craft_collate_fn.py` (previously updated)
- `ocr/lightning_modules/ocr_pl.py` (previously updated)
- `scripts/fix_canonical_orientation_images.py` (previously added)

## Validation

- **Documentation Completeness**: Covers all aspects from discovery to resolution
- **Type Checking**: All related code passes mypy validation
- **Tool Testing**: Correction script validated with dry-run on sample data
- **Mismatch Detection**: Confirmed 93 affected samples in validation set with detection tools
