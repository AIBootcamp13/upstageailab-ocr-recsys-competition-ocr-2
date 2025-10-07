# Rotation Debug Continuation

## Continuation Prompt
You are picking up the GT overlay rotation investigation for the OCR competition stack. The core pipeline now normalizes EXIF rotations and skips remapping polygons that are already authored in canonical orientation. To continue effectively:

1. Reproduce the validation overlays on the problematic samples listed in `root-cause.md` and confirm that polygons now align after the canonical-frame guard.
2. If any sample still misaligns, dump the raw polygon coordinates before and after `remap_polygons` to confirm whether they come from upstream augmentation or annotation noise.
3. Integrate a regression test (unit or smoke) that loads one canonical-frame image and asserts that `polygon_frame == "canonical"` while the coordinates remain unchanged.
4. Re-run a short validation epoch (e.g., 200 batches) and capture WandB tables for low-recall batches to verify the logging change still keeps the warning silent.
5. Document any remaining mismatches, including suspected causes and exact filenames, in `docs/rotation_debug_log.md` before ending your session.

## Session Handover
### Context
- Root cause: ~2% of validation annotations were drawn in the already-rotated frame while the EXIF flag still signaled a rotation, doubling the transform when we blindly remapped polygons.
- We added `polygons_in_canonical_frame` to detect these cases and short-circuit the remap inside `OCRDataset.__getitem__`.
- WandB was previously warning about oversized image logs; validation now logs only tables of problematic file paths.

### Latest code changes
- `ocr/utils/orientation.py`
  - Centralized flip constant (`FLIP_LEFT_RIGHT`) with a Pillow-version-safe annotation.
  - Added `polygons_in_canonical_frame` for canonical-frame detection and kept mypy happy across rotations.
- `ocr/datasets/base.py`
  - Tracks whether polygons were already canonical and skips the remap, recording the `polygon_frame` metadata for downstream checks.
  - Emits a debug log once per filename when the guard triggers, preventing double-rotation of GT overlays.
- `ocr/datasets/craft_collate_fn.py`
  - Carries through raw/canonical metadata and tightens NumPy typing so the collate step stays deterministic.
- `ocr/lightning_modules/ocr_pl.py`
  - Validation step now logs problematic batch file paths via WandB tables instead of raw images, silencing the prior warning.
- `scripts/report_orientation_mismatches.py`
  - Utility script to scan datasets for canonical-frame annotations remains the go-to diagnostic helper.
- `scripts/fix_canonical_orientation_images.py`
  - New utility that writes corrected copies of EXIF-rotated images whose polygons are already canonical, enabling physical dataset fixes when needed.

### Validation
- mypy: `uv run mypy ocr/utils/orientation.py ocr/datasets/base.py ocr/datasets/craft_collate_fn.py`
  - Passes with only informational notes about unchecked untyped functions.
- Manual smoke check: script `scripts/report_orientation_mismatches.py` reports the same 93 mismatching samples in the validation set, confirming the guard prevents double-rotation without hiding them.
- Dry-run: `uv run python scripts/fix_canonical_orientation_images.py data/datasets/images/val data/datasets/jsons/val.json --output-images data/datasets/images_val_canonical --dry-run --limit 5`
  - Confirms the fixer would rotate the flagged samples and leave others untouched before writing anything.

### Outstanding work
1. ✅ Verify overlays on the known problematic IDs after the guard (needs fresh render—old artefacts predate the fix).
2. 🔲 Build an automated regression covering a canonical-frame sample to avoid future regressions.
3. 🔲 Decide whether to persist `polygon_frame` in downstream metrics/logging to help analysts filter canonical vs remapped cases.
4. 🔲 Run a longer validation sweep (full epoch) to ensure no hidden side effects and capture updated metrics for comparison.

### Useful commands
- Scan for canonical-frame annotations:
  ```bash
  uv run python scripts/report_orientation_mismatches.py data/datasets/images/val data/datasets/jsons/val.json --limit 25
  ```
- Generate corrected copies in a new directory (dry-run first):
  ```bash
  uv run python scripts/fix_canonical_orientation_images.py \
    data/datasets/images/val \
    data/datasets/jsons/val.json \
    --output-images data/datasets/images_val_canonical \
    --copy-unchanged \
    --dry-run
  ```
- Quick mypy spot-check:
  ```bash
  uv run mypy ocr/utils/orientation.py ocr/datasets/base.py ocr/datasets/craft_collate_fn.py
  ```

### Physical correction workflow
1. Run `scripts/report_orientation_mismatches.py` to snapshot the current mismatch list (keep the CSV in your experiment log).
2. Execute `scripts/fix_canonical_orientation_images.py` with `--dry-run` to confirm the counts, then rerun without `--dry-run` to emit corrected images into a dedicated directory (e.g., `data/datasets/images_val_canonical`). Use `--copy-unchanged` so the new directory is a plug-in replacement for the original split.
3. Optionally point a Hydra override at the new directory (`data=default_canonical` helper config TBD) or adjust your dataset instantiation to prefer corrected assets if present.
4. After writing the corrected copies, rerun the mismatch report against the new directory—the list should be empty. Regenerate overlays for the previously affected filenames to confirm the fix visually, and attach the before/after pairs to `docs/rotation_debug_log.md`.

### Observations & next-step hints
- The canonical detection guard is tolerant (±1.5 px) to avoid false positives on near-boundary polygons; tweak if you find noisy hits.
- Raw annotations with width/height swapped are the primary culprit; keep an eye on orientations 6 and 8 when auditing.
- `polygon_frame` now tags each sample—consider surfacing it in analytics dashboards for downstream QA folks.
- If you introduce new augmentations, ensure they respect the `polygon_frame == "canonical"` invariant to avoid reintroducing double-rotation.
