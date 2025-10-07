## 🧭 Status: EXIF Rotation Bug (2025-10-05)

Silent and stubborn. Images carrying EXIF orientation tags are entering the training/visualization stack with upright pixels but **misaligned polygons**, leading to catastrophic recall/Hmean swings whenever those samples surface.

### Symptoms
- Validation metrics plunge on specific batches despite stable logits.
- Bounding boxes appeared 90° off in offline visualizer before we patched it.
- WandB logs occasionally show upright images with sideways overlays.

### Root cause summary
| Stage | Image orientation | Polygon orientation | Notes |
|-------|-------------------|---------------------|-------|
| Dataset (`ocr/datasets/base.py`) | Normalized via `normalize_pil_image` | **Remapped to canonical frame** | Polygons now flow through Albumentations in sync with rotated pixels. Regression test renamed to `test_dataset_remaps_polygons_with_orientation`. |
| Training callbacks (W&B) | Rotated via same helper | Copies polygons straight through | Logs replicate the mismatch originating from the dataset. |
| Offline visualizer (`ui/visualize_predictions.py`) | **Fixed** — rotates image only | Uses model output directly | Now consistent with inference results after recent fix. |
| Streamlit viewer (`ui/_visualization/viewer.py`) | Rotates image | Draws polygon strings untouched | Accuracy depends entirely on upstream data orientation. |
| Inference engine (`ui/utils/inference/engine.py`) | `cv2.imread` (ignores EXIF) | N/A (uses model predictions) | Model predicts in raw sensor orientation; any downstream rotation must keep polygons as-is. |

### Impact
- Training sees mismatched supervision → degraded convergence, noisy gradients, evaluation spikes.
- Monitoring surfaces confusing GT/pred overlays, hindering debugging.
- Any new tool risks re-implementing a different partial fix.

---

## ✅ Recent progress
1. Documented rotation handling across dataset, logging, visualization, inference.
2. Verified that OpenCV path ignores EXIF while PIL honours it.
3. Removed extra polygon rotation in offline visualizer to match inference output.
4. Ran targeted sanity scripts to confirm orientation transform maths.
5. Wired `OCRDataset` into the shared helpers and updated dataset-level tests to expect rotated polygons.
6. Added Albumentations regression (`tests/ocr/datasets/test_exif_rotation.py::test_dataset_albumentations_preserves_polygon_alignment`) to ensure keypoints stay aligned post-normalization.
7. Refactored W&B logging and Streamlit/offline viewers to consume the shared normalization utilities, keeping overlays aligned across tooling.
8. Added flattened-polygon regression (`tests/ocr/utils/test_orientation.py::test_remap_polygons_flattens_to_viewer_format`) covering viewer/logging data shapes.
9. Executed a single-epoch smoke run (2 train/2 val batches) with orientation metadata propagated end-to-end; run notes captured in `logs/orientation_debug/2025-10-05_04-training-smoke.md`.

---

## 🎯 Goal
Establish a **single, reusable orientation pipeline** that:
1. Normalizes images to a canonical orientation (when desired).
2. Applies matching transforms to polygons/boxes/segmentation masks.
3. Exposes helpers for both PIL- and NumPy-based flows.
4. Keeps tests, callbacks, visualizers, and inference in sync.

---

## 🛠️ Work plan (resume checklist)

### Phase 1 — Build unified utilities
- [x] Add `ocr/utils/orientation.py` with:
  - `normalize_pil_image(image: Image.Image) -> tuple[Image.Image, int]`
  - `normalize_ndarray(image: np.ndarray, orientation: int) -> np.ndarray`
  - `remap_polygons(polygons, width, height, orientation)` supporting NumPy arrays/lists
- [x] Include orientation constants + small helpers (e.g., `orientation_requires_rotation(exif)`)
- [x] Write unit tests covering representative EXIF values (1–8) for image + polygon transforms *(test module in `tests/ocr/utils/test_orientation.py`; execution currently blocked upstream because `tests/conftest.py` imports Torch, which is not installed in this environment)*

### Phase 2 — Adopt in dataset pipeline
- [x] Update `OCRDataset.__getitem__` to call `normalize_pil_image`
- [x] Rotate polygons via `remap_polygons` before Albumentations
- [x] Adjust `tests/ocr/datasets/test_exif_rotation.py` to expect rotated polygons
- [x] Verify Albumentations keypoint transforms still behave (regression: `tests/ocr/datasets/test_exif_rotation.py::test_dataset_albumentations_preserves_polygon_alignment`)

### Phase 3 — Align downstream tooling
- [x] `wandb_image_logging.py`: reuse utility for both image + polygons
- [x] `ui/_visualization/viewer.py`: parse polygons, call helper so overlays stay aligned
- [x] `ui/visualize_predictions.py`: optionally switch to helper for clarity (already consistent)

### Phase 4 — Fix inference preprocessing
- [x] Introduce EXIF-aware loading inside `InferenceEngine`
    - Option A: switch to PIL loading + convert to NumPy
    - Option B: detect orientation via PIL, then run `normalize_array`
- [x] Ensure prediction polygons remain untouched; confirm Streamlit UI still renders correctly

### Phase 5 — Regression + monitoring
- [x] Add smoke test that loads a known EXIF image through dataset → model → logger → visualizer
- [x] Extend `tests/debug/data_analyzer.py` to flag EXIF-tag distribution + potential mismatches
- [ ] Capture before/after metrics during next training run

---

8. Added flattened-polygon regression (`tests/ocr/utils/test_orientation.py::test_remap_polygons_flattens_to_viewer_format`) covering viewer/logging data shapes.
9. Reworked inference engine to ingest via PIL, track orientation metadata, and remap predictions back to raw alignment.
10. Updated Streamlit helper parsing to accept comma-delimited polygons and added UI regression tests (`tests/ui/test_visualization_helpers.py`).
11. Brought `ui/visualize_annotations.py` into the orientation stack with polygon overlays plus dedicated regressions (`tests/ui/test_visualize_annotations.py`).
12. Smoke run produced W&B overlays without polygon drift for mirrored orientations; metrics remain intentionally low pending full retrain.
- Expanded `tests/debug/data_analyzer.py` with EXIF distribution counts and orientation-alignment audit helpers.
- Any legacy scripts (e.g., deprecated Streamlit or notebooks) should be audited once utilities exist.
- Consider documenting orientation strategy in `docs/generating-submissions.md` or similar
- Future enhancement: integrate page-angle correction (Doctr pipeline) after EXIF normalization.

### ⏭️ Suggested next focus
- Kick off Phase 5 by wiring an end-to-end smoke test through the EXIF-aware pipeline and instruments in `tests/debug/data_analyzer.py`.
