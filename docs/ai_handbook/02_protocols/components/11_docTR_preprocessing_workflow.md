# docTR-Enhanced Preprocessing Workflow

_Last updated: 2025-09-30_

## Overview
This guide captures the workflow we used to adopt the docTR preprocessing pipeline inside our OCR stack. It is intended to be the one-stop reference when you need to pick the work back up from a fresh context, review required toggles, or extend the feature set (for example, into the Streamlit inference UI).

Key additions delivered so far:
- Optional docTR-powered orientation correction, perspective rectification, and padding cleanup in `ocr/datasets/preprocessing.DocumentPreprocessor`.
- Hydra-configurable switches (`configs/preset/datasets/preprocessing.yaml`) so experiments can turn the features on/off without code changes.
- Unit tests (`tests/test_preprocessing.py`) guaranteeing behaviour when docTR is available.

## Picking up from a new context
1. **Sync & install dependencies**
   - Pull the repo and ensure `python-doctr>=1.0.0` is available (managed via `uv`).
   - Warm up the virtualenv: `uv sync` (or `uv pip install python-doctr` when rehydrating in a lightweight workspace).
2. **Verify preprocessing state**
   - Skim `ocr/datasets/preprocessing.py` for manual changes (orientation pipeline is optional and guarded).
   - Confirm Hydra presets (`configs/preset/datasets/preprocessing.yaml`) expose the toggles you intend to flip during the session.
3. **Run the focused tests**
   - `uv run pytest tests/test_preprocessing.py` – the suite is fast and validates docTR + OpenCV behaviour.
4. **Decide on experiment knobs**
   - Use the Hydra keys documented below to enable or disable pieces of the pipeline.
5. **Record deltas**
   - Add a short note in `docs/ai_handbook/04_experiments` or the relevant project log capturing which toggles were used and why. This keeps the continuation trail clear for the next round.

## Configuration knobs (Hydra)
All entries live under `preprocessing:` in `configs/preset/datasets/preprocessing.yaml` and bubble straight into `DocumentPreprocessor`.

| Key | Purpose |
| --- | --- |
| `enable_document_detection` | Find document boundaries via contour analysis. |
| `enable_orientation_correction` | Estimate page angle with docTR and deskew when the threshold is exceeded. |
| `orientation_angle_threshold` | Minimum absolute angle (°) before we attempt correction. |
| `orientation_expand_canvas` | Allow canvas expansion while rotating to avoid cropping corners. |
| `orientation_preserve_original_shape` | After rotation, resize back to the original shape. |
| `use_doctr_geometry` | Prefer docTR’s `extract_rcrops` for perspective correction before falling back to OpenCV. |
| `doctr_assume_horizontal` | Hint to docTR crops that text is horizontally aligned (faster + more stable when true). |
| `enable_padding_cleanup` | Strip leftover black borders via docTR padding removal. |
| `enable_enhancement` / `enhancement_method` | Choose conservative vs. “office lens” photometric enhancements. |
| `enable_text_enhancement` | Apply morphological text cleanup when needed. |
| `target_size` | Final padded resolution (width, height). |
| `enable_final_resize` | Keep the rectified page at its native resolution when disabled. |
| `document_detection_min_area_ratio` | Ignore small contours when hunting for page boundaries. |
| `document_detection_use_adaptive` | Run an adaptive-threshold fallback when Canny misses the page. |
| `document_detection_use_fallback_box` | Fall back to the largest bounding box of foreground pixels when contours fail. |

## Validating changes quickly
- `uv run pytest tests/test_preprocessing.py`
- Spin up the inference UI (`uv run streamlit run ui/inference_ui.py`) when experimenting with visual toggles; keep a note of which preset was active.
- In the Streamlit sidebar, expand **docTR options** to tweak detection thresholds, orientation settings, and the new resize toggle without editing YAML.

## Troubleshooting detection & scaling
When docTR enhancements appear ineffective:

1. **Inspect metadata in the UI.** The preprocessing panel now records `document_detection_method` (values like `canny_contour`, `adaptive_threshold`, `bounding_box`, or `failed`). If the method ends up as `failed`, docTR geometry never ran.
2. **Check the Hydra overrides.** `document_detection_min_area_ratio`, `document_detection_use_adaptive`, and `document_detection_use_fallback_box` mirror the UI controls. Lower the area ratio for receipts or enable the adaptive/bounding-box fallbacks before rerunning.
3. **Verify orientation toggles.** A low `orientation_angle_threshold` or disabling canvas expansion can leave rotated pages uncropped. Adjust these in the UI or preset and rerun.
4. **Skip the final resize to compare scales.** Disable `enable_final_resize` in the UI to keep the rectified page in its native size. This is useful when you want W&B and local logs to match the uploaded resolution.
5. **Check W&B logs after updates.** The validation logger now crops out black padding before uploading, so images should no longer appear 3× larger. If you still see magnified results, confirm that `enable_final_resize` is off for the run and that the inference UI metadata shows the expected `final_shape`.

## Streamlit inference UI integration
- The sidebar now exposes a **docTR preprocessing** checkbox sourced from `configs/ui/inference.yaml`. Toggle it to run the preprocessing stack before inference and surface metadata/visuals per image.
- When enabled, each result card shows a "🧪 docTR Preprocessing" section with the original upload (overlaying detected corners) and the rectified output.
- The `InferenceService` persists preprocessing mode per checkpoint to avoid re-running already processed images when you flip between docTR-enabled and baseline runs.

To refresh the demo assets or re-run the visual regression locally:

```bash
uv run python scripts/generate_doctr_demo.py
```

This script renders synthetic document imagery, processes it with docTR on/off, and saves outputs under `outputs/protocols/doctr_preprocessing/` for inclusion in docs or presentations.

### Sample before/after

| Original Upload | After docTR Preprocessing |
| --- | --- |
| ![Original synthetic document](outputs/protocols/doctr_preprocessing/demo_original.png) | ![docTR rectified output](outputs/protocols/doctr_preprocessing/demo_doctr.png) |

For a baseline comparison, `demo_opencv.png` showcases the OpenCV-only pathway when docTR features are disabled. Metadata exports (`*.metadata.json`) capture the processing steps and corner geometry used in the UI visualisations.

## Recommended documentation practice
- **Continuation prompt:** capture a ready-to-run prompt (see template below) inside the PR description or project log. Include toggles, open todos, and observed issues.
- **Context checkpoint:** when you finish a session, add a bullet list of “Next Session” items referencing Hydra keys, sample commands, and any failing tests.
- **UI alignment:** if you extend the Streamlit apps, update this protocol with the new entry points, so the next person can trace the workflow from config → code → UI.

## Continuation prompt template
```
You are resuming the docTR preprocessing integration work.
State: {summary of last run, toggles enabled}
Open Tasks:
- [ ] Implement {feature}
- [ ] Validate {test or UI check}
Context Files:
- ocr/datasets/preprocessing.py
- configs/preset/datasets/preprocessing.yaml
- tests/test_preprocessing.py
Key Commands:
- uv run pytest tests/test_preprocessing.py
- uv run streamlit run ui/inference_ui.py
```
Replace the braces with session-specific details before you sign off.

## Next planned extensions
- Integrate docTR’s demo Streamlit components into `ui/inference_ui.py` to visualise the new preprocessing steps.
- Ship a Hydra preset that toggles docTR features on/off for quick A/B runs (see upcoming preset file).
- Document UI controls once the integration lands.
