# Orientation Debugging Plan

_Last updated: 2025-10-05_

## Context

Recent training runs revealed severe precision/recall swings tied to EXIF orientation handling. Images reach the model in a normalized orientation, yet polygons, metrics, and downstream tooling occasionally disagree. This plan centralizes the debugging workflow so every contributor can reproduce, diagnose, and fix orientation regressions without ad-hoc spelunking.

## Objectives

1. **Verify** every pipeline stage (dataset ➜ augmentation ➜ model ➜ metric ➜ logging ➜ UI) consumes a consistent orientation contract.
2. **Detect** mismatches automatically via targeted tests, smoke runs, and dataset audits.
3. **Document** outcomes and follow-ups so progress is transparent and new regressions are caught early.

## Scope

- Data ingestion (`ocr/datasets`), Lightning modules, metrics, callbacks, inference utilities, and visualization code.
- Regression and smoke tests under `tests/ocr`, `tests/integration`, and `tests/debug`.
- Monitoring artifacts (W&B panels, saved overlays) produced during validation.

_Out of scope_: Doctr-based page angle correction beyond EXIF normalization. That work can re-use this framework once this plan is complete.

## Roles

| Area | Contact | Notes |
| --- | --- | --- |
| Dataset + Collate | Core training team | Owns Albumentations pipelines and metadata propagation |
| Lightning module | Training infra | Maintains metric computation and logging |
| UI / Tooling | Visualization squad | Validates Streamlit/offline/W&B overlays |
| QA | Test maintainers | Keeps regression suites green |

## Artifacts & Commands

| Artifact | Purpose | Command (run from repo root) |
| --- | --- | --- |
| Orientation unit tests | Validate helpers | ```bash
uv run pytest tests/ocr/utils/test_orientation.py
``` |
| EXIF integration smoke | Dataset ➜ model ➜ metrics | ```bash
uv run pytest tests/integration/test_exif_orientation_smoke.py
``` |
| Dataset audit script | Sample EXIF-tagged data | ```bash
uv run python tests/debug/data_analyzer.py --sample 50 --dump orientation_audit.json
``` |
| Training smoke | Verify metric alignment | ```bash
uv run python runners/train.py trainer.max_epochs=1 dataloaders.val_dataloader.batch_size=4 logger.wandb.enabled=false
``` |
| Visualization check | Confirm overlays post-training | ```bash
uv run python ui/visualize_predictions.py --image <path> --pred-json <path>
``` |

## Phase Breakdown

### Phase 1 — Static Audit

- [ ] Remove redundant rotation helpers (e.g. `OCRDataset.rotate_image`).
- [ ] Ensure all orientation manipulations import from `ocr.utils.orientation`.
- [ ] Add type hints / lint rules preventing per-module reinventions.
- Deliverable: Clean diff with obsolete helpers removed, approved via PR.

### Phase 2 — Dataset Verification

- [ ] Expand `tests/ocr/datasets/test_exif_rotation.py` to cover EXIF {1,3,6,8} samples.
- [ ] Run Albumentations regression to check keypoint alignment (`tests/ocr/datasets/test_exif_rotation.py::test_dataset_albumentations_preserves_polygon_alignment`).
- [ ] Capture summary CSV from `tests/debug/data_analyzer.py` (raw vs remapped polygon IoUs).
- Deliverable: `orientation_dataset_report.json` checked into `results/` (gitignored) with mean IoU ≥ 0.99.

### Phase 3 — Training & Metric Validation

- [x] Execute 1-epoch smoke training run; capture CLEval metrics per batch. *(See `logs/orientation_debug/2025-10-05_04-training-smoke.md`.)*
- [ ] For batches with recall < 0.85, export `raw_size`, `orientation`, and predictions to `logs/orientation_debug/`.
- [ ] Recompute CLEval offline using `ocr.utils.orientation.remap_polygons` to confirm corrections.
- Deliverable: Notebook or markdown summary in `logs/orientation_debug/README.md` with findings.

### Phase 4 — Visualization Checks

- [ ] Regenerate W&B image logging samples (`WandbImageLoggingCallback`).
- [ ] Cross-check Streamlit viewer (`ui/_visualization/viewer.py`) and offline visualizer overlays against exported predictions.
- [ ] Document any discrepancies with screenshots in `docs/orientation-findings/`.
- Deliverable: Screenshot set showing aligned overlays in all viewers.

### Phase 5 — Regression Gating

- [ ] Add `make check-orientation` target running the artifacts above.
- [ ] Wire CI job (or local pre-commit) to execute the target on PRs touching orientation-sensitive files.
- [ ] Update `docs/generating-submissions.md` with the orientation pipeline contract.
- Deliverable: CI green run proving the gate passes.

## Tracking Dashboard

| Task | Owner | Status | Notes |
| --- | --- | --- | --- |
| Static audit complete |  | ☐ | Remove dead helpers, confirm imports |
| Dataset verification |  | ☐ | Tests + analyzer report |
| Training smoke analyzed |  | ☐ | Document low-recall batches |
| Visualization verified |  | ☐ | W&B + Streamlit screenshots |
| Regression gate in place |  | ☐ | `make check-orientation` wired |

> Update the status column with ✅ / 🚧 / ❌ as work progresses. Link to PRs or log files in the Notes column.

## Risks & Mitigations

- **Hidden orientation paths** (e.g., legacy scripts): run a repo-wide search for `rotate(`, `transpose(` tied to PIL/NumPy and review manually.
- **Albumentations drift**: pin version in `pyproject.toml`; document required config in the plan.
- **High cost smoke runs**: reuse cached datasets; reduce validation set with Hydra overrides when debugging.

## Daily Log Template

```
### YYYY-MM-DD
- What changed:
- Tests / commands run:
- Issues found:
- Next steps:
```

Keep this file committed; teams can append daily notes to the bottom during the debugging window.
