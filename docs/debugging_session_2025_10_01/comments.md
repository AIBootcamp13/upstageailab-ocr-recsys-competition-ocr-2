# Document relocated# Session Handoff Notes — 2025-10-02 (Evening Update)



Comments and KEEP/REVERT notes have moved to [`docs/next_job/comments.md`](../next_job/comments.md).## Porting plan snapshot

- ✅ **Confirmed fix**: Only the explicit `postprocess` block in `configs/model/architectures/dbnet.yaml` improved recall on the dismantled `05_refactor/preprocessor_d` branch. Keep this change and the accompanying regression test when moving back to `05_refactor/preprocessor`.

Please record all future updates there.- 🚫 **Speculative rewrites**: The BaseHead/BaseLoss registry plumbing, runner rewrites, and callback diff churn introduced by `preprocessor_d` provided no measurable recall gains. Prefer reverting these wholesale to regain the leaner `preprocessor` branch before layering new fixes.

- 🧪 **Guardrail test**: `pytest tests/test_architecture.py::TestOCRModel::test_dbnet_architecture_passes_postprocess_config` now enforces that Hydra forwards the postprocess thresholds. Run it after every config merge.
- 🗒️ **Action item**: Use `git diff --stat 05_refactor/preprocessor..05_refactor/preprocessor_d` as a checklist. For each diff chunk, mark KEEP / REVERT in this file once inspected.

We now have multiple historical baselines plus the actively failing `05_refactor/preprocessor` branch. A high-performing variant still exists (`history-2`), but the underlying Hydra layout has diverged so far that we cannot run cross-branch experiments with identical commands. Targeted diffing and component-level porting are required before any fair comparison.

## Immediate Recommendations

- **Preserve both worktrees**: keep `/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2` (current refactor) and `/home/vscode/workspace/ocr-mobilnet-rewind/upstageailab-ocr-recsys-competition-ocr-2` (historical rewind) intact. The rewind tree depends on a symlinked dataset and was restored from the W&B git-state snapshot.
- **Avoid one-click config reuse**: Hydra groups, defaults, and runtime keys were heavily rearranged. Copying full configs between branches or running past commands verbatim won’t work; isolate only the specific modules (encoder/decoder/loss heads, trainer knobs, dataset transforms) that you intend to port.
- **Define narrow comparison slices**:
	1. Identify one behaviour delta at a time (e.g., DBLoss logits vs. legacy loss).
	2. Map the relevant Hydra nodes in each branch, then reproduce just that section in the other branch.
	3. Re-run with branch-local defaults to confirm stability before combining changes.
- **Document metric parity gaps**: the broken refactor run logs only 14 summary metrics and is missing validation images. Regression tracking should include the extra W&B keys that exist in `history-2`.

## Comparison Targets

- Located outside the project root: `/home/vscode/workspace/ocr-mobilnet-rewind/upstageailab-ocr-recsys-competition-ocr-2`
- Uses a symlink to the shared dataset
- Recreated from a W&B `git state` command

### alias: "history-2"
- Date: 2025-09-26 11:02:37
- Location: `/home/vscode/workspace/ocr-mobilnet-rewind/upstageailab-ocr-recsys-competition-ocr-2`
- Branch: `wchoi189_model-b12-lr0e+00_loss0.0000`
- Final run name: `wchoi189_model-b12-lr0e+00_loss0.0000`
- Characteristics: highest performance at just 1 epoch; W&B validation images reveal a few GT/Pred rotation mismatches.

Log output:
```bash
Epoch 0: 100%|███████████████████████████████████████████████| 273/273 [01:29<00:00,  3.04it/s, v_num=3em1, val/recall=0.745, val/precision=0.939, val/hmean=0.823]
2025-09-26 11:02:43
Parallel Evaluation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:01<00:00,  8.48it/s]
2025-09-26 11:02:43
Parallel Evaluation: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 404/404 [00:22<00:00, 18.20it/s]
2025-09-26 11:03:36
Testing DataLoader 0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 34/34 [00:34<00:00,  0.98it/s]
2025-09-26 11:03:51
Parallel Evaluation: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 404/404 [00:22<00:00, 18.05it/s]
2025-09-26 11:03:51
Parallel Evaluation: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 404/404 [00:22<00:00, 62.76it/s]
2025-09-26 11:04:50
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
2025-09-26 11:04:50
┃        Test metric        ┃       DataLoader 0        ┃
2025-09-26 11:04:50
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
2025-09-26 11:04:50
│        test/hmean         │    0.8233117461204529     │
2025-09-26 11:04:50
│      test/precision       │    0.9389364123344421     │
2025-09-26 11:04:50
│        test/recall        │    0.7451242208480835     │
2025-09-26 11:04:50
└───────────────────────────┴───────────────────────────┘
```

### alias: "history-1"
- Date: 2025-10-02 02:25:50 (recreated snapshot)
- Location: `/home/vscode/workspace/ocr-mobilnet-rewind/upstageailab-ocr-recsys-competition-ocr-2`
- Final run name: `user_model-b4-lr0e+00_loss0.0000`
- Substantially lower performing than `history-2`; W&B summary drops over half of the original metrics but still shows rotation mismatches in validation images.

Log output:
```bash
2025-10-01 17:31:57
/home/vscode/workspace/ocr-mobilnet-rewind/upstageailab-ocr-recsys-competition-ocr-2/.venv/lib/python3.11/site-packages/lightning/pytorch/callbacks/model_checkpoint.py:751: Checkpoint directory /home/vscode/workspace/ocr-mobilnet-rewind/upstageailab-ocr-recsys-competition-ocr-2/outputs/ocr_training/checkpoints exists and is not empty.
2025-10-01 17:31:57
Epoch 0: 100%|█| 103/103 [01:19<00:00,  1.30it/s, v_num=b8d3, val/recall=0.576, val/precision=0.850,
2025-10-01 17:32:07
Parallel Evaluation: 100%|██████████████████████████████████████████| 32/32 [00:01<00:00, 18.55it/s]
2025-10-01 17:32:07
Parallel Evaluation: 100%|████████████████████████████████████████| 404/404 [00:19<00:00, 20.42it/s]
2025-10-01 17:32:55
Testing DataLoader 0: 100%|█████████████████████████████████████████| 13/13 [00:28<00:00,  0.45it/s]
2025-10-01 17:33:07
Parallel Evaluation: 100%|████████████████████████████████████████| 404/404 [00:19<00:00, 20.49it/s]
2025-10-01 17:33:07
Parallel Evaluation:  99%|███████████████████████████████████████▌| 399/404 [00:19<00:00, 76.95it/s]
2025-10-01 17:34:01
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
2025-10-01 17:34:01
┃        Test metric        ┃       DataLoader 0        ┃
2025-10-01 17:34:01
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
2025-10-01 17:34:01
│        test/hmean         │    0.6776904463768005     │
2025-10-01 17:34:01
│      test/precision       │     0.850307822227478     │
2025-10-01 17:34:01
│        test/recall        │    0.5757800340652466     │
2025-10-01 17:34:01
└───────────────────────────┴───────────────────────────┘
2025-10-01 17:34:01
Finalized run name: user_model-b4-lr0e+00_loss0.0000
```

### alias: mobilenet
- Date: 2025-09-26 20:31:19
- Epoch: 15
- Final run name: `wchoi189_mobilenetv3-small-050-b8-lr1e3_loss0.0000`
- High performance but relies on many more epochs than `history-2`.
- Metrics: `test/hmean=0.8467`, `test/precision=0.9398`, `test/recall=0.7808`
- Contains both test and validation metrics, but no annotations for the official test set.

### alias: "broken-branch"
- Date: 2025-10-02 02:50:30
- Location: `/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/`
- Branch: `05_refactor/preprocessor`
- Final run name: `wchoi189_dbnet-resnet18-unet-db-head-db-loss-bs12-lr1e-3_SCORE_PLACEHOLDER`
- Command: `uv run python runners/train.py`
- W&B run path: `ocr-team2/receipt-text-recognition-ocr-project/975f844i`
- Epoch 1 metrics: `val/recall=0.0252`, `val/precision=0.7939`, `val/hmean=0.0480`
- Summary metrics: only 14 keys, missing validation image uploads.
- Architectural difference: this refactor introduces logits through `BaseHead`/`BaseLoss` wrappers, whereas legacy branches rely on a simpler DBNet stack (`UNet` decoder + original `DBHead`/`DBLoss`). Extra logits/keys likely change the loss behaviour; compatibility with `DBLoss` must be verified.

## Next Session TODO Seeds

1. **Metrics parity audit**: enumerate which W&B metrics vanished in the refactor and restore logging callbacks where necessary (especially validation media).
2. **Loss interface check**: confirm whether the extended head/loss modules expect additional targets; align with the dataset batch format or revert to legacy head/loss for baseline reproduction.
3. **Hydra config mapping**: create a table mapping critical defaults (dataset transforms, trainer intervals, callbacks) between the branches before merging any settings.
4. **Rotation mismatch follow-up**: both history snapshots show GT/Pred rotation issues. Schedule manual inspection to ensure this isn’t caused by differing preprocessors.
