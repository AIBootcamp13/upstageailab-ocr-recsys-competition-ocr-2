# **filename: docs/ai_handbook/03_references/04_evaluation_metrics.md**

# **Reference: Evaluation Metrics**

This document details the evaluation stack that powers both automated reporting and manual inspection across training, validation, and benchmarking runs.

## **1. Primary Metric: CLEval**

The official leaderboard metric for the competition is **CLEval (Character-Level Evaluation)**. Unlike IoU-based checks, CLEval compares predictions to ground truth at the character level, making it sensitive to over- or under-segmentation of text regions.

### **1.1. How CLEval Works**

1. **Polygon Matching:** Predicted text polygons are paired with ground-truth polygons. Invalid polygons (odd coordinate counts, fewer than four vertices) are skipped with a warning so they do not poison the score.
2. **Character Estimation:** CLEval estimates the number of characters represented by each ground-truth polygon.
3. **Granularity-Aware Scoring:** Precision and recall are computed from the estimated character counts with **granularity penalties** applied. These penalties down-weight predictions that merge or split characters relative to the ground truth.

### **1.2. Implementation in the Repo**

| Component | Location | Notes |
| --- | --- | --- |
| Metric implementation | `ocr.metrics.cleval_metric.CLEvalMetric` | TorchMetrics-compatible module with accumulated state storage. |
| Data containers | `ocr.metrics.data` | Dataclasses used to aggregate per-sample results and expose debug counters. |
| Lightning integration | `ocr.lightning_modules.ocr_pl.OCRPLModule` | Instantiates `CLEvalMetric` during validation/test using Hydra-driven config. |
| Parallel evaluator | `evaluate_single_sample` helper in `ocr/lightning_modules/ocr_pl.py` | Spawns per-sample workers during epoch end summarization and reuses the configured metric options. |

Hydra now owns the metric defaults via `configs/metrics/cleval.yaml`. Override any field at runtime, for example `python runners/train.py metrics.eval.case_sensitive=false metrics.eval.scale_wise=true`.

### **1.3. Configurable Parameters**

The constructor accepts the following options:

| Parameter | Default | Purpose |
| --- | --- | --- |
| `case_sensitive` | `True` | Toggle between case-sensitive and case-insensitive transcription comparison. |
| `recall_gran_penalty` / `precision_gran_penalty` | `1.0` | Weight applied to the granularity penalty; lower values reduce the penalty. |
| `vertical_aspect_ratio_thresh` | `0.5` | Threshold used to treat tall boxes as vertical text. |
| `ap_constraint` | `0.3` | Area precision constraint mirroring the CLEval reference implementation. |
| `max_polygons` | `500` | Safety guard; predictions beyond this count are ignored to prevent runaway memory usage. |
| `scale_wise` | `False` | Enables per-scale breakdowns using the provided `scale_bins`. |
| `scale_bins` | `(0.0, 0.005, ..., 1.0)` | Normalized area bins used when `scale_wise=True`. |

The defaults live in `configs/metrics/cleval.yaml` and are validated in `tests/test_metrics.py`; use those tests as a safety net when experimenting with alternative settings.

### **1.4. Reported Outputs**

`CLEvalMetric.compute()` returns a dictionary with the following keys:

* `precision` *(torch.Tensor)* – Character-level precision after penalties.
* `recall` *(torch.Tensor)* – Character-level recall after penalties.
* `f1` *(torch.Tensor)* – Harmonic mean of precision and recall; this is the leaderboard target.
* `num_splitted`, `num_merged`, `num_char_overlapped` – Debug counters that indicate when predictions were fragmented or merged relative to the ground truth.
* `scale_wise` *(dict)* – Optional nested dictionary of `{(low, high): metrics}` when `scale_wise=True`.

These values are logged by Lightning under `val/*` and `test/*` namespaces. When running the standalone evaluator, the raw dictionary is serialized so downstream scripts can perform custom aggregation.

## **2. Secondary / Debugging Metrics**

While CLEval is the official score, the training loop surfaces several supporting metrics:

* **Total Loss / Loss Components** – Reported per step from the detection head (e.g., BCE, Dice). Track these to understand convergence even when CLEval is flat.
* **Split/Merge Counters** – Sourced from CLEval’s internal stats and emitted to the logs for anomaly detection.
* **Scale-Wise F1 (Optional)** – When `scale_wise=True`, we log fine-grained F1 values per polygon size bin to pinpoint weaknesses on tiny or large text.

## **3. Practical Usage**

* **Training & Validation:** The default Lightning module (`OCRPLModule`) resets the metric between epochs, logs `val/precision`, `val/recall`, and `val/hmean`, and performs parallel evaluation for faster epoch ends.
* **Testing:** The same pipeline runs on the `test` dataloader, logging under the `test/*` namespace.
* **Ad-hoc Evaluation:** Import `CLEvalMetric` directly when benchmarking custom predictions:

```python
from ocr.metrics import CLEvalMetric

metric = CLEvalMetric(scale_wise=True)
metric(det_quads=predictions, gt_quads=ground_truth)
scores = metric.compute()
print(scores["f1"].item())
```

Run `pytest tests/test_metrics.py` after modifying the metric logic or tweaking `configs/metrics/cleval.yaml` to ensure expected behaviours (case-sensitivity, penalties, scale-wise aggregation) remain intact.
