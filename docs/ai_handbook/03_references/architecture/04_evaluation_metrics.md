# **filename: docs/ai_handbook/03_references/architecture/04_evaluation_metrics.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=understanding_evaluation_metrics,configuring_CLEval_parameters,debugging_model_performance,interpreting_competition_scores -->

# **Reference: Evaluation Metrics**

This reference document provides comprehensive information about the evaluation metrics used in the OCR project for quick lookup and detailed understanding.

## **Overview**

This reference details the evaluation stack that powers both automated reporting and manual inspection across training, validation, and benchmarking runs. The primary metric is CLEval (Character-Level Evaluation), the official leaderboard metric for the competition that compares predictions to ground truth at the character level.

## **Key Concepts**

### **CLEval (Character-Level Evaluation)**
- **Primary Metric**: Official competition leaderboard score that evaluates text detection at character level
- **Polygon Matching**: Pairs predicted and ground-truth text polygons for character estimation
- **Granularity Penalties**: Down-weights predictions that merge or split characters relative to ground truth
- **Character Estimation**: Estimates character counts from polygon areas for precision/recall calculation

### **Evaluation Pipeline**
1. Polygon validation and filtering (invalid polygons skipped)
2. Character count estimation from polygon areas
3. Precision/recall calculation with granularity penalties
4. F1 score computation as harmonic mean
5. Optional scale-wise breakdown by polygon size

### **Supporting Metrics**
- **Loss Components**: BCE, Dice losses for training convergence monitoring
- **Debug Counters**: Split/merge statistics for anomaly detection
- **Scale-wise F1**: Per-size-bin F1 scores for targeted optimization

## **Detailed Information**

### **How CLEval Works**
1. **Polygon Matching**: Predicted text polygons are paired with ground-truth polygons. Invalid polygons (odd coordinate counts, fewer than four vertices) are skipped with a warning so they do not poison the score.
2. **Character Estimation**: CLEval estimates the number of characters represented by each ground-truth polygon.
3. **Granularity-Aware Scoring**: Precision and recall are computed from the estimated character counts with **granularity penalties** applied. These penalties down-weight predictions that merge or split characters relative to the ground truth.

### **Implementation Components**
| Component | Location | Notes |
| --- | --- | --- |
| Metric implementation | `ocr.metrics.cleval_metric.CLEvalMetric` | TorchMetrics-compatible module with accumulated state storage. |
| Data containers | `ocr.metrics.data` | Dataclasses used to aggregate per-sample results and expose debug counters. |
| Lightning integration | `ocr.lightning_modules.ocr_pl.OCRPLModule` | Instantiates `CLEvalMetric` during validation/test using Hydra-driven config. |
| Parallel evaluator | `evaluate_single_sample` helper in `ocr/lightning_modules/ocr_pl.py` | Spawns per-sample workers during epoch end summarization and reuses the configured metric options. |

### **Reported Outputs**
`CLEvalMetric.compute()` returns a dictionary with the following keys:

* `precision` *(torch.Tensor)* – Character-level precision after penalties.
* `recall` *(torch.Tensor)* – Character-level recall after penalties.
* `f1` *(torch.Tensor)* – Harmonic mean of precision and recall; this is the leaderboard target.
* `num_splitted`, `num_merged`, `num_char_overlapped` – Debug counters that indicate when predictions were fragmented or merged relative to the ground truth.
* `scale_wise` *(dict)* – Optional nested dictionary of `{(low, high): metrics}` when `scale_wise=True`.

These values are logged by Lightning under `val/*` and `test/*` namespaces. When running the standalone evaluator, the raw dictionary is serialized so downstream scripts can perform custom aggregation.

## **Examples**

### **Training & Validation Integration**
The default Lightning module (`OCRPLModule`) resets the metric between epochs, logs `val/precision`, `val/recall`, and `val/hmean`, and performs parallel evaluation for faster epoch ends.

### **Testing Pipeline**
The same pipeline runs on the `test` dataloader, logging under the `test/*` namespace.

### **Ad-hoc Evaluation**
Import `CLEvalMetric` directly when benchmarking custom predictions:

```python
from ocr.metrics import CLEvalMetric

metric = CLEvalMetric(scale_wise=True)
metric(det_quads=predictions, gt_quads=ground_truth)
scores = metric.compute()
print(scores["f1"].item())
```

## **Configuration Options**

### **CLEval Parameters**
| Parameter | Default | Purpose |
| --- | --- | --- |
| `case_sensitive` | `True` | Toggle between case-sensitive and case-insensitive transcription comparison. |
| `recall_gran_penalty` / `precision_gran_penalty` | `1.0` | Weight applied to the granularity penalty; lower values reduce the penalty. |
| `vertical_aspect_ratio_thresh` | `0.5` | Threshold used to treat tall boxes as vertical text. |
| `ap_constraint` | `0.3` | Area precision constraint mirroring the CLEval reference implementation. |
| `max_polygons` | `500` | Safety guard; predictions beyond this count are ignored to prevent runaway memory usage. |
| `scale_wise` | `False` | Enables per-scale breakdowns using the provided `scale_bins`. |
| `scale_bins` | `(0.0, 0.005, ..., 1.0)` | Normalized area bins used when `scale_wise=True`. |

### **Configuration File**
Hydra owns the metric defaults via `configs/metrics/cleval.yaml`. Override any field at runtime:

```bash
python runners/train.py metrics.eval.case_sensitive=false metrics.eval.scale_wise=true
```

## **Best Practices**

### **Validation Testing**
Run `pytest tests/test_metrics.py` after modifying the metric logic or tweaking `configs/metrics/cleval.yaml` to ensure expected behaviours (case-sensitivity, penalties, scale-wise aggregation) remain intact.

### **Monitoring During Training**
- **Track Loss Components**: Monitor BCE, Dice losses for training convergence even when CLEval is flat
- **Watch Debug Counters**: Use split/merge statistics for anomaly detection
- **Enable Scale-wise F1**: Use `scale_wise=True` to pinpoint weaknesses on tiny or large text

### **Performance Optimization**
- **Parallel Evaluation**: Use the parallel evaluator for faster epoch ends during training
- **Memory Safety**: The `max_polygons` limit prevents runaway memory usage
- **Invalid Polygon Handling**: Invalid polygons are skipped with warnings to prevent score poisoning

## **Troubleshooting**

### **CLEval Score Issues**
**Problem**: CLEval scores are unexpectedly low

**Solutions**:
- Check for invalid polygons in predictions (odd coordinate counts, <4 vertices)
- Verify character estimation is working correctly
- Examine granularity penalties - they may be too harsh

### **Scale-wise Evaluation Problems**
**Problem**: Scale-wise breakdowns not working

**Solutions**:
- Ensure `scale_wise=True` is set in configuration
- Check that `scale_bins` are properly defined
- Verify normalized area calculations are correct

### **Memory Issues**
**Problem**: Evaluation runs out of memory

**Solutions**:
- Reduce `max_polygons` limit (default: 500)
- Process fewer samples per batch
- Disable scale-wise evaluation if not needed

### **Logging Issues**
**Problem**: Metrics not appearing in logs

**Solutions**:
- Check Lightning namespace (`val/*`, `test/*`)
- Verify metric is properly instantiated in `OCRPLModule`
- Ensure Hydra configuration is loading correctly

## **Related References**

- **CLEval Implementation**: `ocr/metrics/cleval_metric.py`
- **Data Containers**: `ocr/metrics/data.py`
- **Lightning Integration**: `ocr/lightning_modules/ocr_pl.py`
- **Configuration**: `configs/metrics/cleval.yaml`
- **Tests**: `tests/test_metrics.py`
- **CLEval Paper**: Character-level evaluation for text detection
- **Competition Leaderboard**: Official scoring methodology

---

*This document follows the references template. Last updated: October 13, 2025*
