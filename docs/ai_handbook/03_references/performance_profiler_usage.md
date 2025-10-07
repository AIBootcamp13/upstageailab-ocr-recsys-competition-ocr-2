# Performance Profiler Usage Guide

This guide explains how to use the baseline report generator to profile and analyze OCR model performance.

## Overview

The baseline report generator creates comprehensive performance reports from WandB profiling runs, helping identify bottlenecks and track optimization progress.

## Prerequisites

- WandB account and API access
- Trained OCR model checkpoint
- Performance profiler callback configured

## Step 1: Run Performance Profiling

Execute a test run with performance profiling enabled:

```bash
uv run python runners/test.py \
  data=canonical \
  checkpoint_path="/path/to/your/checkpoint.ckpt" \
  callbacks=performance_profiler \
  callbacks.performance_profiler.verbose=true \
  project_name=OCR_Performance_Baseline \
  exp_name=baseline_profiling_YYYY_MM_DD
```

### Key Parameters

- `callbacks=performance_profiler`: Enables the performance profiling callback
- `callbacks.performance_profiler.verbose=true`: Shows timing output during execution
- `project_name`: WandB project name for organizing runs
- `exp_name`: Descriptive name for the profiling experiment

### Expected Output

During execution, you'll see timing information like:
```
Validation batch 0: 0.690s
Validation batch 10: 0.378s
=== Validation Performance Summary ===
Epoch time: 16.29s
Batch times: mean=0.436s, median=0.423s, p95=0.617s
```

## Step 2: Generate Performance Report

After the profiling run completes, generate the baseline report:

```bash
uv run python scripts/performance/generate_baseline_report.py \
  --run-id <wandb_run_id> \
  --project <wandb_project_name> \
  --entity <wandb_entity> \
  --output docs/performance/baseline_YYYY-MM-DD.md
```

### Required Parameters

- `--run-id`: WandB run ID (found in run URL, e.g., `zr90z4cu`)
- `--project`: WandB project name
- `--entity`: WandB username/entity
- `--output`: Output path for the markdown report

### Optional Parameters

- `--export-json`: Export raw metrics as JSON file

### Example Command

```bash
uv run python scripts/performance/generate_baseline_report.py \
  --run-id zr90z4cu \
  --project receipt-text-recognition-ocr-project \
  --entity ocr-team2 \
  --output docs/performance/baseline_2025-10-07.md \
  --export-json docs/performance/raw_metrics.json
```

## Report Contents

The generated report includes:

### Validation Performance Metrics
- Total validation time
- Number of batches processed
- Batch time statistics (mean, median, P95, P99)
- Batch time standard deviation

### Memory Usage
- GPU memory allocated
- GPU memory reserved
- CPU memory percentage

### Bottleneck Analysis
- Identification of performance bottlenecks
- Variance analysis in batch processing times

### Recommendations
- Suggested optimization strategies
- Next steps based on performance plan

## Example Report Output

```
🔍 Fetching metrics from WandB run: zr90z4cu
📊 Analyzing bottlenecks...
📝 Generating markdown report...
✅ Report saved to: docs/performance/baseline_2025-10-07.md
✨ Baseline report complete!
```

## Sample Metrics

A typical baseline report shows metrics like:
- **Total validation time**: ~16 seconds for 34 batches
- **Mean batch time**: ~436ms
- **P95 batch time**: ~617ms
- **Memory usage**: GPU (~0.06 GB), CPU (~7.8%)

## Troubleshooting

### No Performance Metrics Found
- Ensure the performance profiler callback was properly enabled
- Check that the WandB run completed successfully
- Verify the run ID and project name are correct

### Callback Not Instantiated
- Confirm `callbacks=performance_profiler` is specified
- Check that the callback config file exists: `configs/callbacks/performance_profiler.yaml`

### Memory Metrics Missing
- Ensure GPU is available if profiling GPU memory
- Check that `profile_memory=true` in callback config

## Integration with Performance Optimization Plan

This tool supports the phased performance optimization approach:

1. **Phase 1.1**: PyClipper caching - Use baseline to measure improvement
2. **Phase 1.2**: Parallel processing - Compare before/after metrics
3. **Phase 2**: Memory optimization - Track memory usage reductions
4. **Phase 3**: Automated profiling - Continuous performance monitoring

## Related Files

- `scripts/performance/generate_baseline_report.py`: Main report generator
- `ocr/lightning_modules/callbacks/performance_profiler.py`: Profiling callback
- `configs/callbacks/performance_profiler.yaml`: Callback configuration
- `docs/performance/`: Directory for generated reports
