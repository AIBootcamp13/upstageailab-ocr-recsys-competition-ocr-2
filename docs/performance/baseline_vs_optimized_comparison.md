# Performance Comparison: Baseline vs Optimized

**Generated:** 2025-10-14 02:43:38

## Run Overview

| Configuration | Run ID | Run Name | Status | Created |
|---------------|--------|----------|--------|---------|
| **Baseline (No Optimizations)** | [b1bipuoz](https://wandb.ai/runs/b1bipuoz) | wchoi189_dbnet-resnet18-unet-dbhead-dbloss-bs16-lr1e-3_hmean0.619 | finished | 2025-10-13T15:48:48 |
| **Optimized (Full Caching)** | [9evam0xb](https://wandb.ai/runs/9evam0xb) | wchoi189_dbnet-resnet18-unet-dbhead-dbloss-bs16-lr1e-3_hmean0.705 | finished | 2025-10-13T17:11:46 |

## Configuration Comparison

| Setting | Baseline | Optimized |
|---------|----------|-----------|
| **Precision** | 16-mixed | 32-true |
| **Max Epochs** | 3 | 3 |
| **Batch Size** | 16 | 16 |
| **Image Preloading** | ✅ | ❌ |
| **Tensor Caching** | ✅ | ❌ |
| **Image Caching** | ✅ | ✅ |
| **Maps Caching** | ✅ | ❌ |

## Validation Performance Comparison

| Metric | Baseline | Optimized | Difference |
|--------|----------|-----------|------------|
| **Total Validation Time** | 17.63s | 19.23s | 0.9x slower |
| **Mean Batch Time** | 592.4ms | 654.1ms | 10.4% slower |
| **P95 Batch Time** | 902.0ms | 998.2ms | - |
| **GPU Memory** | 0.06GB | 0.06GB | - |

## Training Metrics Comparison

| Metric | Baseline | Optimized | Difference |
|--------|----------|-----------|------------|
| **Validation H-mean** | 0.7816 | 0.8839 | Optimized H-mean is 13.1% higher than baseline |
| **Validation Precision** | 0.8746 | 0.9233 | - |
| **Validation Recall** | 0.7546 | 0.8638 | - |
| **Training Loss** | 2.1851 | 2.2133 | - |
| **Validation Loss** | 2.0628 | 2.0428 | - |
| **Epoch** | 3 | 3 | - |

## Key Findings

- ❌ **Speed Regression**: Optimized run is 1.1x slower than baseline
- ✅ **Performance Maintained**: H-mean performance is comparable between runs

## Recommendations

- ❌ **Investigate Speed Regression**: The optimized run is slower than expected. Check if caching is actually enabled and working properly.
