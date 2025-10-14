# Performance Comparison: Baseline vs Optimized

**Generated:** 2025-10-14 02:44:08

## Run Overview

| Configuration | Run ID | Run Name | Status | Created |
|---------------|--------|----------|--------|---------|
| **Baseline (No Optimizations)** | [9evam0xb](https://wandb.ai/runs/9evam0xb) | wchoi189_dbnet-resnet18-unet-dbhead-dbloss-bs16-lr1e-3_hmean0.705 | finished | 2025-10-13T17:11:46 |
| **Optimized (Full Caching)** | [b1bipuoz](https://wandb.ai/runs/b1bipuoz) | wchoi189_dbnet-resnet18-unet-dbhead-dbloss-bs16-lr1e-3_hmean0.619 | finished | 2025-10-13T15:48:48 |

## Configuration Comparison

| Setting | Baseline | Optimized |
|---------|----------|-----------|
| **Precision** | 32-true | 16-mixed |
| **Max Epochs** | 3 | 3 |
| **Batch Size** | 16 | 16 |
| **Image Preloading** | ❌ | ✅ |
| **Tensor Caching** | ❌ | ✅ |
| **Image Caching** | ✅ | ✅ |
| **Maps Caching** | ❌ | ✅ |

## Validation Performance Comparison

| Metric | Baseline | Optimized | Difference |
|--------|----------|-----------|------------|
| **Total Validation Time** | 19.23s | 17.63s | 1.1x faster |
| **Mean Batch Time** | 654.1ms | 592.4ms | 9.4% faster |
| **P95 Batch Time** | 998.2ms | 902.0ms | - |
| **GPU Memory** | 0.06GB | 0.06GB | - |

## Training Metrics Comparison

| Metric | Baseline | Optimized | Difference |
|--------|----------|-----------|------------|
| **Validation H-mean** | 0.8839 | 0.7816 | Optimized H-mean is 11.6% lower than baseline |
| **Validation Precision** | 0.9233 | 0.8746 | - |
| **Validation Recall** | 0.8638 | 0.7546 | - |
| **Training Loss** | 2.2133 | 2.1851 | - |
| **Validation Loss** | 2.0428 | 2.0628 | - |
| **Epoch** | 3 | 3 | - |

## Key Findings

- ✅ **Speed Improvement**: Optimized run is 1.1x faster than baseline
- ❌ **Performance Degradation**: Optimized run has 11.6% lower H-mean

## Recommendations

- ❌ **Investigate Performance Drop**: The optimized run shows significantly lower performance. Verify that the same dataset and model are being used.
