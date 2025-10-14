# FP16 Mixed Precision Training Guide

**Version**: 1.0
**Date**: 2025-10-14
**Status**: Experimental (Validation Required)

## Overview

This guide covers 16-bit mixed precision (FP16) training for the DBNet OCR architecture using PyTorch Lightning's automatic mixed precision (AMP) support.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Performance vs Accuracy](#performance-vs-accuracy)
- [Validation Process](#validation-process)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Quick Start

### Using FP16 Configuration

```bash
# Use the safe FP16 trainer configuration
uv run python runners/train.py trainer=fp16_safe

# Or override precision directly
uv run python runners/train.py trainer.precision="16-mixed"
```

### Validation Test

Before using FP16 in production, validate accuracy:

```bash
# 1. Baseline FP32 run (3 epochs)
uv run python runners/train.py \
  trainer.max_epochs=3 \
  trainer.limit_val_batches=50 \
  exp_name=fp32_baseline \
  logger.wandb.enabled=true

# 2. FP16 test run (3 epochs)
uv run python runners/train.py \
  trainer=fp16_safe \
  trainer.max_epochs=3 \
  trainer.limit_val_batches=50 \
  exp_name=fp16_test \
  logger.wandb.enabled=true

# 3. Compare H-mean scores
# Acceptable difference: < 1.0% (e.g., 0.8863 vs 0.8850)
# Unacceptable difference: > 2.0% (indicates numerical instability)
```

## Configuration

### Safe FP16 Configuration (`configs/trainer/fp16_safe.yaml`)

```yaml
trainer:
  precision: "16-mixed"              # Enable FP16 with automatic gradient scaling
  gradient_clip_val: 5.0             # CRITICAL for FP16 stability
  gradient_clip_algorithm: "norm"     # Norm-based clipping
  accumulate_grad_batches: 2         # Effective batch size = batch_size * 2
  benchmark: true                    # Enable cudnn.benchmark
```

### Key Parameters

#### 1. Precision
```yaml
precision: "16-mixed"
```
- **Effect**: Enables automatic mixed precision training
- **Gradient Scaling**: Automatically handled by PyTorch Lightning
- **Loss Scaling**: Dynamic loss scaling prevents underflow

#### 2. Gradient Clipping (CRITICAL!)
```yaml
gradient_clip_val: 5.0
gradient_clip_algorithm: "norm"
```
- **Why Critical**: FP16 has reduced numerical range, gradient explosion more common
- **Recommended Value**: 5.0 (same as FP32 but more important)
- **Algorithm**: "norm" (L2 norm) or "value" (absolute value)

#### 3. Gradient Accumulation
```yaml
accumulate_grad_batches: 2
```
- **Purpose**: Simulate larger batch sizes for stability
- **Trade-off**: 2x slower training, but more stable gradients
- **Recommendation**: Keep at 2 for FP16

## Performance vs Accuracy

### Benchmark Results (PRELIMINARY - REQUIRES VALIDATION)

| Configuration | H-mean | Epoch Time | Memory | Status |
|---------------|--------|------------|---------|---------|
| FP32 Baseline | 0.8863 | 19m 39s | 4.2 GB | ✅ Stable |
| FP16 (No validation) | 0.7816 | 16m 44s | 2.9 GB | ❌ Degraded (11.8% drop) |
| FP16 + Safe Config | TBD | TBD | TBD | ⚠️ Needs validation |

### Expected Performance

**Speed**:
- ~15-20% faster training
- ~30% memory reduction
- Allows larger batch sizes

**Accuracy**:
- Target: < 1% H-mean difference vs FP32
- Acceptable: < 2% difference
- Unacceptable: > 2% difference (numerical instability)

## Validation Process

### Step 1: Baseline Establishment

Run FP32 training to establish baseline:

```bash
uv run python runners/train.py \
  trainer.max_epochs=3 \
  trainer.limit_train_batches=0 \
  trainer.limit_val_batches=50 \
  exp_name=fp32_baseline_$(date +%Y%m%d) \
  logger.wandb.enabled=true \
  trainer.precision="32-true"
```

**Expected Output**:
```
val/hmean: 0.8863 ± 0.01  # Target baseline
```

### Step 2: FP16 Validation

Run FP16 with safe configuration:

```bash
uv run python runners/train.py \
  trainer=fp16_safe \
  trainer.max_epochs=3 \
  trainer.limit_train_batches=0 \
  trainer.limit_val_batches=50 \
  exp_name=fp16_validation_$(date +%Y%m%d) \
  logger.wandb.enabled=true
```

**Monitor For**:
- NaN/Inf values in loss
- Gradient norms (should be similar to FP32)
- Validation metrics stability

### Step 3: Comparison Analysis

```python
# Use WandB or compare logs
import wandb

api = wandb.Api()

# Get runs
fp32_run = api.run("ocr-team2/ocr-competition/RUN_ID_FP32")
fp16_run = api.run("ocr-team2/ocr-competition/RUN_ID_FP16")

# Compare final metrics
fp32_hmean = fp32_run.summary["val/hmean"]
fp16_hmean = fp16_run.summary["val/hmean"]

diff_percent = abs(fp32_hmean - fp16_hmean) / fp32_hmean * 100

print(f"FP32 H-mean: {fp32_hmean:.4f}")
print(f"FP16 H-mean: {fp16_hmean:.4f}")
print(f"Difference: {diff_percent:.2f}%")

if diff_percent < 1.0:
    print("✅ PASS: FP16 validated for production use")
elif diff_percent < 2.0:
    print("⚠️  MARGINAL: Consider additional validation")
else:
    print("❌ FAIL: FP16 not recommended (numerical instability)")
```

### Step 4: Full Dataset Validation

If steps 1-3 pass, validate on full dataset:

```bash
uv run python runners/train.py \
  trainer=fp16_safe \
  trainer.max_epochs=10 \
  exp_name=fp16_full_validation \
  logger.wandb.enabled=true
```

## Troubleshooting

### Issue: NaN/Inf Loss Values

**Symptoms**:
```
Epoch 0: train/loss=NaN
RuntimeError: Function 'PowBackward0' returned nan values
```

**Solutions**:
1. **Increase gradient clipping**: `gradient_clip_val: 10.0`
2. **Enable gradient accumulation**: `accumulate_grad_batches: 4`
3. **Reduce learning rate**: `lr: 0.0005` (half of default)
4. **Fall back to FP32**: `precision: "32-true"`

### Issue: Accuracy Degradation > 2%

**Symptoms**:
```
FP32: val/hmean=0.8863
FP16: val/hmean=0.8650  # 2.4% drop - unacceptable
```

**Root Causes**:
1. **Gradient underflow**: Gradients too small for FP16 representation
2. **Loss scaling issues**: Automatic scaling insufficient
3. **Model architecture**: Some ops not FP16-compatible

**Solutions**:
1. **Check gradient norms**:
```python
# Add to training loop
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm < 1e-6:
            print(f"⚠️  Small gradient: {name} = {grad_norm}")
```

2. **Verify loss scaling**:
```python
# PyTorch Lightning handles this automatically
# Check logs for "GradScaler" warnings
```

3. **Identify incompatible layers**:
```python
# Some custom ops may need FP32
# Use autocast context for problematic sections
with torch.cuda.amp.autocast(enabled=False):
    output = problematic_layer(input.float())
```

### Issue: Training Slower Than FP32

**Symptoms**:
```
FP32: 19m 39s/epoch
FP16: 21m 15s/epoch  # Slower!
```

**Causes**:
1. **Tensor Core not utilized**: Model ops not FP16-optimized
2. **Overhead from type conversions**: Too many FP16↔FP32 conversions
3. **Small batch size**: Tensor Cores need larger batches

**Solutions**:
1. **Enable Tensor Cores**:
```python
torch.set_float32_matmul_precision("medium")  # or "high"
```

2. **Increase batch size**:
```yaml
batch_size: 32  # Double from 16
```

3. **Optimize model architecture**:
   - Use operations with Tensor Core support
   - Minimize dtype conversions

### Issue: Memory Usage Not Reduced

**Symptoms**:
```
FP32: 4.2 GB GPU memory
FP16: 4.1 GB GPU memory  # Expected ~2.9 GB
```

**Causes**:
1. **Optimizer states in FP32**: Adam keeps FP32 copies
2. **Activations not reduced**: Backward pass uses FP32
3. **Model weights duplicated**: Training keeps FP32 master weights

**Note**: This is expected behavior. FP16 training maintains FP32 master weights for numerical stability.

## Best Practices

### 1. Always Validate Before Production

```bash
# Never deploy FP16 without validation
❌ uv run python runners/train.py trainer=fp16_safe  # Don't do this first!

# Always validate first
✅ ./scripts/validate_fp16.sh  # Run validation script
✅ Review comparison results
✅ Approve for production use
```

### 2. Monitor Gradient Norms

```python
# Add to callbacks
class GradientMonitor(Callback):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        if total_norm > 100:  # Warning threshold
            print(f"⚠️  Large gradient norm: {total_norm:.2f}")
```

### 3. Use Conservative Settings Initially

```yaml
# Start with safe settings
trainer:
  precision: "16-mixed"
  gradient_clip_val: 10.0        # More conservative
  accumulate_grad_batches: 4     # Larger effective batch

  # Only reduce after validation
  # gradient_clip_val: 5.0
  # accumulate_grad_batches: 2
```

### 4. Document Validation Results

Create validation report:

```markdown
# FP16 Validation Report

**Date**: 2025-10-14
**Configuration**: configs/trainer/fp16_safe.yaml

## Results

| Metric | FP32 | FP16 | Difference |
|--------|------|------|------------|
| H-mean | 0.8863 | 0.8850 | -0.15% ✅ |
| Training Time | 19m 39s | 16m 44s | -15.0% ✅ |
| Memory Usage | 4.2 GB | 2.9 GB | -30.9% ✅ |

## Approval

✅ Approved for production use
Signed off by: [Name]
Date: [Date]
```

### 5. Have Rollback Plan

```bash
# If FP16 causes issues in production:
# 1. Immediately switch back to FP32
uv run python runners/train.py trainer.precision="32-true"

# 2. Document the issue
echo "FP16 failure: [description]" >> docs/performance/fp16_issues.md

# 3. Investigate root cause
```

## Implementation Checklist

Before enabling FP16 in production:

- [ ] Run FP32 baseline (3 epochs, full validation set)
- [ ] Run FP16 test (3 epochs, full validation set)
- [ ] Verify accuracy difference < 1%
- [ ] Check for NaN/Inf values in logs
- [ ] Validate gradient norms are stable
- [ ] Confirm speedup > 10%
- [ ] Document validation results
- [ ] Get team approval
- [ ] Test on full training run (10+ epochs)
- [ ] Monitor production metrics closely

## Related Documentation

- [BUG_2025_002_MIXED_PRECISION_PERFORMANCE.md](../bug_reports/BUG_2025_002_MIXED_PRECISION_PERFORMANCE.md) - Original FP16 issue
- [baseline_vs_optimized_comparison.md](baseline_vs_optimized_comparison.md) - Performance comparison
- [configs/trainer/fp16_safe.yaml](../../configs/trainer/fp16_safe.yaml) - Safe FP16 configuration
- [configs/trainer/default.yaml](../../configs/trainer/default.yaml) - FP32 baseline

## References

- [PyTorch Lightning AMP Documentation](https://lightning.ai/docs/pytorch/stable/advanced/mixed_precision.html)
- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- [Gradient Scaling Documentation](https://pytorch.org/docs/stable/amp.html#gradient-scaling)

## Changelog

### v1.0 (2025-10-14)
- Initial FP16 training guide
- Created `fp16_safe.yaml` configuration
- Documented validation process
- Added troubleshooting section

## Future Work

- [ ] Complete full validation on DBNet architecture
- [ ] Benchmark on different GPU architectures (A100, V100, etc.)
- [ ] Investigate bfloat16 (bf16) as alternative
- [ ] Create automated validation script
- [ ] Add gradient norm monitoring callback
