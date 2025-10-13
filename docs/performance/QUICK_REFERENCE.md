# Performance Optimizations - Quick Reference

**Last Updated**: 2025-10-13
**Status**: ✅ Production Ready

---

## 🚀 TL;DR - Quick Start

**Default config is already optimized!** Just run:
```bash
uv run python runners/train.py
```

All performance features are **enabled by default** and will give you **4.5-6x speedup**.

---

## 📊 Performance Features Summary

| Feature | Speedup | Memory Cost | Status |
|---------|---------|-------------|--------|
| **Mixed Precision (FP16)** | 2.0x | None | ✅ Always on |
| **RAM Image Caching** | 1.12x | ~200MB | ✅ Val dataset only |
| **Tensor Caching** | 2.5-3.0x | ~800MB-1.2GB | ✅ Val dataset only |
| **Combined** | **4.5-6x** | ~1-1.4GB | ✅ Enabled |

### Expected Timings

| Scenario | Baseline (no opt) | Optimized | Speedup |
|----------|-------------------|-----------|---------|
| **Single epoch** | ~180-200s | ~60-70s (epoch 0) | ~3x |
| **Multi-epoch** | ~180-200s each | ~20-30s (epoch 1+) | **6-8x** 🚀 |
| **3 epochs total** | ~540-600s | ~100-130s | **4.5-6x** |

---

## ⚙️ Configuration Files

### Trainer Config
**File**: [configs/trainer/default.yaml](../../configs/trainer/default.yaml)
```yaml
trainer:
  precision: "16-mixed"  # FP16 mixed precision (2x speedup)
  benchmark: true        # CUDNN auto-tuner
```

### Data Config
**File**: [configs/data/base.yaml](../../configs/data/base.yaml)
```yaml
datasets:
  val_dataset:
    config:
      image_path: ${dataset_base_path}images_val_canonical  # Optimized path
      preload_images: true     # RAM caching (1.12x speedup)
      load_maps: true           # Load pre-computed maps
      cache_config:
        cache_transformed_tensors: true  # Tensor caching (2.5-3x speedup)
        cache_images: true
        cache_maps: true
        log_statistics_every_n: 100  # Cache stats logging
```

---

## 🔧 Common Operations

### Disable All Optimizations (for debugging)
```bash
uv run python runners/train.py \
  trainer.precision=32-true \
  datasets.val_dataset.config.preload_images=false \
  datasets.val_dataset.config.cache_config.cache_transformed_tensors=false
```

### Disable Specific Features

**Disable tensor caching only:**
```bash
uv run python runners/train.py \
  datasets.val_dataset.config.cache_config.cache_transformed_tensors=false
```

**Disable RAM preloading only:**
```bash
uv run python runners/train.py \
  datasets.val_dataset.config.preload_images=false
```

**Use FP32 precision:**
```bash
uv run python runners/train.py \
  trainer.precision=32-true
```

### Monitor Cache Performance

**Enable verbose cache statistics:**
```bash
uv run python runners/train.py \
  datasets.val_dataset.config.cache_config.log_statistics_every_n=50
```

**Check config resolution:**
```bash
uv run python runners/train.py --cfg job --resolve | grep -A 10 "cache_config:"
```

---

## 📋 Verification Checklist

When running training, you should see:

### ✅ Phase 1: Mixed Precision
```
Using 16bit Automatic Mixed Precision (AMP)
```

### ✅ Phase 2: Image Preloading
```
Preloading images from .../images_val_canonical into RAM...
Loading images to RAM: 100%|██████████| 404/404 [00:10<00:00, 38.53it/s]
Preloaded 404/404 images into RAM (100.0%)
```

### ✅ Phase 3: Tensor Caching
```
Cache Statistics - Hits: 380, Misses: 24, Hit Rate: 94.1%
Image Cache Size: 404, Tensor Cache Size: 380, Maps Cache Size: 404
```

### ✅ Performance Gains
- **Epoch 0**: ~60-70s (normal - building cache)
- **Epoch 1+**: ~20-30s (fast - using cache!) 🚀

---

## 🐛 Troubleshooting

### Problem: No speedup after epoch 0

**Check:**
```bash
# Verify tensor caching is enabled
uv run python runners/train.py --cfg job --resolve | grep "cache_transformed_tensors"
```

**Should see:**
```yaml
cache_transformed_tensors: true
```

**If false:** Check `configs/data/base.yaml` line 30.

---

### Problem: Out of memory errors

**Solutions:**

1. **Disable tensor caching** (saves ~800MB-1.2GB):
   ```bash
   uv run python runners/train.py \
     datasets.val_dataset.config.cache_config.cache_transformed_tensors=false
   ```

2. **Reduce batch size**:
   ```bash
   uv run python runners/train.py \
     dataloaders.val_dataloader.batch_size=8
   ```

3. **Use full FP32** (uses more memory but might help with fragmentation):
   ```bash
   uv run python runners/train.py \
     trainer.precision=32-true
   ```

---

### Problem: hmean=0.0 (no predictions)

**This is NOT a caching issue!** This means training didn't happen.

**Check:**
- ❌ Do NOT use `trainer.limit_train_batches=0` (this skips training!)
- ✅ Use `trainer.limit_train_batches=null` or omit it entirely
- ✅ Verify model weights are loading correctly

**Test with optimizations disabled:**
```bash
uv run python runners/train.py \
  trainer.precision=32-true \
  datasets.val_dataset.config.preload_images=false \
  datasets.val_dataset.config.cache_config.cache_transformed_tensors=false \
  trainer.max_epochs=1 \
  trainer.limit_val_batches=10
```

If still hmean=0.0, it's a model/training issue, not a caching issue.

---

### Problem: "Fallback to on-the-fly generation" warnings

**Means image preloading didn't work.**

**Check:**
1. Verify canonical path exists:
   ```bash
   ls -la data/datasets/images_val_canonical | head -5
   ```

2. Check config:
   ```bash
   uv run python runners/train.py --cfg job --resolve | grep "preload_images"
   ```

3. Check for errors in logs during startup

---

### Problem: Cache hit rate is low (<50%)

**Normal for epoch 0** (building cache).

**Should improve in epoch 1+** to >90%.

**If still low in epoch 1:**
- Check `log_statistics_every_n` isn't too high (set to 50-100)
- Verify `cache_transformed_tensors: true` is set
- Check for errors in cache manager logs

---

## 📚 Related Documentation

- **Full Feature Documentation**: [13_performance_optimization_restoration.md](../ai_handbook/05_changelog/2025-10/13_performance_optimization_restoration.md)
- **Benchmark Commands**: [BENCHMARK_COMMANDS.md](./BENCHMARK_COMMANDS.md)
- **Cache Verification Guide**: [CACHE_VERIFICATION_GUIDE.md](./CACHE_VERIFICATION_GUIDE.md)
- **CHANGELOG Entry**: [docs/CHANGELOG.md](../CHANGELOG.md) (search "Performance Optimization Restoration")

---

## 🧪 Performance Benchmarking

### Quick Benchmark (Single Epoch)
```bash
# Baseline
uv run python runners/train.py \
  exp_name=quick_baseline \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=1 \
  trainer.limit_val_batches=50 \
  trainer.precision=32-true \
  datasets.val_dataset.config.preload_images=false \
  datasets.val_dataset.config.cache_config.cache_transformed_tensors=false \
  logger.wandb.enabled=false

# Optimized
uv run python runners/train.py \
  exp_name=quick_optimized \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=1 \
  trainer.limit_val_batches=50 \
  logger.wandb.enabled=false
```

**Expected**: Optimized ~3x faster (~60s vs ~180s)

### Full Benchmark (3 Epochs - Shows True Speedup)
```bash
# Baseline
uv run python runners/train.py \
  exp_name=benchmark_baseline \
  trainer.max_epochs=3 \
  trainer.limit_val_batches=50 \
  trainer.precision=32-true \
  datasets.val_dataset.config.preload_images=false \
  datasets.val_dataset.config.cache_config.cache_transformed_tensors=false \
  logger.wandb.enabled=false \
  seed=42

# Optimized
uv run python runners/train.py \
  exp_name=benchmark_optimized \
  trainer.max_epochs=3 \
  trainer.limit_val_batches=50 \
  logger.wandb.enabled=false \
  seed=42
```

**Expected**:
- Baseline: ~540-600s total
- Optimized: ~100-130s total
- **Speedup: 4.5-6x** 🚀

---

## 💡 Tips & Best Practices

### When to Disable Optimizations

1. **Debugging data issues**: Disable caching to ensure fresh data loads
2. **Limited GPU memory**: Disable tensor caching
3. **Numerical debugging**: Use FP32 to rule out precision issues
4. **First-time setup**: Run baseline first to verify everything works

### When to Keep Optimizations

1. **Multi-epoch training**: Maximum benefit from tensor caching
2. **Hyperparameter tuning**: Faster iterations
3. **Final training runs**: Best performance for production models
4. **Validation-heavy workflows**: Preloading eliminates I/O bottleneck

### Memory Guidelines

- **8GB GPU**: Can run optimized config with batch_size=16
- **6GB GPU**: May need to disable tensor caching or reduce batch size
- **4GB GPU**: Disable tensor caching, use batch_size=8, consider FP32

---

## 🔍 Quick Commands Reference

```bash
# Check if optimizations are enabled
uv run python runners/train.py --cfg job --resolve | grep -A 10 "cache_config"

# Run with verbose cache logging
uv run python runners/train.py datasets.val_dataset.config.cache_config.log_statistics_every_n=10

# Verify image path exists
ls -la data/datasets/images_val_canonical

# Check recent training logs
tail -100 outputs/ocr_training/train.log | grep -E "Preload|Cache|16bit"

# Quick performance test (1 epoch)
uv run python runners/train.py trainer.max_epochs=1 trainer.limit_val_batches=50 logger.wandb.enabled=false
```

---

**Questions or Issues?**
- Check: [CACHE_VERIFICATION_GUIDE.md](./CACHE_VERIFICATION_GUIDE.md)
- Read: [Full Feature Documentation](../ai_handbook/05_changelog/2025-10/13_performance_optimization_restoration.md)
- Search: [CHANGELOG.md](../CHANGELOG.md) for "Performance Optimization Restoration"
