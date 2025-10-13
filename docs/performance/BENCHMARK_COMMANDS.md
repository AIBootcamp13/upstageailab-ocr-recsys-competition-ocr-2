# Performance Benchmark Commands

## Quick Reference: Run These Two Commands

### 🔴 Benchmark 1: Baseline (No Optimizations)

```bash
uv run python runners/train.py \
  exp_name=benchmark_baseline_32bit_no_cache \
  logger.wandb.enabled=false \
  trainer.max_epochs=3 \
  trainer.limit_val_batches=50 \
  trainer.precision=32-true \
  datasets.val_dataset.config.preload_images=false \
  datasets.val_dataset.config.cache_config.cache_transformed_tensors=false \
  seed=42
```

**Expected:**
- 32-bit precision (slower)
- No image preloading (disk I/O every sample)
- No tensor caching (transforms run every epoch)
- Time: ~180-200s per epoch

---

### 🟢 Benchmark 2: Full Optimizations

```bash
uv run python runners/train.py \
  exp_name=benchmark_optimized_16bit_full_cache \
  logger.wandb.enabled=false \
  trainer.max_epochs=3 \
  trainer.limit_val_batches=50 \
  seed=42
```

**Expected:**
- 16-bit mixed precision (FP16 - 2x faster)
- Image preloading to RAM (eliminates disk I/O)
- Tensor caching (transforms run once, cached after)
- Epoch 0: ~60-70s (builds cache)
- Epoch 1+: ~20-30s (**6-8x speedup!**)

---

## What to Look For

### Baseline Run:
```
Using 32bit True Precision (FP32)
⚠ Fallback to on-the-fly generation: X/X samples (100.0%)
Epoch 0: ~180-200s
Epoch 1: ~180-200s  ← Same time, no caching
Epoch 2: ~180-200s  ← Same time, no caching
```

### Optimized Run:
```
Using 16bit Automatic Mixed Precision (AMP)
Preloading images from .../images_val_canonical into RAM...
Preloaded 404/404 images into RAM (100.0%)
Tensor caching enabled...
Epoch 0: ~60-70s  ← Builds tensor cache
Epoch 1: ~20-30s  ← Uses cached tensors! 🚀
Epoch 2: ~20-30s  ← Continues using cache! 🚀
Cache Statistics - Hit Rate: ~95%+
```

---

## Performance Calculation

### Expected Results:
- **Baseline Total**: ~540-600s (3 epochs × 180-200s)
- **Optimized Total**: ~100-130s (60s + 2×20-30s)
- **Speedup**: **4.5-6x faster overall**
- **Per-epoch speedup (after caching)**: **6-8x faster**

### Key Metrics to Extract:
1. Total wall clock time
2. Per-epoch timing
3. Cache hit rate (should be 95%+ on epochs 1+)
4. Final hmean (should be similar ~0.75-0.80 for both)

---

## Troubleshooting

### If hmean=0.0:
- Check that you're NOT using `limit_train_batches=0` (which skips training)
- Verify model weights are loading correctly
- Check for errors in the logs

### If no speedup seen:
- Verify "Preloaded X/X images" message appears
- Check for "Tensor caching enabled" message
- Look for cache statistics in logs
- Ensure 16-bit precision is active ("Using 16bit")

### If baseline is already fast:
- You may have accidentally enabled optimizations
- Check config with: `uv run python runners/train.py --cfg job --resolve`

---

## 🔍 Advanced Usage: Cached Images Nuances & Safety Guidelines

### Feature Compatibility Matrix

| Feature | Can Combine? | Notes |
|---------|-------------|--------|
| **Tensor Caching + Image Preloading** | ✅ **Safe** | Best performance - preload feeds tensor cache |
| **Tensor Caching + Mixed Precision** | ✅ **Safe** | Recommended - FP16 works with cached tensors |
| **Image Preloading + Mixed Precision** | ✅ **Safe** | No interaction - different optimization layers |
| **All Features Combined** | ✅ **Safe** | Default config - gives 4.5-6x speedup |

### ⚠️ Features That Should NOT Be Combined

| Feature A | Feature B | Risk | Reason |
|-----------|-----------|------|--------|
| **Tensor Caching** | **Training Dataset** | 🚨 **Data Leakage** | Cache would memorize training augmentations |
| **Image Preloading** | **Dynamic Datasets** | 🚨 **Stale Data** | Preloaded images won't reflect dataset changes |
| **Tensor Caching** | **Changing Transforms** | 🚨 **Invalid Cache** | Cached tensors use old transform parameters |

### Dataloader Applicability

#### ✅ SAFE: Validation Datasets Only
```yaml
# configs/data/base.yaml - CURRENT CONFIG (SAFE)
datasets:
  val_dataset:
    config:
      preload_images: true                    # ✅ SAFE
      cache_config:
        cache_transformed_tensors: true      # ✅ SAFE - validation only
```

#### 🚨 UNSAFE: Training Datasets
```yaml
# DO NOT DO THIS - Data leakage risk
datasets:
  train_dataset:
    config:
      preload_images: true                    # 🚨 UNSAFE - defeats augmentation
      cache_config:
        cache_transformed_tensors: true      # 🚨 UNSAFE - memorizes training data
```

#### ✅ SAFE: Test/Predict Datasets
```yaml
# SAFE - no training involved
datasets:
  test_dataset:
    config:
      preload_images: true                    # ✅ SAFE - deterministic evaluation
      cache_config:
        cache_transformed_tensors: true      # ✅ SAFE - consistent predictions
```

### Dataset Switching Risks

#### 🚨 HIGH RISK: Switching Datasets Abruptly

**Problem**: Cached tensors become invalid when switching datasets
```bash
# Run 1: Train on dataset A
uv run python runners/train.py datasets.val_dataset.config.image_path=/path/to/dataset_A

# Run 2: Switch to dataset B (DANGER!)
uv run python runners/train.py datasets.val_dataset.config.image_path=/path/to/dataset_B
# ❌ Cached tensors from dataset A still used for dataset B!
```

**Symptoms of Invalid Cache:**
- hmean drops to 0.0 or random values
- Cache hit rate still shows 95%+ (false positive)
- Training appears fast but produces wrong results

#### ✅ SAFE: Dataset Switching Protocol

**Option 1: Disable Cache When Switching**
```bash
# Safe switching - disable cache to force rebuild
uv run python runners/train.py \
  datasets.val_dataset.config.image_path=/path/to/new_dataset \
  datasets.val_dataset.config.cache_config.cache_transformed_tensors=false
```

**Option 2: Clear Cache Directory**
```bash
# If using persistent cache, clear it manually
rm -rf /path/to/cache/dir/*
uv run python runners/train.py datasets.val_dataset.config.image_path=/path/to/new_dataset
```

**Option 3: Use Different Cache Keys**
- Modify cache key generation to include dataset path
- Ensures different datasets use different cache entries

### Memory Management Guidelines

#### RAM Requirements by Feature

| Feature | Memory Cost | When Active |
|---------|-------------|-------------|
| **Base PyTorch** | ~2GB | Always |
| **+ Image Preloading** | +200MB | Dataset init |
| **+ Tensor Caching** | +800MB-1.2GB | After epoch 0 |
| **+ Maps Caching** | +50MB | With load_maps |
| **TOTAL (All enabled)** | ~3-4GB | Full optimization |

#### Memory Monitoring Commands

```bash
# Monitor memory usage during training
uv run python runners/train.py \
  datasets.val_dataset.config.cache_config.log_statistics_every_n=10

# Check system memory
free -h
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

#### Low Memory Workarounds

```yaml
# Memory-constrained config (slower but works)
datasets:
  val_dataset:
    config:
      preload_images: false              # Save 200MB
      cache_config:
        cache_transformed_tensors: false # Save 1GB
        cache_images: false              # Minimal saving
        cache_maps: true                 # Keep - low cost
```

### Cache Invalidation Triggers

#### Automatic Invalidation (Safe)
- Process restart (RAM cache cleared)
- Config changes detected by hash
- Dataset size changes

#### Manual Invalidation Required
- Transform parameter changes
- Dataset content changes (same path, different images)
- Model architecture changes affecting tensor shapes

### Validation Checks for New Users

#### Pre-Training Validation
```python
# Add to dataset __init__ or training script
def validate_cache_config(config):
    """Validate cache configuration for safety."""
    issues = []

    # Check for training dataset caching
    if 'train' in config.get('dataset_name', '').lower():
        if config.get('cache_config', {}).get('cache_transformed_tensors'):
            issues.append("🚨 Tensor caching enabled on training dataset - risk of data leakage")

    # Check memory requirements
    total_memory_gb = 2  # Base PyTorch
    if config.get('preload_images'):
        total_memory_gb += 0.2
    if config.get('cache_config', {}).get('cache_transformed_tensors'):
        total_memory_gb += 1.2

    if total_memory_gb > 8:  # Arbitrary threshold
        issues.append(f"⚠️ High memory usage predicted: {total_memory_gb}GB")

    return issues
```

#### Runtime Safety Checks
```python
# In training loop
def validate_cache_effectiveness():
    """Monitor cache is working correctly."""
    if cache_enabled and epoch > 0:
        hit_rate = cache_manager.get_hit_rate()
        if hit_rate < 0.8:  # Less than 80% hit rate
            logger.warning(f"Low cache hit rate: {hit_rate:.1%} - cache may be invalid")
            # Optionally: disable cache or rebuild
```

### Generating Preloaded Images for New Users

#### Automated Setup Script
```bash
#!/bin/bash
# setup_preloaded_images.sh

echo "Setting up preloaded images for new users..."

# 1. Verify canonical images exist
if [ ! -d "data/datasets/images_val_canonical" ]; then
    echo "❌ Canonical validation images not found"
    echo "Run preprocessing script first:"
    echo "uv run python scripts/preprocess_data.py"
    exit 1
fi

# 2. Test preloading works
echo "Testing image preloading..."
uv run python -c "
from ocr.datasets.base import ValidatedOCRDataset
from ocr.datasets.schemas import DatasetConfig, CacheConfig
import time

config = DatasetConfig(
    image_path='data/datasets/images_val_canonical',
    annotation_path='data/datasets/jsons/val.json',
    preload_images=True,
    cache_config=CacheConfig(cache_images=True)
)

start = time.time()
dataset = ValidatedOCRDataset(config=config, transform=lambda x: x)
preload_time = time.time() - start

print(f'✅ Preloading successful in {preload_time:.1f}s')
print(f'✅ Loaded {len(dataset)} images')
"

echo "✅ Preloaded images setup complete"
```

#### Manual Verification Steps
1. Check canonical images exist: `ls data/datasets/images_val_canonical | wc -l`
2. Verify EXIF normalization: Compare sample image metadata
3. Test loading speed: Time dataset initialization
4. Check memory usage: Monitor RAM during preloading
5. Validate cache: Check cache statistics in logs

### Troubleshooting Common Issues

#### "Preloaded 0/404 images" → "Preloaded 404/404 images"
- ✅ Check image paths are correct
- ✅ Verify images exist and are readable
- ✅ Check for EXIF orientation issues
- ✅ Look for corrupted image files

#### Cache hit rate stuck at 0%
- ✅ Verify `cache_transformed_tensors: true`
- ✅ Check epoch > 0 (cache builds on epoch 0)
- ✅ Ensure same transforms between runs
- ✅ Check for dataset changes

#### Cache hit rate stuck at 0%
- ✅ Verify `cache_transformed_tensors: true`
- ✅ Check epoch > 0 (cache builds on epoch 0)
- ✅ Ensure same transforms between runs
- ✅ Check for dataset changes

#### Training faster but worse hmean
- 🚨 **INVALID CACHE** - Clear cache and rebuild
- 🚨 Check dataset hasn't changed
- 🚨 Verify transforms are identical

---

## 📚 Feature Explanations & Setup Guide

### What Each Performance Feature Does

#### 1. Mixed Precision Training (`trainer.precision: "16-mixed"`)
**What it does:** Uses FP16 (half-precision) instead of FP32 (full-precision) for faster computation
**Speedup:** ~2x faster training
**Memory:** No change (same memory usage)
**Compatibility:** Safe with all other features
**When to disable:** Debugging numerical precision issues

#### 2. RAM Image Preloading (`preload_images: true`)
**What it does:** Loads ALL dataset images into RAM at startup
**Speedup:** ~1.12x (eliminates disk I/O)
**Memory:** ~200MB for validation dataset
**Compatibility:** Requires `cache_images: true`
**When to disable:** Low RAM systems (<8GB)

**How it works:**
```python
# At dataset initialization
for filename in dataset:
    image_data = load_and_process_image(filename)
    cache_manager.set_cached_image(filename, image_data)
```

#### 3. Tensor Caching (`cache_transformed_tensors: true`)
**What it does:** Caches fully processed tensors after transforms/augmentations
**Speedup:** 2.5-3x after cache warm-up (epochs 1+ run 6-8x faster)
**Memory:** ~800MB-1.2GB for validation dataset
**Compatibility:** Only for validation/test datasets (never training!)
**When to disable:** Memory constrained, debugging transforms

**How it works:**
```python
# First access (epoch 0) - slow
image -> transforms -> tensor -> cache

# Subsequent accesses (epoch 1+) - fast
cache -> tensor (skip transforms)
```

#### 4. Image Caching (`cache_images: true`)
**What it does:** Enables RAM storage of processed ImageData objects
**Speedup:** Required for preloading to work
**Memory:** Included in preload_images cost
**Compatibility:** Required for `preload_images: true`

#### 5. Maps Caching (`cache_maps: true`)
**What it does:** Caches probability/threshold maps for evaluation
**Speedup:** Faster evaluation metrics
**Memory:** ~50MB
**Compatibility:** Requires `load_maps: true`

### Setting Up Preloaded Images for New Users

#### Step 1: Verify Data Structure
```bash
# Check if canonical images exist
ls -la data/datasets/images_val_canonical/
# Should contain ~404 .jpg/.png files

# Check annotations
head -5 data/datasets/jsons/val.json
# Should contain image metadata
```

#### Step 2: Generate Canonical Images (if missing)
```bash
# Run preprocessing to create canonical validation images
uv run python scripts/preprocess_data.py

# This creates:
# - data/datasets/images_val_canonical/ (processed images)
# - data/datasets/jsons/val.json (annotations)
```

#### Step 3: Test Preloading
```bash
# Test that preloading works
uv run python -c "
from ocr.datasets.base import ValidatedOCRDataset
from ocr.datasets.schemas import DatasetConfig, CacheConfig

config = DatasetConfig(
    image_path='data/datasets/images_val_canonical',
    annotation_path='data/datasets/jsons/val.json',
    preload_images=True,
    cache_config=CacheConfig(cache_images=True)
)

print('Testing preloading...')
dataset = ValidatedOCRDataset(config=config, transform=lambda x: x)
print(f'✅ Success: {len(dataset)} images preloaded')
"
```

#### Step 4: Verify Configuration
```bash
# Check your config resolves correctly
uv run python runners/train.py --cfg job --resolve | grep -A 20 "val_dataset"
```

### Memory Management Guide

#### Understanding Memory Costs

| Component | Memory | When Active |
|-----------|--------|-------------|
| **PyTorch Base** | 2GB | Always |
| **Model Weights** | 1-2GB | Always |
| **Batch Data** | 0.5-1GB | During training |
| **Image Preload** | 0.2GB | `preload_images: true` |
| **Tensor Cache** | 1-1.5GB | `cache_transformed_tensors: true` |
| **Maps Cache** | 0.05GB | `cache_maps: true` |
| **TOTAL (All features)** | **4.75-6.75GB** | Full optimization |

#### Memory Optimization Strategies

**For 8GB RAM systems:**
```yaml
# Conservative config
datasets:
  val_dataset:
    config:
      preload_images: false      # Save 0.2GB
      cache_config:
        cache_transformed_tensors: false  # Save 1.5GB
        cache_images: true
        cache_maps: true
```

**For 16GB+ RAM systems:**
```yaml
# Full optimization config
datasets:
  val_dataset:
    config:
      preload_images: true       # Enable all features
      cache_config:
        cache_transformed_tensors: true
        cache_images: true
        cache_maps: true
```

#### Monitoring Memory Usage

```bash
# During training
watch -n 5 nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# System memory
watch -n 5 free -h
```

### Troubleshooting Common Issues

#### "Preloaded 0/404 images"
**Symptoms:** Preloading completes instantly with 0 images
**Causes:**
- Wrong image path
- Images don't exist
- EXIF orientation issues
- Unsupported image formats

**Fix:**
```bash
# Check image path
ls data/datasets/images_val_canonical/ | wc -l

# Check for corrupted images
find data/datasets/images_val_canonical/ -name "*.jpg" -exec file {} \; | grep -v "JPEG image"
```

#### "Cache hit rate: 0.0%"
**Symptoms:** Training slow, cache never hits
**Causes:**
- Cache not enabled
- Different transforms between runs
- Dataset changed
- Cache cleared between runs

**Fix:**
```bash
# Force rebuild cache
uv run python runners/train.py \
  datasets.val_dataset.config.cache_config.cache_transformed_tensors=false

# Then re-enable
uv run python runners/train.py \
  datasets.val_dataset.config.cache_config.cache_transformed_tensors=true
```

#### "CUDA out of memory"
**Symptoms:** Training crashes with OOM error
**Causes:**
- Too many features enabled
- Large batch size
- Insufficient GPU memory

**Fix:**
```yaml
# Reduce memory usage
trainer:
  precision: "32-true"  # Use FP32 instead of FP16

datasets:
  val_dataset:
    config:
      preload_images: false
      cache_config:
        cache_transformed_tensors: false
```

#### "Training fast but bad results"
**Symptoms:** Speedup achieved but hmean=0.0 or random
**Causes:**
- Invalid cache (dataset/transforms changed)
- Data leakage (cache used on training data)

**Fix:**
```bash
# Clear all caches and rebuild
rm -rf /tmp/cache_dir/*  # If using persistent cache

# Disable cache temporarily
uv run python runners/train.py \
  datasets.val_dataset.config.cache_config.cache_transformed_tensors=false \
  datasets.val_dataset.config.preload_images=false
```

### Performance Expectations by Hardware

#### RTX 4090 / A100 (High-end)
- **Baseline:** ~180s/epoch
- **Optimized:** ~25s/epoch (7x speedup)
- **Memory:** 24GB+ VRAM recommended

#### RTX 3080 / V100 (Mid-range)
- **Baseline:** ~200s/epoch
- **Optimized:** ~35s/epoch (6x speedup)
- **Memory:** 12GB+ VRAM recommended

#### RTX 3060 / P100 (Entry-level)
- **Baseline:** ~250s/epoch
- **Optimized:** ~60s/epoch (4x speedup)
- **Memory:** 8GB+ VRAM, disable tensor caching

### Advanced Configuration Examples

#### Maximum Performance (24GB+ VRAM)
```yaml
trainer:
  precision: "16-mixed"
  benchmark: true

datasets:
  val_dataset:
    config:
      preload_images: true
      load_maps: true
      cache_config:
        cache_transformed_tensors: true
        cache_images: true
        cache_maps: true
        log_statistics_every_n: 100
```

#### Balanced Performance (12GB VRAM)
```yaml
trainer:
  precision: "16-mixed"
  benchmark: true

datasets:
  val_dataset:
    config:
      preload_images: true      # Keep - low memory cost
      load_maps: true
      cache_config:
        cache_transformed_tensors: false  # Disable - high memory
        cache_images: true
        cache_maps: true
        log_statistics_every_n: 100
```

#### Memory Constrained (8GB VRAM)
```yaml
trainer:
  precision: "32-true"      # FP32 for compatibility
  benchmark: true

datasets:
  val_dataset:
    config:
      preload_images: false     # Disable - save memory
      load_maps: true
      cache_config:
        cache_transformed_tensors: false  # Disable - save memory
        cache_images: false     # Disable
        cache_maps: true        # Keep - low memory
        log_statistics_every_n: 100
```

---

## Alternative: Single Epoch Comparison

For quicker testing, run just 1 epoch on validation only:

```bash
# Baseline (should take ~180-200s)
uv run python runners/train.py \
  exp_name=quick_baseline \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=1 \
  trainer.limit_val_batches=50 \
  trainer.precision=32-true \
  datasets.val_dataset.config.preload_images=false \
  datasets.val_dataset.config.cache_config.cache_transformed_tensors=false \
  logger.wandb.enabled=false

# Optimized (should take ~60-70s first epoch)
uv run python runners/train.py \
  exp_name=quick_optimized \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=1 \
  trainer.limit_val_batches=50 \
  logger.wandb.enabled=false
```

**Note**: Single epoch test won't show tensor caching benefit (need epoch 1+ for that).
To see full benefit, you MUST run multiple epochs.
