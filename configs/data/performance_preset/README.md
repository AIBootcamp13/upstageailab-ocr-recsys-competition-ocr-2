# Performance Presets for Dataset Configuration

This directory contains preset configurations for different performance optimization levels. Use these to easily toggle between different combinations of caching, preloading, and other performance features.

## Available Presets

### 1. `none.yaml` - No Optimizations (Explicit Default)
**Use Case**: Explicitly disable all performance features
- Identical to `minimal` but provided for clarity
- Use when you want to explicitly state "no optimizations"
- Useful for documentation and debugging

### 2. `minimal.yaml` - Minimal Overhead (Default)
**Use Case**: Development, debugging, memory-constrained environments
- No caching
- No preloading
- Minimal memory footprint
- Baseline performance

### 3. `balanced.yaml` - Balanced Performance
**Use Case**: Standard training, moderate memory available
- Image caching enabled
- No tensor caching (safe for training)
- Moderate memory usage
- ~10-15% speedup

### 4. `validation_optimized.yaml` - Validation Speed
**Use Case**: Validation/test datasets only (NOT training!)
- Full caching (images + tensors + maps)
- Image preloading
- Map preloading
- Maximum speedup (~2.5-3x after warm-up)
- High memory usage

### 5. `memory_efficient.yaml` - Memory Constrained
**Use Case**: Low RAM systems, large datasets
- Minimal caching
- No preloading
- Optimized for memory
- Slight performance trade-off

## Usage

### Override in Command Line

```bash
# Use none preset (explicit no optimizations)
uv run python runners/train.py data/performance_preset=none

# Use minimal preset (default - same as 'none')
uv run python runners/train.py data/performance_preset=minimal

# Use balanced preset
uv run python runners/train.py data/performance_preset=balanced

# Use validation optimized (for validation dataset only!)
uv run python runners/train.py data/performance_preset=validation_optimized

# Use memory efficient
uv run python runners/train.py data/performance_preset=memory_efficient
```

### Override in Config File

```yaml
# configs/train.yaml
defaults:
  - data/performance_preset: balanced
```

### Per-Dataset Override

```yaml
# configs/data/base.yaml
datasets:
  train_dataset:
    config:
      defaults:
        - /data/performance_preset: minimal  # Safe for training

  val_dataset:
    config:
      defaults:
        - /data/performance_preset: validation_optimized  # Fast validation
```

## Preset Comparison Table

| Feature | None | Minimal | Balanced | Validation Optimized | Memory Efficient |
|---------|------|---------|----------|---------------------|------------------|
| **Memory Usage** | ~2.0 GB | ~2.0 GB | ~2.4 GB | ~3.5 GB | ~2.0 GB |
| **Speedup** | 1.0x | 1.0x | ~1.12x | ~2.5-3x (after warm-up) | 1.0x |
| **Epoch 0 Time** | Baseline | Baseline | +5% | +20% (cache build) | Baseline |
| **Epoch 1+ Time** | Baseline | Baseline | -10% | -60% to -70% | Baseline |
| **Cache Images** | ❌ | ❌ | ✅ | ✅ | ❌ |
| **Cache Tensors** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Cache Maps** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Preload Images** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Preload Maps** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Load Maps** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Safe for Training** | ✅ | ✅ | ✅ | ❌ | ✅ |

## Performance Guidelines

### When to Use Each Preset

#### Minimal (Default)
- ✅ Development and debugging
- ✅ Memory-constrained systems
- ✅ Training datasets (always safe)
- ✅ Investigating data issues
- ❌ Production training (too slow)

#### Balanced
- ✅ Standard training workflows
- ✅ Moderate memory available (4-6 GB)
- ✅ Good performance/memory trade-off
- ✅ Training datasets (safe)
- ❌ Need maximum speed

#### Validation Optimized
- ✅ Validation/test datasets ONLY
- ✅ Maximum speed required
- ✅ Sufficient memory available (6+ GB)
- ❌ **NEVER use for training datasets** (data leakage risk!)
- ❌ Memory-constrained systems

#### Memory Efficient
- ✅ Low RAM systems (< 4 GB available)
- ✅ Very large datasets
- ✅ Debugging memory issues
- ❌ Need fast training

## Safety Warnings

### ⚠️ CRITICAL: Tensor Caching on Training Data

**DO NOT enable `cache_transformed_tensors` on training datasets!**

This causes data leakage because:
1. Augmentations (random crops, flips) are cached
2. Same augmented samples repeated every epoch
3. Model overfits to cached augmentations
4. Validation metrics look good but model doesn't generalize

**Safe Usage**:
- ✅ Validation datasets (no augmentation)
- ✅ Test datasets (no augmentation)
- ❌ Training datasets (augmentation present)

### ⚠️ Memory Usage

- **Validation Optimized**: Uses ~1.5 GB extra RAM
- **Monitor Usage**: `uv run python scripts/cache_manager.py status`
- **Clear Cache**: `uv run python scripts/cache_manager.py clear --all`

## Customization

Create your own preset by copying an existing one:

```bash
cp configs/data/performance_preset/balanced.yaml \
   configs/data/performance_preset/custom.yaml
```

Then edit `custom.yaml` to your needs:

```yaml
# configs/data/performance_preset/custom.yaml
# @package _global_.datasets.val_dataset.config

# My custom preset
cache_config:
  cache_images: true
  cache_tensors: false  # Safe setting
  cache_maps: true
  log_statistics_every_n: 50  # More frequent logging

preload_images: false  # Save startup time
load_maps: true        # Enable evaluation
```

## Troubleshooting

### High Memory Usage

If you experience OOM (Out of Memory) errors:

1. Switch to `memory_efficient` preset:
   ```bash
   uv run python runners/train.py data/performance_preset=memory_efficient
   ```

2. Clear cache:
   ```bash
   uv run python scripts/cache_manager.py clear --all
   ```

3. Monitor memory:
   ```bash
   watch -n 1 nvidia-smi  # GPU memory
   htop  # CPU memory
   ```

### Slow First Epoch

This is expected with `validation_optimized` preset:
- Epoch 0: Builds cache (20% slower)
- Epoch 1+: Uses cache (2.5-3x faster)

**Trade-off is worth it for multi-epoch training!**

### Cache Not Working

Check cache health:
```bash
uv run python scripts/cache_manager.py health
```

Look for:
- Cache version mismatches
- Stale files (> 7 days old)
- Low hit rates (< 50%)

### Stale Cache

Clear and rebuild:
```bash
uv run python scripts/cache_manager.py clear --all
uv run python runners/train.py  # Rebuilds cache
```

## Related Documentation

- [Cache Management Guide](../../../docs/performance/CACHE_MANAGEMENT_GUIDE.md)
- [Performance Benchmark Commands](../../../docs/performance/BENCHMARK_COMMANDS.md)
- [Dataset Configuration Reference](../base.yaml)

## Examples

### Example 1: Development Workflow

```bash
# Use minimal for quick iteration
uv run python runners/train.py \
  data/performance_preset=minimal \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10
```

### Example 2: Production Training

```yaml
# configs/train_production.yaml
defaults:
  - override /data/performance_preset: balanced  # Training dataset
  - override /data/performance_preset@datasets.val_dataset: validation_optimized  # Validation dataset
```

### Example 3: Memory-Constrained System

```bash
# Use memory efficient preset
uv run python runners/train.py \
  data/performance_preset=memory_efficient \
  batch_size=8  # Reduce batch size too
```

## Version History

- **v1.0** (2025-10-14): Initial presets (minimal, balanced, validation_optimized, memory_efficient)
- Future: Add GPU-specific presets, dynamic preset selection

## Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section
- Review [Cache Management Guide](../../../docs/performance/CACHE_MANAGEMENT_GUIDE.md)
- Run cache health check: `uv run python scripts/cache_manager.py health`
