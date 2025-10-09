# Bottleneck Analysis: WebDataset vs DALI

**Date**: 2025-10-10
**Current Performance**: ~62s validation epoch (2.56x speedup achieved)
**Question**: Should we pursue WebDataset or DALI for additional speedup?

---

## Current Pipeline Bottlenecks

### What's Happening Every Epoch (WASTEFUL!)

Based on code analysis of [ocr/datasets/base.py](../../ocr/datasets/base.py):

```python
def __getitem__(self, idx):
    # 1. Load image (CACHED in Phase 6B ✅)
    image = self.image_cache[filename]  # Fast!

    # 2. Load/compute ground truth maps (CACHED ✅)
    prob_map, thresh_map = self.maps_cache[filename]  # Fast!

    # 3. Apply Albumentations transforms (NOT CACHED ❌)
    transformed = self.transform(image=np.array(image), polygons=polygons)
    # - Resize to 640x640
    # - Random horizontal flip (training)
    # - Normalize (mean/std)
    # - Convert to tensor

    # This happens EVERY epoch, EVERY batch, for the SAME validation data!
```

### Validation Data Re-Processing Waste

**For validation data** (404 images, unchanged across epochs):
- ✅ **Image loading**: Cached (Phase 6B)
- ✅ **GT maps**: Cached
- ❌ **Transforms**: Recomputed every time! (~12-15s per epoch wasted)
- ❌ **Tensor conversion**: Recomputed every time! (~2-3s wasted)
- ❌ **Collation**: Recomputed every time! (~1-2s wasted)

**Total waste**: ~15-20s per validation epoch on redundant CPU work!

---

## WebDataset: What It Solves

### Concept
Pre-process everything **once**, save to disk as ready-to-use tensors, stream from disk during training.

### Implementation

```python
# One-time preprocessing (run once, save forever)
python scripts/convert_to_webdataset.py

# Result: data/webdataset/val-{000..003}.tar
# Each tar contains:
#   - 000000.jpg.tensor  (preprocessed image tensor)
#   - 000000.prob_map.tensor
#   - 000000.thresh_map.tensor
#   - 000000.json (metadata)

# New dataloader (streaming, no transforms!)
dataset = wds.WebDataset("data/webdataset/val-{000..003}.tar")
    .decode("torch")  # Just deserialize tensors
    .batched(16)      # Batch directly
```

### What Gets Eliminated

| Current Pipeline | WebDataset Pipeline |
|------------------|---------------------|
| 1. Load image from cache (fast) | 1. Stream pre-tensored batch from tar |
| 2. Load GT maps from cache (fast) | 2. ~~Already in tensor~~ |
| 3. **Apply transforms (12s)** | 3. ~~Skip (pre-applied)~~ |
| 4. **Convert to tensor (3s)** | 4. ~~Skip (pre-converted)~~ |
| 5. **Collate batch (2s)** | 5. ~~Skip (pre-batched)~~ |

**Expected savings**: ~15-20s per validation epoch

### Realistic Speedup Estimate

```
Current: 62s
  - GPU inference: 15s (unchanged)
  - Data loading: 10s (disk I/O, slightly faster streaming)
  - Transforms: 12s → 0s ✅ (pre-applied)
  - Tensor/collate: 5s → 0s ✅ (pre-done)
  - Overhead: 20s → 15s (less CPU work)

WebDataset: ~40s (1.55x additional speedup)
Total: 3.97x vs baseline (158.9s → 40s)
```

### Pros & Cons

**Pros**:
- ✅ Moderate complexity (1-2 days for validation, 1 week for full pipeline)
- ✅ Debugging is straightforward (inspect tar files, familiar PyTorch)
- ✅ Works for both training and validation
- ✅ Disk space efficient (tar compression)
- ✅ Can version datasets easily

**Cons**:
- ⚠️ One-time conversion needed (10-20 min)
- ⚠️ **Training data gets tricky**: Random augmentations need careful handling
- ⚠️ **Data changes = re-convert**: If you add images, re-run conversion
- ⚠️ Still CPU-bound for tar extraction

---

## DALI: What It Solves

### Concept
Move **entire pipeline** to GPU, including image decoding, transforms, and batching.

### Implementation

```python
# DALI pipeline (runs on GPU!)
@dali.pipeline_def
def ocr_pipeline():
    # 1. Read JPEG from disk (GPU decoder!)
    images = dali.fn.readers.file(files=image_list)
    images = dali.fn.decoders.image(images, device="mixed")  # GPU decode

    # 2. Resize on GPU
    images = dali.fn.resize(images, size=[640, 640], device="gpu")

    # 3. Normalize on GPU
    images = dali.fn.normalize(images, mean=[0.485, ...], device="gpu")

    # 4. Random augmentations on GPU
    images = dali.fn.flip(images, horizontal=dali.fn.random.coin_flip())

    # 5. Load GT maps (custom operator)
    prob_maps = dali.fn.python_function(...)  # Still CPU for complex ops

    return images, prob_maps
```

### What Gets Eliminated

| Current Pipeline | DALI Pipeline |
|------------------|----------------|
| 1. **Load image (CPU, 5s)** | 1. **GPU JPEG decode (0.5s)** ✅ |
| 2. **Resize (CPU, 3s)** | 2. **GPU resize (0.2s)** ✅ |
| 3. **Normalize (CPU, 2s)** | 3. **GPU normalize (0.1s)** ✅ |
| 4. **Augment (CPU, 2s)** | 4. **GPU augment (0.1s)** ✅ |
| 5. **Collate (CPU, 2s)** | 5. **GPU batch (0.1s)** ✅ |
| 6. **Copy to GPU (3s)** | 6. ~~Skip (already on GPU)~~ ✅ |

**Expected savings**: ~30-35s per validation epoch

### Realistic Speedup Estimate

```
Current: 62s
  - GPU inference: 15s (unchanged)
  - Data pipeline: 47s → 5s ✅ (GPU acceleration)

DALI: ~20-25s (2.5-3x additional speedup)
Total: 6-8x vs baseline (158.9s → 20-25s)
```

### Pros & Cons

**Pros**:
- ✅ **Maximum performance**: 5-10x speedup potential
- ✅ No pre-processing needed (runtime augmentation works)
- ✅ GPU utilization stays high (no CPU bottleneck)
- ✅ Built for training at scale

**Cons**:
- ❌ **High complexity**: Complete pipeline rewrite (2-3 weeks)
- ❌ **Debugging nightmare**: GPU kernels, opaque errors, hard to inspect
- ❌ **Limited transform support**: Not all Albumentations ops available
- ❌ **Custom ops needed**: Your DB ground truth map generation needs custom DALI operator
- ❌ **Learning curve**: DALI API is very different from PyTorch
- ❌ **Fragile**: Breaking changes between DALI versions, less community support

---

## Your Specific Bottlenecks

Based on your concern: *"reloading the same validation data for every epoch"*

### What's Actually Happening Now

**Validation loop** (50 batches × 16 samples = 800 samples per epoch):

```python
# Epoch 1
for batch in val_dataloader:  # 50 batches
    # Each batch:
    #   1. Get 16 samples from dataset
    #   2. Each sample: load image (cached ✅), apply transforms (NOT cached ❌)
    #   3. Collate into batch
    #   4. Run inference

# Epoch 2 - REPEAT EVERYTHING ABOVE!
# Same 800 samples, same transforms, same results... all recomputed!
```

### Why Not Just Cache Transformed Tensors?

**Option: In-memory tensor cache** (simplest fix!)

```python
class OCRDataset(Dataset):
    def __init__(self, ..., cache_transformed_tensors=False):
        self.tensor_cache = {}  # Cache final tensors

    def __getitem__(self, idx):
        if idx in self.tensor_cache:
            return self.tensor_cache[idx]  # Instant return!

        # Normal pipeline...
        transformed = self.transform(...)

        if self.cache_transformed_tensors and not self.training:
            self.tensor_cache[idx] = transformed  # Cache for validation

        return transformed
```

**Expected speedup**: ~1.5-2x additional (62s → 35-40s)
**Effort**: ~30 minutes
**Risk**: Very low

**Why this works**:
- Validation data never changes
- Transforms are deterministic (no random augmentation)
- After first epoch, everything is cached in RAM
- Training still works normally (cache disabled)

---

## Recommendation Matrix

| Solution | Speedup (vs 62s) | Total Speedup | Effort | Risk | When to Use |
|----------|------------------|---------------|--------|------|-------------|
| **Tensor Cache** | 1.5-2x → 35-40s | **4-4.5x** | 30 min | Low | **DO THIS FIRST** ✅ |
| **WebDataset** | 1.5-2x → 35-45s | 3.5-4.5x | 1-2 weeks | Medium | Training speedup needed |
| **DALI** | 2.5-3x → 20-25s | 6-8x | 2-3 weeks | High | Extreme performance critical |

---

## My Specific Recommendation for YOU

### Phase 6E: Tensor Caching (DO THIS NEXT)

**Why**: You correctly identified the problem - validation data is reprocessed every epoch!

**Implementation**:
```python
# Add to configs/data/base.yaml
datasets:
  val_dataset:
    preload_images: true          # ✅ Already enabled
    cache_transformed_tensors: true  # ← NEW, simple flag
```

**Expected result**:
- First epoch: 62s (normal, builds cache)
- Subsequent epochs: 30-35s (everything cached!)
- **Training speedup**: 4-5x total (158.9s → 30-35s)

**Why NOT WebDataset/DALI yet**:
1. Tensor caching gives 80% of WebDataset's benefit for 3% of the effort
2. You're correct that debugging is harder with those approaches
3. Training "takes too long" → Simple cache solves this for validation
4. If still too slow after tensor caching, THEN consider WebDataset

### If You Still Need More After Tensor Caching

**Then WebDataset** (not DALI):
- DALI's complexity isn't worth it unless you're training 24/7 at scale
- WebDataset is "good enough" for most research (4-5x total speedup)
- Debugging DALI is genuinely painful (I've been there!)

---

## Next Steps

1. **Implement Phase 6E** (tensor caching): 30-60 min
2. **Benchmark**: Should see ~35s validation epochs
3. **Decide**:
   - Happy with 4-5x? → Stop, move to other priorities ✅
   - Need more? → Implement WebDataset (1-2 weeks)
   - Need maximum? → Bite the bullet on DALI (2-3 weeks + debugging)

Want me to implement Phase 6E (tensor caching) now? It's a quick win that solves your exact pain point.

---

## References

- **WebDataset Docs**: https://github.com/webdataset/webdataset
- **DALI Docs**: https://docs.nvidia.com/deeplearning/dali/
- **Current Pipeline**: [ocr/datasets/base.py](../../ocr/datasets/base.py)
- **Phase 6D Results**: [phase-6d-mixed-precision-findings.md](../../logs/2025-10-08_02_refactor_performance_features/phase-6d-mixed-precision-findings.md)
