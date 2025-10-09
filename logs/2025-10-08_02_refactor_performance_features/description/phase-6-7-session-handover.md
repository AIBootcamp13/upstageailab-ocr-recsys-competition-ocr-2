# Phase 6-7 Session Handover: Aggressive Performance Optimization

**Date**: 2025-10-09
**Session**: Advanced Performance Optimization (WebDataset + Parallel Preprocessing)
**Context**: Phases 1-4 Complete, Ready for Tier 2-3 Optimizations

---

## Executive Summary

Phases 1-4 achieved code simplification but minimal speedup (~1.0x). Investigation revealed map generation was only 5% of pipeline time. To achieve 2-5x speedup, we need to optimize the REAL bottlenecks:

- Image loading & decoding (30%)
- Transform pipeline (25%)
- Model inference (35%)

Phase 6-7 implements aggressive optimizations with parallel development using Qwen for test generation and isolated tasks.

---

## Current State

### ✅ Completed (Phases 1-4)

**Phase 1**: Offline preprocessing script generates `.npz` maps
**Phase 2**: Pipeline refactored to load pre-processed maps
**Phase 3**: Documentation and configuration cleanup
**Phase 4**: Performance investigation identified real bottlenecks

### 📊 Benchmark Results

| Metric | Value |
|--------|-------|
| Validation epoch (50 batches) | 31.3s |
| Maps loaded successfully | 100% |
| RAM preloading speedup | 0.91x (slower) |
| **Conclusion** | Map loading is NOT the bottleneck |

### 🎯 Real Bottlenecks (from profiling)

1. **Image Loading** (30%) - JPEG decode, PIL operations
2. **Transforms** (25%) - Albumentations pipeline
3. **Model Inference** (35%) - Forward pass
4. **Map Loading** (5%) - Already optimized ✅

---

## Phase 6: WebDataset or RAM Caching (Tier 2)

**Goal**: Optimize data loading to eliminate I/O bottleneck and enable better parallelization

**Expected Gain**: 2-5x faster data loading

### Option A: WebDataset (Recommended for Scalability)

**When to use**: Large datasets, cloud training, need for streaming

#### Benefits
- Sequential I/O (faster than random access)
- Built-in sharding for multi-GPU
- Reduces filesystem overhead (1 tar file vs 1000s of images)
- Better caching behavior

#### Implementation Plan

##### 6.1A: Install WebDataset
```bash
uv add webdataset
```

##### 6.2A: Create Conversion Script

**Task for Claude**: Create `scripts/convert_to_webdataset.py`

```python
# Pseudo-code structure
import webdataset as wds

def convert_dataset(image_dir, maps_dir, output_tar):
    """
    Convert images + maps to WebDataset .tar format

    Each sample becomes:
    - <basename>.jpg (image)
    - <basename>.npz (maps)
    - <basename>.json (metadata: polygons, orientation, etc.)
    """
    with wds.TarWriter(output_tar) as sink:
        for image_file in image_dir.glob("*.jpg"):
            # Load image
            # Load corresponding .npz map
            # Load metadata from annotations
            # Write to tar: {key: basename, jpg: bytes, npz: bytes, json: metadata}
            pass

# Convert train and val datasets
convert_dataset("images/train", "images/train_maps", "train.tar")
convert_dataset("images_val_canonical", "images_val_canonical_maps", "val.tar")
```

**Task for Qwen**: Generate unit tests for `convert_to_webdataset.py`
```bash
# After Claude creates the script, delegate test generation to Qwen
cat scripts/convert_to_webdataset.py | qwen --yolo --prompt "Generate comprehensive pytest unit tests for this WebDataset conversion script. Include tests for: edge cases, missing files, corrupted data, tar file integrity, sample format validation."
```

##### 6.3A: Update OCRDataset to Use WebDataset

**Task for Claude**: Modify `ocr/datasets/base.py`

```python
# Pseudo-code
import webdataset as wds

class OCRWebDataset(Dataset):
    def __init__(self, tar_path, transform):
        self.dataset = (
            wds.WebDataset(tar_path)
            .decode("pil")  # Auto-decode JPEG
            .rename(image="jpg", maps="npz", metadata="json")
            .map(self._process_sample)
        )
        self.transform = transform

    def _process_sample(self, sample):
        # Apply transforms
        # Extract prob_map/thresh_map from npz
        # Return formatted item
        pass
```

**Task for Qwen**: Generate tests for WebDataset integration
```bash
cat ocr/datasets/webdataset.py | qwen --yolo --prompt "Generate pytest tests for this WebDataset wrapper class. Test: sample loading, transform application, error handling, edge cases with corrupted tar files."
```

##### 6.4A: Update Configuration

**Task for Claude**: Create `configs/data/webdataset.yaml`

```yaml
datasets:
  train_dataset:
    _target_: ocr.datasets.OCRWebDataset
    tar_path: ${dataset_base_path}train.tar
    transform: ${transforms.train_transform}
  val_dataset:
    _target_: ocr.datasets.OCRWebDataset
    tar_path: ${dataset_base_path}val.tar
    transform: ${transforms.val_transform}
```

##### 6.5A: Benchmark

**Task for Claude**: Run benchmark to measure improvement

```bash
# Baseline (current)
time uv run python runners/train.py trainer.limit_val_batches=100 trainer.max_epochs=1

# WebDataset
time uv run python runners/train.py data=webdataset trainer.limit_val_batches=100 trainer.max_epochs=1
```

### Option B: Full RAM Caching (Recommended for Small Datasets)

**When to use**: Dataset fits in RAM (<32GB), single machine training

#### Benefits
- Zero I/O after initial load
- Simplest implementation
- Maximum speed for small datasets

#### Implementation Plan

##### 6.1B: Add RAM Caching Config

**Task for Claude**: Update `configs/data/base.yaml`

```yaml
data:
  preload_everything: true  # Load images + maps to RAM

datasets:
  val_dataset:
    preload_images: true  # Load PIL images to RAM
    preload_maps: true    # Load maps to RAM (already implemented)
```

##### 6.2B: Extend OCRDataset with Image Caching

**Task for Claude**: Modify `ocr/datasets/base.py`

```python
def __init__(self, ..., preload_images=False):
    # ... existing code ...
    self.image_cache = {}

    if preload_images:
        self._preload_images_to_ram()

def _preload_images_to_ram(self):
    """Preload decoded PIL images to RAM."""
    from tqdm import tqdm
    for filename in tqdm(self.anns.keys(), desc="Loading images to RAM"):
        image_path = self.image_path / filename
        pil_image = Image.open(image_path)
        # Store decoded RGB image as numpy array
        self.image_cache[filename] = np.array(pil_image.convert("RGB"))
```

**Task for Qwen**: Generate tests for image caching
```bash
cat ocr/datasets/base.py | qwen --yolo --prompt "Generate pytest tests for the image preloading functionality. Test: memory usage, cache hit rate, error handling, concurrent access."
```

##### 6.3B: Benchmark

**Task for Claude**: Measure improvement

```bash
# Without image caching
time uv run python runners/train.py datasets.val_dataset.preload_images=false trainer.limit_val_batches=100

# With image caching
time uv run python runners/train.py datasets.val_dataset.preload_images=true trainer.limit_val_batches=100
```

---

## Phase 7: NVIDIA DALI (Tier 3 - Maximum Performance)

**Goal**: Offload data loading and augmentation to GPU for maximum throughput

**Expected Gain**: Eliminate CPU bottleneck, 5-10x faster data loading

**Warning**: Most complex implementation, requires significant refactoring

### Implementation Plan

#### 7.1: Install DALI

**Task for Claude**: Add DALI to dependencies

```bash
uv add nvidia-dali-cuda120  # Or appropriate CUDA version
```

#### 7.2: Create DALI Pipeline

**Task for Claude**: Create `ocr/datasets/dali_pipeline.py`

```python
# Pseudo-code
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types

@dali.pipeline_def
def ocr_pipeline(image_dir, maps_dir, batch_size, num_threads):
    """
    DALI pipeline for OCR data loading
    - Read images from disk (GPU accelerated)
    - Decode JPEG on GPU
    - Load .npz maps
    - Apply augmentations on GPU
    - Return batches ready for training
    """
    # Read images
    images, labels = fn.readers.file(file_root=image_dir, random_shuffle=False)

    # Decode on GPU
    images = fn.decoders.image(images, device="mixed")  # CPU->GPU decode

    # Resize (GPU)
    images = fn.resize(images, size=[640, 640], device="gpu")

    # Normalize (GPU)
    images = fn.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device="gpu")

    # Load maps (CPU, then transfer to GPU)
    # ... map loading logic ...

    return images, prob_maps, thresh_maps
```

**Task for Qwen**: Generate DALI pipeline tests
```bash
cat ocr/datasets/dali_pipeline.py | qwen --yolo --prompt "Generate comprehensive tests for this DALI pipeline. Test: pipeline execution, GPU memory usage, batch output shapes, augmentation correctness, error handling."
```

#### 7.3: Create DALI DataLoader

**Task for Claude**: Create `ocr/datasets/dali_loader.py`

```python
from nvidia.dali.plugin.pytorch import DALIGenericIterator

class DALIDataLoader:
    def __init__(self, pipeline, size, batch_size):
        self.dali_iterator = DALIGenericIterator(
            pipeline,
            output_map=["images", "prob_maps", "thresh_maps"],
            size=size,
            auto_reset=True
        )

    def __iter__(self):
        for data in self.dali_iterator:
            # Format DALI output to match PyTorch DataLoader format
            yield self._format_batch(data)
```

**Task for Qwen**: Generate DataLoader tests

#### 7.4: Update Lightning Module

**Task for Claude**: Modify `ocr/lightning_modules/ocr_pl.py`

```python
def train_dataloader(self):
    if self.config.use_dali:
        from ocr.datasets.dali_loader import DALIDataLoader
        return DALIDataLoader(...)
    else:
        # Existing PyTorch DataLoader
        return DataLoader(...)
```

#### 7.5: Benchmark

**Task for Claude**: Comprehensive benchmark

```bash
# Baseline (PyTorch DataLoader)
time uv run python runners/train.py trainer.max_epochs=1

# DALI
time uv run python runners/train.py use_dali=true trainer.max_epochs=1
```

---

## Parallel Development Strategy with Qwen

### Workflow

1. **Claude** creates core implementation (scripts, classes, logic)
2. **Qwen** generates comprehensive tests in parallel
3. **Claude** reviews and integrates tests
4. **Both** work simultaneously to maximize throughput

### Task Distribution

| Task Type | Owner | Rationale |
|-----------|-------|-----------|
| Architecture decisions | Claude | Requires advanced reasoning |
| Script creation | Claude | Complex logic |
| Test generation | Qwen | Excellent at comprehensive test coverage |
| Benchmarking | Claude | Requires interpretation |
| Documentation | Claude | Requires context understanding |
| Refactoring | Both | Qwen for mechanical, Claude for complex |
| Bug fixing | Claude | Requires debugging skills |

### Qwen Delegation Examples

#### Example 1: Test Generation
```bash
# After Claude creates convert_to_webdataset.py
cat scripts/convert_to_webdataset.py | qwen --yolo --prompt "Generate comprehensive pytest unit tests for this WebDataset conversion script. Include: edge cases, missing files, corrupted data, tar integrity checks, memory usage tests, concurrent conversion tests."
```

#### Example 2: Refactoring Helper
```bash
# Qwen can help with mechanical refactoring
cat ocr/datasets/base.py | qwen --yolo --prompt "Extract the map loading logic into a separate MapLoader class. Maintain all functionality and add type hints."
```

#### Example 3: Documentation
```bash
# Qwen generates API docs
cat ocr/datasets/webdataset.py | qwen --yolo --prompt "Generate comprehensive API documentation in Google docstring format. Include usage examples, parameters, return values, and common pitfalls."
```

#### Example 4: Benchmark Script
```bash
# Qwen creates benchmark harness
qwen --yolo --prompt "Create a Python script that benchmarks data loading performance. Compare: baseline DataLoader, WebDataset, RAM caching. Output: CSV with timing results, speedup factors, memory usage. Use cProfile for detailed profiling."
```

---

## Decision Matrix: Which Approach?

| Criterion | WebDataset | RAM Cache | DALI |
|-----------|------------|-----------|------|
| **Implementation Effort** | Medium | Low | High |
| **Speedup Potential** | 2-3x | 2-4x | 5-10x |
| **Memory Usage** | Low | High | Medium |
| **Scalability** | Excellent | Poor | Excellent |
| **Multi-GPU** | Built-in | Manual | Built-in |
| **Maintenance** | Low | Low | Medium |
| **Best For** | Large datasets | Small datasets | Max performance |

### Recommendations

**Start Here**:
1. **Phase 6B** (RAM Cache) - Quick win for validation dataset (<1GB)
2. Measure improvement
3. If still too slow, proceed to Phase 6A (WebDataset) or Phase 7 (DALI)

**Skip**: Don't jump straight to DALI unless you need absolute maximum performance

---

## Implementation Timeline

### Week 1: Phase 6B (RAM Caching)
- **Day 1**: Implement image preloading (Claude)
- **Day 1**: Generate tests (Qwen, parallel)
- **Day 2**: Benchmark and optimize
- **Expected**: 2x speedup on validation

### Week 2: Phase 6A (WebDataset) - If needed
- **Day 1**: Conversion script (Claude + Qwen tests)
- **Day 2**: Dataset wrapper (Claude + Qwen tests)
- **Day 3**: Integration and benchmark
- **Expected**: 3x speedup

### Week 3: Phase 7 (DALI) - If needed
- **Day 1-2**: Pipeline implementation (Claude)
- **Day 2**: Test generation (Qwen, parallel)
- **Day 3-4**: DataLoader integration
- **Day 5**: Comprehensive benchmark
- **Expected**: 5-10x speedup

---

## Success Criteria

### Phase 6
- [ ] Validation epoch time < 15s (50% reduction from 31s)
- [ ] Zero accuracy regression (H-mean maintained)
- [ ] Tests passing (>90% coverage)
- [ ] Documentation updated

### Phase 7
- [ ] Validation epoch time < 10s (70% reduction)
- [ ] GPU utilization > 90% during training
- [ ] CPU bottleneck eliminated
- [ ] Multi-GPU scaling confirmed

---

## Rollback Plan

Each phase is independent:
- **Phase 6B fails**: Revert to disk loading (current state)
- **Phase 6A fails**: Revert to standard Dataset
- **Phase 7 fails**: Keep Phase 6 improvements

Git branches:
- `feature/ram-caching` (Phase 6B)
- `feature/webdataset` (Phase 6A)
- `feature/dali` (Phase 7)

---

## Quick Start Commands

### Setup
```bash
# Create feature branch
git checkout -b feature/aggressive-optimization

# Install dependencies (as needed)
uv add webdataset  # For Phase 6A
uv add nvidia-dali-cuda120  # For Phase 7
```

### Phase 6B: RAM Caching
```bash
# 1. Implement image preloading (Claude)
# Edit ocr/datasets/base.py - add _preload_images_to_ram()

# 2. Generate tests (Qwen, parallel)
cat ocr/datasets/base.py | qwen --yolo --prompt "Generate pytest tests for image preloading"

# 3. Benchmark
time uv run python runners/train.py datasets.val_dataset.preload_images=true trainer.limit_val_batches=100
```

### Phase 6A: WebDataset
```bash
# 1. Create conversion script (Claude)
# Create scripts/convert_to_webdataset.py

# 2. Generate tests (Qwen)
cat scripts/convert_to_webdataset.py | qwen --yolo --prompt "Generate comprehensive tests"

# 3. Convert datasets
uv run python scripts/convert_to_webdataset.py

# 4. Benchmark
time uv run python runners/train.py data=webdataset trainer.limit_val_batches=100
```

---

## Key Files Reference

### To Modify

| File | Purpose | Phase |
|------|---------|-------|
| `ocr/datasets/base.py` | Add image preloading | 6B |
| `scripts/convert_to_webdataset.py` | Convert to tar | 6A |
| `ocr/datasets/webdataset.py` | WebDataset wrapper | 6A |
| `ocr/datasets/dali_pipeline.py` | DALI pipeline | 7 |
| `ocr/lightning_modules/ocr_pl.py` | DataLoader selection | All |
| `configs/data/base.yaml` | Configuration | All |

### Documentation

| File | Contents |
|------|----------|
| `docs/preprocessing_guide.md` | Update with Phase 6-7 |
| `docs/CHANGELOG.md` | Document changes |
| `logs/.../phase-6-7-findings.md` | Performance results |

---

## Context for AI Agents

### For Claude (Advanced Reasoning)

When implementing Phase 6-7:

1. **Start Simple**: Begin with Phase 6B (RAM caching)
2. **Measure Everything**: Benchmark after each change
3. **Delegate to Qwen**: Let Qwen handle test generation
4. **Document Findings**: Create phase-X-findings.md after each phase
5. **Maintain Quality**: Ensure H-mean doesn't regress

### For Qwen (Test Generation & Mechanical Tasks)

You excel at:
- Comprehensive test generation
- API documentation
- Mechanical refactoring
- Benchmark script creation

Typical tasks:
```bash
# Test generation (your specialty)
cat <script.py> | qwen --yolo --prompt "Generate comprehensive pytest tests"

# Documentation
cat <module.py> | qwen --yolo --prompt "Generate API documentation in Google format"

# Refactoring
cat <file.py> | qwen --yolo --prompt "Extract <functionality> into separate class"
```

---

## Session Continuation Prompt

```
I'm implementing Phase 6-7 of the OCR data pipeline optimization project. We've completed Phases 1-4 (preprocessing + investigation) and identified that image loading, transforms, and model inference are the real bottlenecks.

Read the session handover:
@logs/2025-10-08_02_refactor_performance_features/description/phase-6-7-session-handover.md

Current performance:
- Validation epoch: 31.3s (50 batches)
- Map loading already optimized (5% of time)
- Need to optimize: image loading (30%), transforms (25%), inference (35%)

Goal: Achieve 2-5x speedup through WebDataset or RAM caching

Start with Phase 6B (RAM caching for images) as it's the quickest win.

I'll use Qwen in parallel for test generation. After I create each component, delegate test generation to Qwen using the examples in the handover doc.

Let's begin with implementing image preloading in ocr/datasets/base.py.
```

---

**Status**: Ready for Phase 6-7 Implementation
**Recommended Start**: Phase 6B (RAM Caching)
**Parallel Worker**: Qwen for test generation
**Expected Timeline**: 1-3 weeks depending on phases implemented
**Token Budget**: 98k remaining (sufficient for Phase 6, may need new session for Phase 7)
