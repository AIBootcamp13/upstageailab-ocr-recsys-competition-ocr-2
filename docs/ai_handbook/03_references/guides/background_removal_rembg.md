# Background Removal with rembg

This document explains how to use [rembg](https://github.com/danielgatis/rembg) for background removal in your OCR receipt text detection pipeline.

## Overview

rembg is an AI-powered tool that automatically removes backgrounds from images using deep learning models. For OCR tasks, this can help by:

- Removing distracting background elements
- Focusing attention on text regions
- Improving text detection accuracy
- Reducing noise in the preprocessing pipeline

## Installation

rembg is already installed in this project via UV:

```bash
uv add rembg onnxruntime
```

The installation includes:
- `rembg`: Core background removal library
- `onnxruntime`: Required for running the AI models

## Usage

### 1. As an Albumentations Transform

The background removal is implemented as an Albumentations-compatible transform that can be integrated into your existing preprocessing pipelines.

#### Basic Usage

```python
from ocr.datasets.preprocessing.background_removal import create_background_removal_transform

# Create a background removal transform
bg_remover = create_background_removal_transform(
    model="u2net",          # Model to use
    alpha_matting=True,     # Enable for better edges
    p=1.0                   # Apply to all images
)

# Apply to an image
result = bg_remover.apply(image)
```

#### Advanced Configuration

```python
from ocr.datasets.preprocessing.background_removal import BackgroundRemoval

# Full control over parameters
bg_remover = BackgroundRemoval(
    model="u2net",
    alpha_matting=True,
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=10,
    alpha_matting_erode_size=10,
    only_mask=False,
    post_process_mask=False,
    p=1.0
)
```

### 2. Integration with Hydra Configs

Add background removal to your transform pipelines by modifying the config files:

#### Option A: Use the provided config

```yaml
# In your train.yaml or transform config
transforms:
  train_transform:
    _target_: ${dataset_path}.DBTransforms
    transforms:
      # Add background removal early in pipeline
      - _target_: ocr.datasets.preprocessing.external.BackgroundRemoval
        model: "u2net"
        alpha_matting: true
        p: 1.0
      # ... rest of your transforms
```

#### Option B: Use the convenience config

```yaml
# configs/transforms/with_background_removal.yaml
defaults:
  - base

# This config extends base transforms with background removal
```

Then use it in your training:

```bash
uv run python runners/train.py transforms=with_background_removal
```

### 3. Available Models

rembg supports several models with different trade-offs:

- **`u2net`** (default): Best quality, slower processing
- **`u2netp`**: Faster, slightly lower quality
- **`u2net_cloth_seg`**: Specialized for clothing/textiles
- **`silueta`**: Lightweight model
- **`isnet`**: Alternative architecture

### 4. Performance Considerations

- **First run**: Models are downloaded automatically (~176MB for u2net)
- **GPU acceleration**: ONNX Runtime automatically uses GPU if available
- **Batch processing**: Consider processing images in batches for better performance
- **Caching**: Models are cached locally after first download

## Pipeline Integration Points

### Where to Apply Background Removal

1. **Early preprocessing** (recommended): Apply before geometric transformations
2. **After document detection**: Apply only to detected document regions
3. **Conditional application**: Apply only to images with complex backgrounds

### Example Pipeline Order

```
Input Image → Background Removal → Resize → Document Detection → Perspective Correction → Enhancement → Normalization → Model
```

### Conditional Usage

For production systems, you might want to apply background removal conditionally:

```python
# Only apply to images with detected background complexity
if has_complex_background(image):
    image = bg_remover.apply(image)
```

## Testing and Validation

### Demo Script

Run the included demo script to test background removal:

```bash
uv run python demo_background_removal.py
```

This will:
- Process any images in `demo_images/` directory
- Create test images if none exist
- Test different models
- Save results as PNG files

### Integration Testing

Test with your OCR pipeline:

```python
# Test background removal with your transforms
from ocr.datasets.transforms import ValidatedDBTransforms

transforms = ValidatedDBTransforms(config_with_background_removal)
result = transforms(sample)
```

## Configuration Files

- `configs/transforms/background_removal.yaml`: Pre-configured background removal options
- `configs/transforms/with_background_removal.yaml`: Complete transform pipeline including background removal
- `ocr/datasets/preprocessing/background_removal.py`: Implementation
- `ocr/datasets/preprocessing/external.py`: Optional dependency handling

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Ensure `rembg` and `onnxruntime` are installed
2. **Model download fails**: Check internet connection for first run
3. **Memory issues**: Use lighter models (`u2netp`) for large images
4. **Quality issues**: Enable `alpha_matting=True` for better edges

### Performance Tuning

```python
# For faster processing
bg_remover = BackgroundRemoval(
    model="u2netp",        # Lighter model
    alpha_matting=False,   # Disable for speed
)

# For highest quality
bg_remover = BackgroundRemoval(
    model="u2net",
    alpha_matting=True,
    alpha_matting_erode_size=5,  # Fine-tune edges
)
```

## Comparison with Existing Methods

| Method | Pros | Cons |
|--------|------|------|
| rembg | AI-powered, high quality | Slower, requires GPU/CPU resources |
| Shadow removal | Fast, specialized for receipts | Limited to shadow removal only |
| Traditional CV | Very fast | Lower quality, manual tuning required |

Consider using rembg for:
- Complex backgrounds with multiple objects
- High-quality preprocessing requirements
- Research and experimentation

Use traditional methods for:
- Speed-critical production systems
- Simple backgrounds
- Resource-constrained environments
