# Enhanced Preprocessing Pipeline Usage Guide

**Version**: 1.0
**Date**: 2025-10-15
**Status**: Production Ready

## Overview

The Enhanced Preprocessing Pipeline integrates Office Lens quality document preprocessing features into a modular, configurable system. It combines Phase 1 (advanced document detection) and Phase 2 (advanced enhancement) features with quality-based processing decisions and performance monitoring.

## Quick Start

### Basic Usage

```python
from ocr.datasets.preprocessing.enhanced_pipeline import create_office_lens_preprocessor

# Create Office Lens quality preprocessor
preprocessor = create_office_lens_preprocessor()

# Process image
result = preprocessor(image)

# Access results
processed_image = result["image"]
metadata = result["metadata"]
metrics = result["metrics"]
quality_scores = result["quality_assessment"]
```

### Fast Processing (Basic Features Only)

```python
from ocr.datasets.preprocessing.enhanced_pipeline import create_fast_preprocessor

# Create fast preprocessor (no advanced features)
preprocessor = create_fast_preprocessor(target_size=(640, 640))

result = preprocessor(image)
```

## Configuration

### Full Configuration Example

```python
from ocr.datasets.preprocessing.enhanced_pipeline import (
    EnhancedDocumentPreprocessor,
    EnhancedPipelineConfig,
    EnhancementStage,
    QualityThresholds
)
from ocr.datasets.preprocessing.config import DocumentPreprocessorConfig
from ocr.datasets.preprocessing.advanced_noise_elimination import (
    NoiseEliminationConfig,
    NoiseReductionMethod
)
from ocr.datasets.preprocessing.document_flattening import (
    FlatteningConfig,
    FlatteningMethod
)
from ocr.datasets.preprocessing.intelligent_brightness import (
    BrightnessConfig,
    BrightnessMethod
)

# Configure base preprocessing
base_config = DocumentPreprocessorConfig(
    enable_document_detection=True,
    enable_perspective_correction=True,
    enable_enhancement=True,
    enhancement_method="office_lens",
    target_size=(640, 640),
)

# Configure advanced features
noise_config = NoiseEliminationConfig(
    method=NoiseReductionMethod.COMBINED,
    preserve_text_regions=True,
    shadow_removal_strength=0.8,
)

flattening_config = FlatteningConfig(
    method=FlatteningMethod.ADAPTIVE,
    grid_size=20,
    edge_preservation_strength=0.8,
)

brightness_config = BrightnessConfig(
    method=BrightnessMethod.AUTO,
    clahe_clip_limit=2.0,
    auto_gamma=True,
)

# Configure quality thresholds
quality_thresholds = QualityThresholds(
    min_noise_elimination_effectiveness=0.5,
    min_flattening_quality=0.4,
    min_brightness_quality=0.3,
)

# Create pipeline configuration
pipeline_config = EnhancedPipelineConfig(
    base_config=base_config,
    enable_advanced_noise_elimination=True,
    enable_document_flattening=True,
    enable_intelligent_brightness=True,
    noise_elimination_config=noise_config,
    flattening_config=flattening_config,
    brightness_config=brightness_config,
    enable_quality_checks=True,
    quality_thresholds=quality_thresholds,
    enable_performance_logging=True,
    log_stage_timing=True,
)

# Create preprocessor
preprocessor = EnhancedDocumentPreprocessor(pipeline_config)
```

### Custom Enhancement Chain

```python
# Configure specific enhancement order
config = EnhancedPipelineConfig(
    enable_advanced_noise_elimination=True,
    enable_document_flattening=True,
    enable_intelligent_brightness=True,
    enhancement_chain=[
        EnhancementStage.BRIGHTNESS_ADJUSTMENT,  # First
        EnhancementStage.NOISE_ELIMINATION,      # Second
        EnhancementStage.DOCUMENT_FLATTENING,    # Third
    ],
)

preprocessor = EnhancedDocumentPreprocessor(config)
```

## Feature Selection

### Noise Elimination Only

```python
config = EnhancedPipelineConfig(
    enable_advanced_noise_elimination=True,
    enable_document_flattening=False,
    enable_intelligent_brightness=False,
)

preprocessor = EnhancedDocumentPreprocessor(config)
```

### Brightness Adjustment Only

```python
config = EnhancedPipelineConfig(
    enable_advanced_noise_elimination=False,
    enable_document_flattening=False,
    enable_intelligent_brightness=True,
)

preprocessor = EnhancedDocumentPreprocessor(config)
```

### Flattening + Brightness (No Noise Elimination)

```python
config = EnhancedPipelineConfig(
    enable_advanced_noise_elimination=False,
    enable_document_flattening=True,
    enable_intelligent_brightness=True,
)

preprocessor = EnhancedDocumentPreprocessor(config)
```

## Quality-Based Processing

### Understanding Quality Scores

The pipeline provides quality scores for each enhancement stage:

- **Noise Elimination**: `effectiveness_score` (0-1)
  - Measures noise reduction vs. content preservation
  - >0.5 = acceptable, >0.7 = good, >0.9 = excellent

- **Document Flattening**: `overall_quality` (0-1)
  - Combines distortion, edge preservation, and smoothness
  - >0.4 = acceptable, >0.6 = good, >0.8 = excellent

- **Brightness Adjustment**: `overall_quality` (0-1)
  - Combines contrast, uniformity, histogram spread, text preservation
  - >0.3 = acceptable, >0.5 = good, >0.7 = excellent

### Configuring Quality Thresholds

```python
# Strict thresholds (high quality requirements)
strict_thresholds = QualityThresholds(
    min_noise_elimination_effectiveness=0.7,
    min_flattening_quality=0.6,
    min_brightness_quality=0.5,
)

# Lenient thresholds (accept more results)
lenient_thresholds = QualityThresholds(
    min_noise_elimination_effectiveness=0.3,
    min_flattening_quality=0.2,
    min_brightness_quality=0.2,
)

config = EnhancedPipelineConfig(
    enable_quality_checks=True,
    quality_thresholds=strict_thresholds,
    # ... other config
)
```

### Disabling Quality Checks

```python
config = EnhancedPipelineConfig(
    enable_quality_checks=False,  # Always use enhancement results
    # ... other config
)
```

## Performance Monitoring

### Accessing Performance Metrics

```python
result = preprocessor(image)
metrics = result["metrics"]

# Total processing time
print(f"Total time: {metrics['total_time_ms']:.2f}ms")

# Individual stage times
for stage, time_ms in metrics["stage_times_ms"].items():
    print(f"{stage}: {time_ms:.2f}ms")

# Stages executed
print(f"Executed: {metrics['stages_executed']}")
print(f"Skipped: {metrics['stages_skipped']}")

# Quality scores
for stage, score in metrics["quality_scores"].items():
    print(f"{stage} quality: {score:.2f}")
```

### Performance Logging

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

config = EnhancedPipelineConfig(
    enable_performance_logging=True,
    log_stage_timing=True,
    # ... other config
)

preprocessor = EnhancedDocumentPreprocessor(config)
result = preprocessor(image)  # Logs performance info
```

## Integration Examples

### Integration with Existing Pipeline

```python
# Use enhanced pipeline as drop-in replacement
from ocr.datasets.preprocessing.enhanced_pipeline import create_office_lens_preprocessor

# Replace:
# from ocr.datasets.preprocessing import DocumentPreprocessor
# preprocessor = DocumentPreprocessor()

# With:
preprocessor = create_office_lens_preprocessor()

# Same interface
result = preprocessor(image)
processed_image = result["image"]
```

### Batch Processing

```python
import cv2
from pathlib import Path

preprocessor = create_office_lens_preprocessor()

image_paths = list(Path("input_dir").glob("*.jpg"))
results = []

for img_path in image_paths:
    image = cv2.imread(str(img_path))
    result = preprocessor(image)

    # Save processed image
    output_path = f"output_dir/{img_path.stem}_processed.jpg"
    cv2.imwrite(output_path, result["image"])

    # Collect metrics
    results.append({
        "path": str(img_path),
        "time_ms": result["metrics"]["total_time_ms"],
        "quality_scores": result["quality_assessment"],
    })

# Analyze batch performance
import pandas as pd
df = pd.DataFrame(results)
print(f"Average processing time: {df['time_ms'].mean():.2f}ms")
```

### Albumentations Integration

```python
from ocr.datasets.preprocessing.enhanced_pipeline import create_office_lens_preprocessor
from ocr.datasets.preprocessing.pipeline import LensStylePreprocessorAlbumentations
import albumentations as A

# Create enhanced preprocessor
enhanced_preprocessor = create_office_lens_preprocessor()

# Wrap for Albumentations
preprocessing_transform = LensStylePreprocessorAlbumentations(
    preprocessor=enhanced_preprocessor.base_preprocessor
)

# Use in Albumentations pipeline
transform = A.Compose([
    preprocessing_transform,
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=5, p=0.3),
])

# Apply to image and keypoints
transformed = transform(image=image, keypoints=keypoints)
```

## Best Practices

### 1. Feature Selection Based on Use Case

**For clean, well-lit documents**:
```python
preprocessor = create_fast_preprocessor()  # Fastest
```

**For documents with shadows or uneven lighting**:
```python
preprocessor = create_office_lens_preprocessor(
    enable_noise_elimination=True,
    enable_flattening=False,  # Not needed
    enable_brightness=True,
)
```

**For crumpled or deformed documents**:
```python
preprocessor = create_office_lens_preprocessor(
    enable_noise_elimination=True,
    enable_flattening=True,  # Critical
    enable_brightness=True,
)
```

### 2. Quality Threshold Tuning

Start with default thresholds and adjust based on results:

```python
# Monitor quality scores
result = preprocessor(image)
quality_scores = result["quality_assessment"]

# If scores consistently below threshold, lower threshold or
# adjust enhancement parameters
```

### 3. Performance Optimization

**For real-time processing**:
- Use `create_fast_preprocessor()`
- Disable quality checks: `enable_quality_checks=False`
- Reduce grid size for flattening: `grid_size=10`
- Use simpler methods: `CLAHE` instead of `CONTENT_AWARE`

**For batch processing (quality focus)**:
- Enable all features
- Use adaptive methods
- Enable quality checks

### 4. Error Handling

```python
try:
    result = preprocessor(image)
    processed_image = result["image"]
except Exception as e:
    logger.error(f"Preprocessing failed: {e}")
    # Fallback to basic resize
    processed_image = cv2.resize(image, (640, 640))
```

## Troubleshooting

### Issue: Processing Too Slow

**Solutions**:
1. Disable unused features
2. Use `create_fast_preprocessor()`
3. Reduce `grid_size` for flattening
4. Disable quality checks

### Issue: Quality Scores Always Low

**Solutions**:
1. Lower quality thresholds
2. Adjust enhancement method parameters
3. Check input image quality
4. Try different enhancement methods

### Issue: Results Look Worse Than Original

**Solutions**:
1. Enable quality checks
2. Increase quality thresholds
3. Disable specific features that aren't helping
4. Use AUTO methods for brightness and noise

## Performance Benchmarks

Based on validation testing (400x600 images):

| Configuration | Avg Time (ms) | Use Case |
|--------------|---------------|----------|
| Fast preprocessor | ~50ms | Real-time, clean documents |
| Noise + Brightness | ~80ms | Shadowy documents |
| Full Office Lens | ~150ms | Crumpled, noisy documents |
| Base pipeline only | ~30ms | Minimal processing |

*Times are approximate and vary by image complexity*

## Migration Guide

### From DocumentPreprocessor

```python
# Old code
from ocr.datasets.preprocessing import DocumentPreprocessor

preprocessor = DocumentPreprocessor(
    enable_enhancement=True,
    enhancement_method="office_lens",
)

# New code (enhanced features)
from ocr.datasets.preprocessing.enhanced_pipeline import create_office_lens_preprocessor

preprocessor = create_office_lens_preprocessor()

# Same result access
result = preprocessor(image)
processed_image = result["image"]
```

### Adding Advanced Features Gradually

```python
# Step 1: Start with base + brightness
config = EnhancedPipelineConfig(
    enable_intelligent_brightness=True,
)

# Step 2: Add noise elimination
config.enable_advanced_noise_elimination = True

# Step 3: Add flattening for difficult cases
config.enable_document_flattening = True
```

## See Also

- [Advanced Noise Elimination](../../../datasets/preprocessing/advanced_noise_elimination.py)
- [Document Flattening](../../../datasets/preprocessing/document_flattening.py)
- [Intelligent Brightness](../../../datasets/preprocessing/intelligent_brightness.py)
- [Base Pipeline](../../../datasets/preprocessing/pipeline.py)
- [Data Contracts](../../../pipeline/preprocessing-data-contracts.md)
