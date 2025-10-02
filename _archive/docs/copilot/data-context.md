# Data Context and Domain Knowledge

## Dataset Structure

### ICDAR Format (Competition Standard)
```
data/
├── datasets/
│   ├── images/
│   │   ├── train/           # Training images
│   │   ├── val/            # Validation images
│   │   └── test/           # Test images
│   └── jsons/
│       ├── train.json      # Training annotations
│       ├── val.json       # Validation annotations
│       └── test.json      # Test annotations (no GT)
```

### Annotation Format
```json
{
  "images": {
    "image1.jpg": {
      "words": {
        "word_1": {
          "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        },
        "word_2": {
          "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        }
      }
    }
  }
}
```

## Data Processing Pipeline

### 1. Dataset Class (OCRDataset)
```python
class OCRDataset(Dataset):
    def __init__(self, image_path, annotation_path, transform):
        # Loads images and annotations
        # Handles EXIF orientation
        # Filters images without annotations

    def __getitem__(self, idx):
        # Returns: {
        #     'image': transformed_image,
        #     'polygons': ground_truth_polygons,
        #     'inverse_matrix': transform_matrix
        # }
```

### 2. Transform Pipeline
- **Albumentations**: Image augmentations
- **Normalization**: Image normalization
- **Polygon transforms**: Coordinate transformations

### 3. DataLoader Configuration
```python
# Typical config
dataloader:
  batch_size: 8
  num_workers: 4
  shuffle: true
  collate_fn: db_collate_fn  # Custom collation for polygons
```

## Evaluation Metrics

### CLEval (Character-Level Evaluation)
- **Purpose**: Industry-standard OCR evaluation
- **Metrics**: Precision, Recall, F1-score
- **Granularity**: Character-level evaluation

### Key Parameters
```python
cleval_metric = CLEvalMetric(
    case_sensitive=True,           # Case sensitivity
    recall_gran_penalty=1.0,       # Granularity penalty weights
    precision_gran_penalty=1.0,
    vertical_aspect_ratio_thresh=0.5,
    ap_constraint=0.3             # Area precision constraint
)
```

### Evaluation Process
1. **Model Prediction**: Generate probability maps
2. **Post-processing**: Extract polygons from maps
3. **Matching**: Match predicted polygons to ground truth
4. **Scoring**: Calculate precision/recall at character level

## Common Data Challenges

### 1. Polygon Format Issues
- **Self-intersecting polygons**: Invalid geometry
- **Empty polygons**: Zero-area regions
- **Incorrect winding**: Clockwise vs counter-clockwise

### 2. Image Quality Issues
- **EXIF orientation**: Images not properly rotated
- **Color spaces**: RGB vs BGR vs grayscale
- **Compression artifacts**: JPEG artifacts affecting text

### 3. Annotation Quality
- **Missing annotations**: Images without ground truth
- **Inconsistent labeling**: Different annotators, different standards
- **Occlusion**: Text partially obscured

## Augmentation Strategies

### Geometric Transformations
```python
import albumentations as A

geometric_augs = A.Compose([
    A.Rotate(limit=10, p=0.5),
    A.Affine(scale=(0.8, 1.2), rotate=(-5, 5), p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
])
```

### Color Transformations
```python
color_augs = A.Compose([
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.RandomBrightnessContrast(p=0.5),
])
```

### Custom Augmentations
- **Text-specific**: Blur, noise, compression artifacts
- **Document-specific**: Paper texture, lighting variations
- **Geometric**: Perspective transforms, elastic deformations

## Data Validation

### 1. Image Validation
```python
def validate_image(image_path: Path) -> bool:
    """Validate image can be loaded and processed"""
    try:
        img = Image.open(image_path)
        img.verify()
        return img.format in ['JPEG', 'PNG']
    except Exception:
        return False
```

### 2. Annotation Validation
```python
def validate_polygons(polygons: List[np.ndarray]) -> List[np.ndarray]:
    """Filter valid polygons"""
    valid_polygons = []
    for poly in polygons:
        if len(poly) >= 3 and cv2.contourArea(poly) > 0:
            valid_polygons.append(poly)
    return valid_polygons
```

## Performance Considerations

### 1. Memory Usage
- **Large images**: Resize or crop to manageable sizes
- **Batch size**: Balance GPU memory vs training speed
- **Data types**: Use float16 where possible

### 2. I/O Bottlenecks
- **Num workers**: Optimize DataLoader workers
- **Caching**: Cache processed data in memory/disk
- **Prefetching**: Use DataLoader prefetching

### 3. Augmentation Speed
- **CPU intensive**: Use efficient libraries (albumentations)
- **GPU acceleration**: Move augmentations to GPU when possible
- **Caching**: Cache expensive augmentations

## Debugging Data Issues

### 1. Visualization Tools
```python
def visualize_sample(image, polygons, predictions=None):
    """Visualize image with ground truth and predictions"""
    # Draw polygons on image
    # Show side-by-side comparison
    pass
```

### 2. Statistical Analysis
```python
def analyze_dataset(dataset):
    """Analyze dataset statistics"""
    # Polygon counts, sizes, aspect ratios
    # Image dimensions, color distributions
    # Text length distributions
    pass
```

### 3. Error Analysis
```python
def analyze_errors(predictions, ground_truth):
    """Analyze prediction errors"""
    # False positives, false negatives
    # Error patterns by text size/orientation
    pass
```</content>
<parameter name="filePath">/home/vscode/workspace/upstage-receipt-text-detection-dbnet-baseline/docs/copilot/data-context.md
