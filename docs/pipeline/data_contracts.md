# Data Contracts: OCR Pipeline Shape and Type Specifications

**Purpose**: This document defines the exact data contracts (shapes, types, and formats) that must be maintained across the OCR pipeline. These contracts prevent the repetitive data type/shape errors that have caused multiple commit rollbacks.

**Audience**: Developers (human + AI) adding new features, debugging issues, or modifying the pipeline.

**Enforcement**: Contracts are validated by automated tests and runtime checks.

---

## 📋 Table of Contents

1. [Dataset Output Contract](#dataset-output-contract)
2. [Transform Pipeline Contract](#transform-pipeline-contract)
3. [Collate Function Contract](#collate-function-contract)
4. [Model Input/Output Contract](#model-inputoutput-contract)
5. [Loss Function Contract](#loss-function-contract)
6. [Common Data Types](#common-data-types)
7. [Validation Rules](#validation-rules)
8. [Debugging Guide](#debugging-guide)

---

## Dataset Output Contract

### OCRDataset.__getitem__() → dict

**Purpose**: Defines the exact format returned by dataset samples before transformation.

```python
{
    "image": np.ndarray,           # Shape: (H, W, 3), dtype: uint8 or float32
    "polygons": List[np.ndarray],  # Each: shape (N, 2), dtype: float32
    "prob_maps": np.ndarray,       # Shape: (H, W), dtype: float32, range: [0, 1]
    "thresh_maps": np.ndarray,     # Shape: (H, W), dtype: float32, range: [0, 1]
    "image_filename": str,         # Relative path to image file
    "image_path": str,             # Absolute path to image file
    "inverse_matrix": np.ndarray,  # Shape: (3, 3), dtype: float32 (affine transform)
    "shape": Tuple[int, int],      # Original image shape (H, W)
}
```

**Validation Rules**:
- `image.shape[2] == 3` (RGB channels)
- `len(polygons)` ≥ 0 (can be empty for images without text)
- All polygons have shape `(N, 2)` where N ≥ 3
- `prob_maps.shape == thresh_maps.shape == (H, W)`
- `image.shape[:2] == (H, W)` matches prob_maps shape

**Common Violations**:
- PIL Image passed instead of numpy array
- Polygons with wrong shape `(1, N, 2)` instead of `(N, 2)`
- Missing or extra channels in image

---

## Transform Pipeline Contract

### DBTransforms.__call__() → dict

**Purpose**: Defines input/output contract for the Albumentations-based transform pipeline.

**Input Contract**:
```python
{
    "image": np.ndarray,           # Shape: (H, W, 3), dtype: uint8 or float32
    "polygons": List[np.ndarray],  # Each: shape (N, 2), dtype: float32
}
```

**Output Contract**:
```python
{
    "image": torch.Tensor,         # Shape: (3, H', W'), dtype: float32, normalized
    "polygons": List[np.ndarray],  # Each: shape (M, 2), dtype: float32 (transformed)
    "prob_maps": torch.Tensor,     # Shape: (1, H', W'), dtype: float32
    "thresh_maps": torch.Tensor,   # Shape: (1, H', W'), dtype: float32
    "inverse_matrix": np.ndarray,  # Shape: (3, 3), dtype: float32 (updated)
}
```

**Key Transformations**:
- Image: `uint8 (H,W,3)` → `float32 (3,H',W')` (normalized to [-2.1, 2.6])
- Polygons: Geometric transformation applied, may change point count
- Maps: `float32 (H,W)` → `float32 (1,H',W')` (added channel dimension)

**Albumentations Contract**:
- Must return dict with `"image"` key
- Keypoints (polygons) automatically transformed
- Custom transforms must inherit from `A.ImageOnlyTransform` or `A.DualTransform`

---

## Collate Function Contract

### DBCollateFN.__call__() → dict

**Purpose**: Defines batch collation contract for DataLoader.

**Input Contract** (batch: List[dict]):
```python
[
    {
        "image": torch.Tensor,         # Shape: (3, H_i, W_i), dtype: float32
        "polygons": List[np.ndarray],  # Variable length, each shape (N, 2)
        "prob_maps": torch.Tensor,     # Shape: (1, H_i, W_i), dtype: float32
        "thresh_maps": torch.Tensor,   # Shape: (1, H_i, W_i), dtype: float32
        "image_filename": str,
        "image_path": str,
        "inverse_matrix": np.ndarray,  # Shape: (3, 3)
        "shape": Tuple[int, int],
    },
    # ... batch_size items with potentially different H_i, W_i
]
```

**Output Contract**:
```python
{
    "images": torch.Tensor,        # Shape: (batch_size, 3, H_max, W_max), dtype: float32
    "polygons": List[List[np.ndarray]],  # Shape: (batch_size, variable), each polygon (N, 2)
    "prob_maps": torch.Tensor,     # Shape: (batch_size, 1, H_max, W_max), dtype: float32
    "thresh_maps": torch.Tensor,   # Shape: (batch_size, 1, H_max, W_max), dtype: float32
    "image_filenames": List[str],  # Length: batch_size
    "image_paths": List[str],      # Length: batch_size
    "inverse_matrices": List[np.ndarray],  # Length: batch_size, each (3, 3)
    "shapes": List[Tuple[int, int]],      # Length: batch_size
}
```

**Key Operations**:
- **Padding**: Images/maps padded to `max(H_i, W_i)` with zeros
- **Polygon Filtering**: Invalid polygons removed, shapes normalized to `(N, 2)`
- **Batch Stacking**: Individual tensors stacked into batch dimension

**Validation Rules**:
- All output tensors have consistent batch dimension
- Polygon shapes normalized (no `(1, N, 2)` shapes allowed)
- Image and map dimensions match after padding

---

## Model Input/Output Contract

### OCRModel.forward() → dict

**Purpose**: Defines the complete model pipeline contract.

**Input Contract**:
```python
{
    "images": torch.Tensor,        # Shape: (B, 3, H, W), dtype: float32
    "prob_maps": torch.Tensor,     # Shape: (B, 1, H, W), dtype: float32 (optional)
    "thresh_maps": torch.Tensor,   # Shape: (B, 1, H, W), dtype: float32 (optional)
    # ... other batch fields from collate
}
```

**Output Contract** (training mode):
```python
{
    "prob_maps": torch.Tensor,     # Shape: (B, 1, H', W'), predicted probabilities
    "thresh_maps": torch.Tensor,   # Shape: (B, 1, H', W'), predicted thresholds
    "binary_maps": torch.Tensor,   # Shape: (B, 1, H', W'), binarized predictions
    "loss": torch.Tensor,          # Shape: (), scalar loss value
    "loss_dict": dict,             # Detailed loss components
}
```

**Output Contract** (inference mode):
```python
{
    "prob_maps": torch.Tensor,     # Shape: (B, 1, H', W')
    "thresh_maps": torch.Tensor,   # Shape: (B, 1, H', W')
    "binary_maps": torch.Tensor,   # Shape: (B, 1, H', W')
}
```

**Pipeline Stages**:
1. **Encoder**: `(B, 3, H, W)` → `List[torch.Tensor]` (multi-scale features)
2. **Decoder**: `List[torch.Tensor]` → `torch.Tensor (B, C, H', W')` (C=256 typically)
3. **Head**: `torch.Tensor (B, C, H', W')` → model outputs

**Common Issues**:
- Head expecting 256 channels but receiving 3 (raw images)
- Wrong spatial dimensions between encoder/decoder/head
- Loss expecting different shapes than model outputs

---

## Loss Function Contract

### DBLoss.forward() → Tuple[torch.Tensor, dict]

**Purpose**: Defines loss computation contract.

**Input Contract**:
```python
pred: dict = {
    "prob_maps": torch.Tensor,     # Shape: (B, 1, H, W), predicted probabilities
    "thresh_maps": torch.Tensor,   # Shape: (B, 1, H, W), predicted thresholds
    "binary_maps": torch.Tensor,   # Shape: (B, 1, H, W), binarized predictions
}

gt_binary: torch.Tensor,           # Shape: (B, 1, H, W), ground truth binary map
gt_thresh: torch.Tensor,           # Shape: (B, 1, H, W), ground truth threshold map
```

**Output Contract**:
```python
loss: torch.Tensor,                # Shape: (), scalar total loss
loss_dict: dict = {
    "loss_prob": torch.Tensor,     # Probability map loss
    "loss_thresh": torch.Tensor,   # Threshold map loss
    "loss_binary": torch.Tensor,   # Binary map loss
    # ... other loss components
}
```

**Validation Rules**:
- All tensors must have identical spatial dimensions `(H, W)`
- Batch sizes must match across all inputs
- Ground truth tensors required for training

---

## Common Data Types

### Polygon Representations

```python
# Single polygon - CORRECT format
polygon: np.ndarray = np.array([
    [100, 200],  # x, y coordinates
    [150, 200],
    [150, 250],
    [100, 250]
], dtype=np.float32)  # Shape: (4, 2)

# Multiple polygons in batch
polygons: List[np.ndarray] = [polygon1, polygon2, ...]

# WRONG formats (cause collate errors):
wrong_polygon = np.array([polygon])  # Shape: (1, 4, 2) - extra dimension
```

### Image Formats

```python
# Raw image (dataset output)
image: np.ndarray  # Shape: (H, W, 3), dtype: uint8, range: [0, 255]

# Normalized image (after transforms)
image: torch.Tensor  # Shape: (3, H, W), dtype: float32, range: [-2.1, 2.6]

# Batch of images
images: torch.Tensor  # Shape: (B, 3, H, W), dtype: float32
```

### Map Formats

```python
# Single map (dataset output)
prob_map: np.ndarray  # Shape: (H, W), dtype: float32, range: [0, 1]

# After collate (with channel dimension)
prob_maps: torch.Tensor  # Shape: (B, 1, H, W), dtype: float32

# Model prediction
pred_maps: torch.Tensor  # Shape: (B, 1, H', W'), dtype: float32
```

---

## Validation Rules

### Runtime Validation Functions

```python
def validate_dataset_sample(sample: dict) -> None:
    """Validate dataset output contract."""
    assert "image" in sample
    assert isinstance(sample["image"], np.ndarray)
    assert sample["image"].ndim == 3
    assert sample["image"].shape[2] == 3
    # ... additional checks

def validate_collate_batch(batch: dict) -> None:
    """Validate collate output contract."""
    assert "images" in batch
    assert isinstance(batch["images"], torch.Tensor)
    assert batch["images"].dim() == 4
    assert batch["images"].shape[1] == 3  # RGB channels
    # ... additional checks
```

### Automated Testing

All contracts validated by:
- `tests/integration/test_collate_integration.py` - End-to-end pipeline
- `tests/ocr/datasets/test_polygon_filtering.py` - Polygon shape handling
- `tests/ocr/datasets/test_transform_pipeline_contracts.py` - Transform contracts
- `scripts/validate_pipeline_contracts.py` - Runtime validation utility

---

## Debugging Guide

### Common Error Patterns

1. **"Image object has no attribute 'shape'"**
   - **Cause**: PIL Image passed where numpy array expected
   - **Fix**: Convert PIL → numpy: `np.array(pil_image)`

2. **"Target size must be the same as input size"**
   - **Cause**: Loss inputs have mismatched dimensions
   - **Fix**: Ensure model output and ground truth have same spatial size

3. **"Expected 256 channels, got 3"**
   - **Cause**: Raw images passed to head instead of decoder features
   - **Fix**: Use full pipeline: `images → encoder → decoder → head`

4. **"Polygon shape (1, N, 2) invalid"**
   - **Cause**: Extra dimension in polygon arrays
   - **Fix**: Squeeze: `polygon = polygon.squeeze(0)` or normalize in collate

### Quick Fixes

```python
# PIL → NumPy conversion
if isinstance(image, Image.Image):
    image = np.array(image)

# Polygon shape normalization
for i, polygon in enumerate(polygons):
    if polygon.ndim == 3 and polygon.shape[0] == 1:
        polygons[i] = polygon.squeeze(0)

# Tensor dimension alignment
if pred_maps.shape[-2:] != gt_maps.shape[-2:]:
    # Resize prediction to match ground truth
    pred_maps = F.interpolate(pred_maps, size=gt_maps.shape[-2:], mode='bilinear')
```

### Prevention Checklist

- [ ] Read this document before modifying pipeline components
- [ ] Run `scripts/validate_pipeline_contracts.py` after changes
- [ ] Execute `tests/integration/test_collate_integration.py` before commit
- [ ] Check tensor shapes with `print(x.shape)` at pipeline stages
- [ ] Use type hints and validate inputs in new functions

---

## Contract Evolution

**When to Update**: When adding new features that change data formats.

**Process**:
1. Update this document first
2. Update validation functions
3. Update tests
4. Update implementation
5. Verify with integration tests

**Breaking Changes**: Require updating all downstream components.

---

**Last Updated**: October 11, 2025
**Version**: 1.0
**Maintainer**: Data Pipeline Team</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/pipeline/data_contracts.md
