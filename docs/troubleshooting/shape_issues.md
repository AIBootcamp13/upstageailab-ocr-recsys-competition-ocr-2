# Troubleshooting: Shape and Type Issues

**Purpose**: Quick reference guide for diagnosing and fixing the most common data contract violations in the OCR pipeline.

**Symptoms**: Training crashes, shape mismatch errors, type errors, unexpected NaN/inf values.

---

## 🚨 Quick Diagnosis

### 1. Check Error Type

| Error Pattern | Likely Cause | Quick Fix |
|---------------|-------------|-----------|
| `"Image object has no attribute 'shape'"` | PIL Image passed to numpy function | Convert: `np.array(pil_image)` |
| `"Target size must be the same"` | Dimension mismatch in loss | Check upscale factors, interpolate |
| `"expected X channels, got Y"` | Wrong tensor passed to layer | Check pipeline stage order |
| `"Polygon shape (1, N, 2) invalid"` | Extra polygon dimension | Squeeze: `polygon.squeeze(0)` |
| `NaN in loss` | Invalid input data | Check normalization, data ranges |

### 2. Check Pipeline Stage

Run this diagnostic script:

```python
# debug/pipeline_diagnostic.py
import torch
from ocr.datasets import OCRDataset, DBCollateFN
from ocr.datasets.transforms import DBTransforms

def diagnose_pipeline():
    """Step-by-step pipeline diagnosis."""
    print("🔍 OCR Pipeline Diagnostic")
    print("=" * 50)

    # 1. Dataset output
    dataset = OCRDataset(...)
    sample = dataset[0]
    print(f"Dataset sample keys: {list(sample.keys())}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Image type: {type(sample['image'])}")
    print(f"Polygons count: {len(sample['polygons'])}")

    # 2. Transform output
    transforms = DBTransforms(...)
    transformed = transforms(sample)
    print(f"Transformed image shape: {transformed['image'].shape}")
    print(f"Transformed image type: {type(transformed['image'])}")

    # 3. Collate output
    collate_fn = DBCollateFN()
    batch = collate_fn([transformed])
    print(f"Batch images shape: {batch['images'].shape}")
    print(f"Batch images type: {type(batch['images'])}")

    print("✅ Pipeline diagnostic complete")

if __name__ == "__main__":
    diagnose_pipeline()
```

---

## 🔧 Common Fixes

### Fix 1: PIL Image Type Confusion

**Error**: `AttributeError: 'Image' object has no attribute 'shape'`

**Root Cause**: PIL Images passed where numpy arrays expected.

**Locations**: `base.py` lines 325, 174, 278

**Fix**:
```python
# Before
image.shape  # Fails on PIL Image

# After
def safe_get_image_size(image) -> Tuple[int, int]:
    """Safely get image size regardless of type."""
    if isinstance(image, Image.Image):
        return image.size  # PIL: (W, H)
    elif isinstance(image, np.ndarray):
        return image.shape[1], image.shape[0]  # NumPy: (H, W, C) → (W, H)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

# Usage
width, height = safe_get_image_size(image)
```

### Fix 2: Polygon Shape Normalization

**Error**: `ValueError: Polygon shape (1, N, 2) invalid`

**Root Cause**: Extra dimension in polygon arrays from collate function.

**Fix**:
```python
def normalize_polygon_shapes(polygons: List[np.ndarray]) -> List[np.ndarray]:
    """Normalize polygon shapes to (N, 2) format."""
    normalized = []
    for polygon in polygons:
        if polygon.ndim == 3 and polygon.shape[0] == 1:
            # Remove extra batch dimension
            polygon = polygon.squeeze(0)
        elif polygon.ndim == 2:
            # Already correct shape
            pass
        else:
            # Invalid shape, skip
            continue
        normalized.append(polygon)
    return normalized

# Usage in collate function
batch["polygons"] = [normalize_polygon_shapes(polys) for polys in batch["polygons"]]
```

### Fix 3: Model Input Channel Mismatch

**Error**: `RuntimeError: expected input[2, 256, H, W] to have 256 channels, but got 3`

**Root Cause**: Raw images passed to head instead of decoder features.

**Fix**:
```python
# WRONG: Direct image to head
predictions = head(batch["images"])  # 3 channels

# CORRECT: Full pipeline
encoded = encoder(batch["images"])    # List of feature tensors
decoded = decoder(encoded)            # 256 channels
predictions = head(decoded)           # Expects 256 channels

# OR for testing: Mock decoder output
mock_decoder = torch.randn(batch_size, 256, H, W)
predictions = head(mock_decoder)
```

### Fix 4: Loss Dimension Mismatch

**Error**: `ValueError: Target size (B, 1, H, W) must be the same as input size (B, 1, H', W')`

**Root Cause**: Head upscale factor not matched in ground truth.

**Fix**:
```python
import torch.nn.functional as F

# Head has upscale=4, so predictions are 4x larger
pred_prob = predictions["prob_maps"]    # Shape: (B, 1, 4*H, 4*W)
gt_prob = batch["prob_maps"]            # Shape: (B, 1, H, W)

# Resize ground truth to match predictions
gt_prob_resized = F.interpolate(gt_prob, scale_factor=4, mode='bilinear')
gt_thresh_resized = F.interpolate(batch["thresh_maps"], scale_factor=4, mode='bilinear')

# Compute loss
loss, loss_dict = loss_fn(predictions, gt_prob_resized, gt_thresh_resized)
```

### Fix 5: Transform Contract Violation

**Error**: Albumentations errors or missing keys in transform output.

**Root Cause**: Custom transforms not following Albumentations contract.

**Fix**:
```python
# WRONG: Direct return (bypasses Albumentations)
class WrongTransform:
    def __call__(self, image, **kwargs):
        return image * 0.9  # Direct return

# CORRECT: Albumentations contract
class CorrectTransform(A.ImageOnlyTransform):
    def apply(self, img, **params):
        return img * 0.9  # Albumentations handles return

    def get_transform_init_args_names(self):
        return []  # Required method
```

---

## 🔍 Deep Diagnosis

### Step-by-Step Debugging

1. **Isolate the Failing Component**
   ```python
   # Test dataset alone
   sample = dataset[0]
   print("Dataset OK")

   # Test transforms alone
   transformed = transforms(sample)
   print("Transforms OK")

   # Test collate alone
   batch = collate_fn([transformed])
   print("Collate OK")

   # Test model alone
   predictions = model(batch["images"])
   print("Model OK")
   ```

2. **Check Tensor Shapes at Each Stage**
   ```python
   def print_shapes(data, prefix=""):
       if isinstance(data, dict):
           for k, v in data.items():
               print_shapes(v, f"{prefix}.{k}")
       elif isinstance(data, torch.Tensor):
           print(f"{prefix}: {data.shape} {data.dtype}")
       elif isinstance(data, np.ndarray):
           print(f"{prefix}: {data.shape} {data.dtype}")
   ```

3. **Validate Against Contracts**
   ```python
   from docs.pipeline.data_contracts import validate_dataset_sample

   try:
       validate_dataset_sample(sample)
       print("✅ Dataset contract OK")
   except AssertionError as e:
       print(f"❌ Dataset contract violation: {e}")
   ```

### Memory and Performance Issues

**High Memory Usage**:
- Check for tensor accumulation: `del intermediate_tensors`
- Use gradient checkpointing for large models
- Reduce batch size

**Slow Training**:
- Profile with `torch.profiler`
- Check for unnecessary tensor conversions
- Optimize data loading pipeline

---

## 🛠️ Development Tools

### Contract Validation Script

```bash
# Run before committing
python scripts/validate_pipeline_contracts.py

# Run specific component tests
python -m pytest tests/ -k "contract" -v

# Debug with shapes
python debug/shape_debugger.py
```

### IDE Integration

**VS Code Debug Configuration**:
```json
{
    "name": "Debug Pipeline",
    "type": "python",
    "request": "launch",
    "program": "debug/pipeline_diagnostic.py",
    "console": "integratedTerminal"
}
```

**PyCharm Type Hints**:
```python
from typing import Dict, List, Tuple, Union
import torch
import numpy as np
from PIL import Image

def process_sample(sample: Dict[str, Union[np.ndarray, List[np.ndarray]]]) -> Dict[str, torch.Tensor]:
    """Process with proper type hints."""
    # Implementation
```

---

## 📚 Prevention Guidelines

### Code Review Checklist

- [ ] Data types validated at function boundaries
- [ ] Tensor shapes documented in docstrings
- [ ] Error messages include shape information
- [ ] Unit tests include shape assertions
- [ ] Integration tests cover full pipeline

### Best Practices

1. **Always check shapes**: `assert x.shape == expected_shape`
2. **Use type hints**: `def func(x: torch.Tensor) -> torch.Tensor:`
3. **Validate inputs**: Early failure with clear messages
4. **Document contracts**: Update `data_contracts.md` for changes
5. **Test edge cases**: Empty batches, extreme dimensions

### Template for New Components

```python
def new_pipeline_component(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process pipeline data.

    Args:
        data: Input data conforming to contract X

    Returns:
        Processed data conforming to contract Y

    Contracts:
        Input: See docs/pipeline/data_contracts.md#contract-x
        Output: See docs/pipeline/data_contracts.md#contract-y
    """
    # Validate input contract
    validate_input_contract(data)

    # Process data
    result = process_data(data)

    # Validate output contract
    validate_output_contract(result)

    return result
```

---

## 🚨 Emergency Fixes

### When Training Crashes Immediately

1. **Check data loading**: `python -c "dataset[0]; print('Dataset OK')"`
2. **Check transforms**: `python -c "transforms(dataset[0]); print('Transforms OK')"`
3. **Check collate**: `python -c "collate([transforms(dataset[0])]); print('Collate OK')"`
4. **Simplify model**: Use dummy model that just returns zeros

### When Loss is NaN

1. **Check input ranges**: All tensors in valid ranges?
2. **Check normalization**: Images properly normalized?
3. **Check ground truth**: Valid probability/threshold maps?
4. **Gradient clipping**: Add `torch.nn.utils.clip_grad_norm_()`

---

**Last Updated**: October 11, 2025
**Version**: 1.0
**Quick Reference**: See `docs/pipeline/data_contracts.md` for complete specifications</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/troubleshooting/shape_issues.md
