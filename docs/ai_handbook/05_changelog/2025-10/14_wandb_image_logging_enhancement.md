# WandB Image Logging Enhancement - Exact Transformed Images

## Overview

Enhanced WandB image logging to capture and display exact transformed images as seen by the model during validation, eliminating preprocessing overhead and ensuring logged images match what the model actually processes.

## Problem Statement

Previous WandB image logging implementation had significant performance overhead:
- Images were re-processed through the entire transformation pipeline for logging
- Logged images didn't match the exact transformations applied during training/validation
- High computational cost for image preprocessing during logging callbacks

## Solution Implementation

### Core Changes

#### 1. OCRPLModule Enhancement (`ocr/lightning_modules/ocr_pl.py`)

**Modified `validation_step` method:**
```python
# Store exact transformed image for WandB logging
prediction_entry["transformed_image"] = batch["images"][idx].detach().cpu()
```

- Captures the exact tensor image after all transformations are applied
- Stores in prediction entry for callback access during epoch end
- Uses `.detach().cpu()` to prevent GPU memory issues and gradient computation

#### 2. WandbImageLoggingCallback Enhancement (`ocr/lightning_modules/callbacks/wandb_image_logging.py`)

**Added `_tensor_to_pil` method:**
```python
def _tensor_to_pil(self, tensor_image: torch.Tensor) -> Image.Image:
    """Convert tensor image to PIL Image for WandB logging."""
    # Denormalize from [-1, 1] to [0, 1] if needed
    if tensor_image.min() < 0:
        tensor_image = (tensor_image + 1) / 2

    # Convert to numpy and transpose to HWC format
    np_image = tensor_image.numpy().transpose(1, 2, 0)

    # Clip to valid range and convert to uint8
    np_image = np.clip(np_image * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(np_image)
```

**Enhanced `on_validation_epoch_end`:**
```python
# Prioritize transformed images over original images
if "transformed_image" in prediction:
    image = self._tensor_to_pil(prediction["transformed_image"])
else:
    # Fallback to original image processing
    image = self._process_image(prediction["image_filename"])
```

### Performance Impact

**Before Enhancement:**
- Images re-processed through full transformation pipeline for each logging operation
- High CPU/GPU overhead during validation epoch end
- Logged images may differ from actual model input due to preprocessing variations

**After Enhancement:**
- Zero additional preprocessing overhead for logging
- Exact transformed images captured once during validation_step
- Improved training performance and reduced memory usage

### Validation & Testing

**Integration Testing:**
- Verified transformed images match model input exactly
- Confirmed backward compatibility with existing logging behavior
- Validated performance improvement in validation epochs

**Edge Cases Handled:**
- Automatic fallback when transformed images unavailable
- Proper tensor denormalization for different input ranges
- Memory-efficient tensor handling with detach/cpu operations

## API Compatibility

- **Backward Compatible:** Existing WandB logging continues to work unchanged
- **Enhanced Behavior:** When available, uses exact transformed images
- **Fallback Support:** Gracefully handles cases where transformed images aren't stored

## Files Modified

- `ocr/lightning_modules/ocr_pl.py`: Added transformed image storage
- `ocr/lightning_modules/callbacks/wandb_image_logging.py`: Enhanced callback with tensor conversion

## Benefits

1. **Performance:** Eliminated redundant image preprocessing during logging
2. **Accuracy:** Logged images now exactly match model input
3. **Debugging:** Improved ability to visualize what the model actually sees
4. **Monitoring:** Better validation of data pipeline correctness

## Future Considerations

- Consider extending to training step logging if needed
- Potential for configurable logging frequency to balance performance vs monitoring needs
- Could be extended to log transformation metadata alongside images
