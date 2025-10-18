# **filename: docs/ai_handbook/05_changelog/2025-10/18_streamlit_encoder_compatibility_fix.md**

**Date**: 2025-10-18
**Type**: Bug Fix
**Component**: Streamlit UI, Checkpoint Catalog
**Impact**: Medium - Fixes encoder name extraction and compatibility validation

## **Summary**

Fixed encoder name extraction in the checkpoint catalog to handle cases where config parsing fails, and updated the compatibility schema to support "mobilenetv3" as an alias for "mobilenetv3_small_050".

## **Problem**

Users encountered the error "No compatibility schema found for encoder 'None'" when loading checkpoints in the Streamlit inference UI. This occurred when:

1. Checkpoint config parsing failed (missing or malformed config files)
2. The fallback directory-based encoder extraction didn't extract encoder names from experiment names
3. The schema didn't include "mobilenetv3" as an alias for the specific "mobilenetv3_small_050" model

## **Changes Made**

### **1. Enhanced Encoder Extraction**

**File**: `ui/apps/inference/services/checkpoint_catalog.py`

Modified `_extract_from_directory_structure()` to extract encoder names from experiment directory names when config parsing fails:

```python
# Also try to extract encoder from experiment name
if not metadata.encoder_name:
    exp_parts = exp_dir.replace("_", "-").split("-")
    for part in exp_parts:
        if any(keyword in part for keyword in ["resnet", "mobilenet", "efficientnet", "vgg"]):
            metadata.encoder_name = part
            break
```

### **2. State-Dict-Based Encoder Extraction**

**File**: `ui/apps/inference/services/checkpoint_catalog.py`

Added `_extract_encoder_from_state_dict()` function that analyzes PyTorch Lightning checkpoint state_dict to identify encoder architecture:

```python
def _extract_encoder_from_state_dict(checkpoint_path: Path) -> str | None:
    # Analyzes layer structure to identify ResNet variants (18, 34, 50, 101, 152)
    # Detects MobileNetV3, EfficientNet, and VGG patterns
    # Returns standardized encoder names for compatibility matching
```

### **3. Robust Fallback Chain**

Implemented a three-tier fallback system for encoder detection:
1. **Primary**: Config file parsing (`config.yaml`)
2. **Secondary**: Directory name parsing (experiment folder names)
3. **Tertiary**: State dict analysis (checkpoint contents)

### **4. Hydra Configuration**

**File**: `configs/hydra/default.yaml`

Added job configuration to ensure proper working directory handling during training.## **Testing**

- Verified that checkpoints with experiment names containing "mobilenetv3" now properly extract encoder names
- Confirmed that the compatibility schema accepts both "mobilenetv3_small_050" and "mobilenetv3" encoder names
- Tested with existing checkpoints to ensure no regression

## **Migration Notes**

- No breaking changes - existing checkpoints continue to work
- New checkpoints with directory-based naming will now be properly recognized
- The schema now supports both specific model names and common aliases
