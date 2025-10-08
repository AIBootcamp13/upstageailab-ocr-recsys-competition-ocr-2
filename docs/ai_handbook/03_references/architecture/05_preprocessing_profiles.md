# **Reference: Preprocessing Profiles and Critical Matching Requirement**

This document details the preprocessing profile system and the critical requirement to maintain consistency between training and inference preprocessing.

## **1. Overview of Preprocessing Profiles**

The system provides several preprocessing profiles through the Command Builder UI to enhance OCR model performance:

### **1.1 Available Profiles**

| Profile | ID | Description | Use Case |
|---------|----|-------------|----------|
| **None** | `none` | Standard transforms without preprocessing | Clean datasets, baseline experiments |
| **Lens-style (balanced)** | `lens_style` | Document detection + perspective correction + conservative enhancement | General receipt/document OCR |
| **Lens-style + Office mode** | `lens_style_office` | Adds text enhancement and aggressive sharpening | Faint receipts, low-quality scans |
| **CamScanner** | `camscanner` | Uses CamScanner's LSD line detection for document boundaries | Complex backgrounds, tilted documents |
| **docTR demo** | `doctr_demo` | Full docTR pipeline with geometry rectification | Experimental, most sophisticated |

### **1.2 Technical Implementation**

Preprocessing profiles are implemented through:

1. **UI Integration**: Dropdown selectors in Command Builder Train, Test, and Predict pages
2. **Hydra Override Generation**: Automatic generation of appropriate Hydra overrides when a profile is selected
3. **Configuration Files**:
   - `configs/data/preprocessing.yaml` - Data config that uses preprocessing transforms
   - `configs/preset/datasets/preprocessing*.yaml` - Profile-specific presets

## **2. Critical Rule: Training-Inference Consistency**

### **⚠️ CRITICAL REQUIREMENT**

**You MUST use the same preprocessing profile for inference/prediction that you used during training!**

This is absolutely critical because:

- **Model Learning**: The model learns to recognize text patterns from preprocessed images during training
- **Distribution Shift**: Different preprocessing at inference creates a distribution shift from the training data
- **Document Detection**: Document detection and perspective correction fundamentally alter the input characteristics
- **Performance Degradation**: Mismatched preprocessing will cause significantly degraded performance

### **2.1 Example of Proper Usage**

**Training with CamScanner preprocessing:**
```bash
# Through Command Builder UI - select "CamScanner document detection" profile
# Or manually:
uv run python runners/train.py \
  data=preprocessing \
  +preset/datasets=preprocessing_camscanner \
  ...
```

**Inference with matching preprocessing (CORRECT):**
```bash
# Through Command Builder UI - select SAME "CamScanner document detection" profile
# Or manually:
uv run python runners/predict.py \
  data=preprocessing \
  +preset/datasets=preprocessing_camscanner \
  ...
```

**Inference without matching preprocessing (WRONG):**
```bash
# NEVER do this after training with preprocessing!
uv run python runners/predict.py \
  data=default \
  # This will fail to perform well!
```

## **3. How Preprocessing Profiles Work**

When you select a profile in the Command Builder:

1. **Automatic Override Generation**: The UI automatically generates appropriate Hydra overrides
2. **Configuration Application**:
   - For training: `data=preprocessing` + appropriate preset
   - For inference/test: matching `data=preprocessing` + preset
3. **Pipeline Activation**: The preprocessing pipeline activates with the selected profile settings

## **4. Profile-Specific Details**

### **4.1 Lens-style Profile**
- Uses document detection to locate document boundaries
- Applies perspective correction to rectify tilted documents
- Conservative enhancement to improve contrast

### **4.2 CamScanner Profile**
- Uses LSD (Line Segment Detector) algorithm for document boundary detection
- More aggressive perspective correction
- Better for complex backgrounds

### **4.3 docTR Demo Profile**
- Full docTR pipeline integration
- Orientation detection and correction
- Advanced geometry rectification

## **5. Verification Steps**

To verify preprocessing consistency:

1. **Check Training Config**: Confirm preprocessing profile used during training
2. **Verify Inference Config**: Ensure matching profile is used for inference
3. **Validate Results**: Compare performance with baseline to ensure preprocessing is beneficial
4. **Monitor Errors**: Look for distribution shift symptoms like poor confidence calibration
