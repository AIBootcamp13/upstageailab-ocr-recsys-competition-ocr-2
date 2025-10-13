# Preprocessing Profiles

> **AI Cues**
> - **priority**: critical
> - **use_when**: users need to configure preprocessing, ensure training-inference consistency, select appropriate profiles for different document types

## Overview

This reference details the preprocessing profile system that enhances OCR model performance through document detection, perspective correction, and image enhancement. The system provides multiple profiles accessible through the Command Builder UI, with a critical requirement to maintain consistency between training and inference preprocessing.

## Key Concepts

### Preprocessing Profiles

- **Profile System**: Collection of predefined preprocessing pipelines for different document types
- **Document Detection**: Automatic identification of document boundaries in images
- **Perspective Correction**: Geometric rectification of tilted or skewed documents
- **Image Enhancement**: Contrast improvement and text sharpening techniques
- **Training-Inference Consistency**: Critical requirement to use identical preprocessing during training and inference

### Available Profiles

| Profile | ID | Description | Use Case |
|---------|----|-------------|----------|
| **None** | `none` | Standard transforms without preprocessing | Clean datasets, baseline experiments |
| **Lens-style (balanced)** | `lens_style` | Document detection + perspective correction + conservative enhancement | General receipt/document OCR |
| **Lens-style + Office mode** | `lens_style_office` | Adds text enhancement and aggressive sharpening | Faint receipts, low-quality scans |
| **CamScanner** | `camscanner` | Uses CamScanner's LSD line detection for document boundaries | Complex backgrounds, tilted documents |
| **docTR demo** | `doctr_demo` | Full docTR pipeline with geometry rectification | Experimental, most sophisticated |

### Critical Consistency Rule

**⚠️ CRITICAL REQUIREMENT**: You MUST use the same preprocessing profile for inference/prediction that you used during training!

- **Model Learning**: Models learn patterns from preprocessed training images
- **Distribution Shift**: Different preprocessing creates training/inference mismatch
- **Performance Impact**: Mismatched preprocessing causes significant performance degradation

## Detailed Information

### Technical Implementation

Preprocessing profiles are implemented through:

1. **UI Integration**: Dropdown selectors in Command Builder Train, Test, and Predict pages
2. **Hydra Override Generation**: Automatic generation of appropriate Hydra overrides when a profile is selected
3. **Configuration Files**:
   - `configs/data/preprocessing.yaml` - Data config that uses preprocessing transforms
   - `configs/preset/datasets/preprocessing*.yaml` - Profile-specific presets

### How Profiles Work

When you select a profile in the Command Builder:

1. **Automatic Override Generation**: The UI automatically generates appropriate Hydra overrides
2. **Configuration Application**:
   - For training: `data=preprocessing` + appropriate preset
   - For inference/test: matching `data=preprocessing` + preset
3. **Pipeline Activation**: The preprocessing pipeline activates with the selected profile settings

### Profile-Specific Details

#### Lens-style Profile
- Uses document detection to locate document boundaries
- Applies perspective correction to rectify tilted documents
- Conservative enhancement to improve contrast

#### CamScanner Profile
- Uses LSD (Line Segment Detector) algorithm for document boundary detection
- More aggressive perspective correction
- Better for complex backgrounds

#### docTR Demo Profile
- Full docTR pipeline integration
- Orientation detection and correction
- Advanced geometry rectification

## Examples

### Training with Preprocessing

**Training with CamScanner preprocessing:**
```bash
# Through Command Builder UI - select "CamScanner document detection" profile
# Or manually:
uv run python runners/train.py \
  data=preprocessing \
  +preset/datasets=preprocessing_camscanner
```

### Inference with Matching Preprocessing (CORRECT)

```bash
# Through Command Builder UI - select SAME "CamScanner document detection" profile
# Or manually:
uv run python runners/predict.py \
  data=preprocessing \
  +preset/datasets=preprocessing_camscanner
```

### Inference WITHOUT Matching Preprocessing (WRONG)

```bash
# NEVER do this after training with preprocessing!
uv run python runners/predict.py \
  data=default
  # This will cause poor performance due to distribution shift!
```

## Configuration Options

### Profile Selection

Available profiles are configured in `configs/ui_meta/preprocessing_profiles.yaml`:

- `none`: Standard transforms without preprocessing
- `lens_style`: Balanced document detection and correction
- `lens_style_office`: Enhanced for low-quality documents
- `camscanner`: LSD-based document boundary detection
- `doctr_demo`: Full docTR pipeline integration

### Hydra Overrides

Each profile generates specific Hydra overrides:
- `data=preprocessing`: Base preprocessing configuration
- `+preset/datasets=preprocessing_{profile_id}`: Profile-specific settings

## Best Practices

### Training-Inference Consistency

1. **Document Your Profile**: Always record which preprocessing profile was used during training
2. **Use UI Selection**: Prefer Command Builder UI to ensure consistent profile application
3. **Validate Performance**: Compare preprocessed vs. non-preprocessed results to confirm benefits
4. **Monitor Distribution Shift**: Watch for symptoms like poor confidence calibration

### Profile Selection Guidelines

- **Clean Datasets**: Use `none` for baseline experiments
- **General Documents**: Use `lens_style` for most receipt/document OCR
- **Low Quality**: Use `lens_style_office` for faint or poor scans
- **Complex Backgrounds**: Use `camscanner` for tilted documents with complex backgrounds
- **Experimental**: Use `doctr_demo` for advanced geometry correction

## Troubleshooting

### Performance Degradation

**Problem**: Model performs poorly after training with preprocessing

**Solutions**:
- Verify inference uses the same preprocessing profile as training
- Check that Hydra overrides are correctly applied
- Compare with baseline (no preprocessing) to isolate issues

### Configuration Errors

**Problem**: Preprocessing not activating during training/inference

**Solutions**:
- Ensure `data=preprocessing` is set in configuration
- Verify correct preset is loaded (`+preset/datasets=preprocessing_*`)
- Check Command Builder UI profile selection

### Distribution Shift Symptoms

**Problem**: Poor confidence calibration or unexpected predictions

**Solutions**:
- Confirm training and inference preprocessing match exactly
- Test with `none` profile to establish baseline
- Validate preprocessing transforms are deterministic

### Profile Selection Issues

**Problem**: Unsure which profile to use for specific document types

**Solutions**:
- Start with `lens_style` for general use cases
- Use `camscanner` for documents with complex backgrounds
- Test multiple profiles on sample data to compare performance

## Related References

- **Preprocessing Config**: `configs/data/preprocessing.yaml`
- **Profile Presets**: `configs/preset/datasets/preprocessing*.yaml`
- **UI Metadata**: `configs/ui_meta/preprocessing_profiles.yaml`
- **Command Builder**: UI integration for profile selection
- **docTR Documentation**: Advanced document processing pipeline
- **CamScanner Algorithm**: LSD line detection for document boundaries
