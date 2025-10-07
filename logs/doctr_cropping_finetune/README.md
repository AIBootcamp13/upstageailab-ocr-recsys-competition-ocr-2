# docTR Cropping Fine-Tuning Session

## Session Overview
This session focuses on fine-tuning the docTR preprocessing image cropping function, specifically the document boundary detection that produces the translucent green overlay in the Streamlit inference UI.

## Problem Statement
The green box overlay that previews the detected document boundaries performs poorly in certain cases, with some instances being outright unacceptable. The user has problematic images that need iterative testing and fine-tuning.

## Current Implementation Analysis

### Detection Pipeline
The document detection uses a multi-stage fallback approach in `DocumentDetector`:

1. **CamScanner method** (if enabled): Uses LSD (Line Segment Detector) for advanced line detection
2. **Canny edge detection**: Standard edge detection with contour analysis
3. **Adaptive thresholding** (if enabled): Fallback using adaptive threshold
4. **Bounding box fallback** (if enabled): Uses largest foreground bounding box

### Configuration Knobs
Key parameters that can be adjusted:

| Parameter | Current Default | Purpose |
|-----------|----------------|---------|
| `document_detection_min_area_ratio` | 0.18 | Minimum area ratio for valid document detection |
| `document_detection_use_adaptive` | true | Enable adaptive thresholding fallback |
| `document_detection_use_fallback_box` | true | Enable bounding box fallback |
| `document_detection_use_camscanner` | false | Enable CamScanner-style LSD detection |

### Visual Overlay
The green overlay is drawn in `ui/apps/inference/components/results.py`:
- Solid green outline: `(0, 255, 0, 255)`
- Translucent green fill: `(0, 255, 0, 40)`
- Green center dot: `(0, 128, 0, 255)`

## Assessment of Current Issues

### Potential Problems
1. **Detection accuracy**: The contour-based detection may fail on complex backgrounds or irregular document shapes
2. **Parameter interactions**: Multiple fallback methods may produce inconsistent results
3. **Area ratio threshold**: 0.18 may be too restrictive for small receipts or too permissive for noise
4. **Method selection**: The priority order of detection methods may not suit all image types

### Validation Challenges
- Difficult to quantify "poor performance" numerically
- Visual inspection required for each iteration
- Need systematic testing across problematic image set

## Fine-Tuning Plan

### Phase 1: Baseline Establishment
1. Collect and categorize problematic images
2. Establish current performance baseline with default settings
3. Document specific failure modes

### Phase 2: Parameter Optimization
1. **Area Ratio Tuning**: Test range 0.05-0.3 in 0.05 increments
2. **Method Ablation**: Test combinations of detection methods
3. **CamScanner Integration**: Evaluate LSD-based detection for complex cases

### Phase 3: Iterative Testing
1. Implement quick testing script for batch processing
2. Create side-by-side comparison interface
3. Track improvements quantitatively where possible

### Phase 4: UI Integration (Optional)
1. Add fine-tuning controls to Streamlit interface
2. Implement real-time parameter adjustment
3. Add visual debugging overlays

## Success Criteria
- Significant improvement in green overlay accuracy on problematic images
- Consistent detection across different document types
- Minimal false positives/negatives
- Maintainable parameter set

## Session Structure
- `problematic_images/`: Test images that demonstrate issues
- `parameter_tests/`: Results from different parameter combinations
- `scripts/`: Testing and analysis utilities
- `logs/`: Session progress and findings
- `ui_integration/`: Optional Streamlit enhancements

## Next Steps
1. Collect problematic images
2. Create baseline testing script
3. Begin parameter sweep on area ratio</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/doctr_cropping_finetune/README.md
