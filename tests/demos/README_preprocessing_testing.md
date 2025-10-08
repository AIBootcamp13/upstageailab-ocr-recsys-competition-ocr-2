# Preprocessing Testing Framework

This directory contains tools for systematically testing and analyzing the OCR preprocessing features, particularly focusing on doctr rcrop geometry and camscanner-style detection.

## Overview

The testing framework consists of two main scripts:

1. `test_preprocessing_systematic.py` - Runs comprehensive tests on preprocessing features
2. `analyze_preprocessing_results.py` - Analyzes and compares test results

## Features Tested

### doctr rcrop geometry
- **With bounding-box fallback**: Tests doctr rcrop with fallback enabled
- **Without bounding-box fallback**: Tests doctr rcrop with fallback disabled
- **With orientation correction**: Tests rcrop combined with orientation correction
- **Logging**: All values (corners, angles, processing times) are logged to terminal

### camscanner-style detection
- **Effectiveness testing**: Tests LSD-based line detection for document boundary detection
- **Cropping validation**: Verifies that detected documents are properly cropped

### Orientation correction
- **Angle detection**: Tests docTR-based angle estimation
- **Correction application**: Tests rotation correction with canvas expansion
- **Redetection**: Tests corner redetection after rotation

## Usage

### 1. Running Tests

```bash
# Basic usage - test with default settings
python tests/demos/test_preprocessing_systematic.py --image-dir /path/to/test/images --output-dir results/

# Advanced usage with custom settings
python tests/demos/test_preprocessing_systematic.py \
    --image-dir data/datasets/images/test/ \
    --output-dir preprocessing_test_results/ \
    --max-samples 20 \
    --log-level DEBUG
```

**Parameters:**
- `--image-dir`: Directory containing test images (supports jpg, png, bmp, tiff)
- `--output-dir`: Where to save results and generated images
- `--max-samples`: Maximum number of images to test (default: 10)
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

### 2. Analyzing Results

```bash
# Show summary report
python tests/demos/analyze_preprocessing_results.py --results-dir preprocessing_test_results/ --summary

# Create performance comparison chart
python tests/demos/analyze_preprocessing_results.py --results-dir preprocessing_test_results/ --chart performance_chart.png

# Export detailed analysis report
python tests/demos/analyze_preprocessing_results.py --results-dir preprocessing_test_results/ --export-report detailed_analysis.json

# Show details for specific sample
python tests/demos/analyze_preprocessing_results.py --results-dir preprocessing_test_results/ --sample-details 0
```

## Test Configurations

The framework tests 5 different preprocessing configurations:

1. **doctr_rcrop_with_fallback**: docTR rcrop with bounding-box fallback
2. **doctr_rcrop_without_fallback**: docTR rcrop without fallback
3. **doctr_rcrop_with_orientation**: docTR rcrop + orientation correction
4. **camscanner_detection**: CamScanner-style LSD line detection
5. **opencv_baseline**: OpenCV-only processing (no docTR)

## Output Structure

```
preprocessing_test_results/
├── preprocessing_test.log          # Detailed execution log
├── consolidated_results.json       # All test results
├── comparison_report.json          # Cross-configuration comparison
├── processed_images/               # Processed result images
│   ├── sample_000_doctr_rcrop_with_fallback_processed.jpg
│   └── ...
├── overlays/                       # Detection overlay images
│   ├── sample_000_doctr_rcrop_with_fallback_overlay.jpg
│   └── ...
└── visual_comparison_sample_0.png # Side-by-side comparisons
```

## Result Analysis

### Key Metrics Tracked

- **Detection Success Rate**: Percentage of images where document boundaries were detected
- **Processing Time**: Average time per image for each configuration
- **Corner Coordinates**: Exact detected corner positions
- **Detection Methods**: Which algorithm succeeded (camscanner, canny_contour, adaptive_threshold, bounding_box)
- **Orientation Corrections**: Angle corrections applied and redetection success

### Visual Analysis

- **Overlay Images**: Original images with detected corners and edges drawn
- **Processed Images**: Final cropped and enhanced results
- **Comparison Charts**: Performance metrics across configurations

### Terminal Logging

During execution, the framework logs:
- Detection method used for each image
- Processing time per image
- Corner coordinates when detected
- Orientation correction angles
- Success/failure status

## Troubleshooting

### Common Issues

1. **No docTR features available**
   - Install python-doctr: `pip install python-doctr[torch]`
   - Some features will fall back to OpenCV-only processing

2. **Low detection success rates**
   - Check image quality - preprocessing works best with clear document photos
   - Try different configurations (camscanner often works better for receipts)

3. **Slow processing**
   - docTR features are slower but more accurate
   - Use OpenCV baseline for faster processing

### Manual Validation

For each test image, you should:
1. **Visually inspect overlays** - Are detected corners accurate?
2. **Check processed images** - Is the document properly cropped?
3. **Review terminal logs** - Are corner coordinates reasonable?
4. **Compare configurations** - Which method works best for your images?

## Integration with Existing Code

The testing framework uses the same `DocumentPreprocessor` class used in production:

```python
from ocr.datasets.preprocessing.pipeline import DocumentPreprocessor

# Create preprocessor with test configuration
preprocessor = DocumentPreprocessor(
    use_doctr_geometry=True,
    document_detection_use_fallback_box=True,
    enable_orientation_correction=True,
    document_detection_use_camscanner=False
)

# Process single image
result = preprocessor(image)
processed_image = result["image"]
metadata = result["metadata"]
```

## Extending the Framework

### Adding New Test Configurations

Edit the `test_configs` list in `PreprocessingTester.__init__()`:

```python
self.test_configs.append({
    "name": "my_custom_config",
    "use_doctr_geometry": False,
    "document_detection_use_fallback_box": False,
    "enable_orientation_correction": True,
    "document_detection_use_camscanner": True,
})
```

### Adding New Metrics

Modify `test_preprocessing_config()` to track additional metadata from the preprocessing pipeline.
