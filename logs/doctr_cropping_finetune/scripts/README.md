# Testing Scripts

This folder contains scripts for testing and fine-tuning the docTR document detection parameters.

## Scripts Overview

### `baseline_test.py`
Tests the current default document detection parameters on a set of images.

**Usage:**
```bash
python scripts/baseline_test.py --images /path/to/images --output /path/to/output
```

**Parameters:**
- `--images`: Directory containing test images
- `--output`: Output directory for results (default: `./parameter_tests/baseline`)

**Output:**
- Individual overlay images showing detected document boundaries
- `results.json`: Detailed results for each image
- Console summary of detection success rate

### `parameter_sweep.py`
Tests multiple combinations of document detection parameters to find optimal settings.

**Usage:**
```bash
python scripts/parameter_sweep.py --images /path/to/images --output /path/to/output --max-images 10
```

**Parameters:**
- `--images`: Directory containing test images
- `--output`: Output directory for results (default: `./parameter_tests/sweep`)
- `--max-images`: Limit number of images for quick testing (optional)

**Tested Parameters:**
- `min_area_ratio`: [0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30]
- Method combinations:
  - Canny only
  - Canny + adaptive thresholding
  - All methods except CamScanner
  - CamScanner only
  - All methods

**Output:**
- Overlay images for each parameter combination on each image
- `sweep_results.json`: Detailed results for all combinations
- `sweep_summary.json`: Summary statistics per parameter set
- Console ranking of top 5 parameter configurations

## File Naming Convention

Overlay images are named as: `{original_name}_{params}_overlay.png`

Where `{params}` encodes the parameters used:
- `area{ratio}`: Area ratio threshold
- `adapt{0|1}`: Adaptive thresholding enabled
- `fb{0|1}`: Fallback bounding box enabled
- `cam{0|1}`: CamScanner LSD detection enabled

Example: `receipt_area0.15_adapt1_fb1_cam0_overlay.png`

## Running Tests

1. Place test images in `problematic_images/` folder
2. Run baseline test to establish current performance
3. Run parameter sweep to find better configurations
4. Compare overlay images visually to assess improvements
5. Update configuration based on best performing parameters
