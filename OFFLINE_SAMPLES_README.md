# Offline Sample Generation

This directory contains tools for generating offline samples using the Microsoft Lens-style preprocessing pipeline. These samples are useful for:

- **Demonstrating preprocessing effects** on real OCR data
- **Testing preprocessing configurations** before training
- **Benchmarking performance improvements** from preprocessing
- **Visualizing before/after comparisons** for documentation

## Generated Samples

The sample generation creates three types of outputs:

### 1. Original Images (`original/`)
- Unprocessed images from the dataset
- Named as `sample_XXX_filename_original.jpg`

### 2. Processed Images (`processed/`)
- Images after applying the preprocessing pipeline
- Named as `sample_XXX_filename_processed.jpg`
- Includes metadata JSON files with processing details

### 3. Comparison Visualizations (`comparison/`)
- Side-by-side before/after comparisons
- Named as `sample_XXX_filename_comparison.jpg`
- Includes processing information overlay

## Usage

### Generate Samples

```bash
# Generate 10 samples with full preprocessing pipeline
python generate_offline_samples.py --num-samples 10

# Generate samples with custom output directory
python generate_offline_samples.py --num-samples 5 --output-dir outputs/my_samples

# Generate samples with selective preprocessing disabled
python generate_offline_samples.py --num-samples 5 \
    --no-document-detection \
    --no-perspective-correction \
    --output-dir outputs/enhancement_only
```

### View Sample Statistics

```bash
# Display generation statistics and configuration
python demo_preprocessing_samples.py --mode stats
```

### Visualize Samples

```bash
# Display grid of sample comparisons
python demo_preprocessing_samples.py --mode grid --num-samples 6

# Display single sample with detailed information
python demo_preprocessing_samples.py --mode single --sample-idx 0
```

## Command Line Options

### Sample Generation (`generate_offline_samples.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--source-dir` | `data/datasets/images` | Directory containing source images |
| `--output-dir` | `outputs/samples` | Directory to save processed samples |
| `--num-samples` | `10` | Number of samples to generate |
| `--no-document-detection` | `False` | Disable document boundary detection |
| `--no-perspective-correction` | `False` | Disable perspective correction |
| `--no-enhancement` | `False` | Disable image enhancement |
| `--no-text-enhancement` | `False` | Disable text-specific enhancement |

### Sample Visualization (`demo_preprocessing_samples.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--sample-dir` | `outputs/samples` | Directory containing generated samples |
| `--mode` | `grid` | Display mode: `single`, `grid`, or `stats` |
| `--sample-idx` | `0` | Sample index for single mode |
| `--num-samples` | `5` | Number of samples to display in grid mode |
| `--cols` | `3` | Number of columns in grid mode |

## Preprocessing Pipeline

The Microsoft Lens-style preprocessing includes:

### 1. Document Detection
- Edge detection using Canny algorithm
- Contour analysis to find document boundaries
- Quadrilateral detection and corner extraction

### 2. Perspective Correction
- Homography-based perspective correction
- 4-point homography estimation from detected corners
- Automatic document straightening

### 3. Image Enhancement
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Bilateral filtering**: Noise reduction while preserving edges
- **Unsharp masking**: Image sharpening for better detail

### 4. Text Enhancement
- Morphological operations for text region enhancement
- Adaptive thresholding for text binarization
- Alpha blending for natural appearance

## Sample Output Structure

```
outputs/samples/
├── original/
│   ├── sample_000_filename_original.jpg
│   ├── sample_001_filename_original.jpg
│   └── ...
├── processed/
│   ├── sample_000_filename_processed.jpg
│   ├── sample_000_filename_metadata.json
│   ├── sample_001_filename_processed.jpg
│   ├── sample_001_filename_metadata.json
│   └── ...
├── comparison/
│   ├── sample_000_filename_comparison.jpg
│   ├── sample_001_filename_comparison.jpg
│   └── ...
└── generation_report.json
```

## Metadata Information

Each processed sample includes detailed metadata:

```json
{
  "original_shape": [1280, 960, 3],
  "processing_steps": [
    "document_detection",
    "perspective_correction",
    "image_enhancement",
    "text_enhancement"
  ],
  "document_corners": "[[370 132] [681 143] [664 1255] [352 1236]]",
  "perspective_matrix": "...homography matrix...",
  "enhancement_applied": [
    "clahe",
    "bilateral_filter",
    "unsharp_masking"
  ],
  "final_shape": [640, 640, 3]
}
```

## Example Workflows

### Quick Demo
```bash
# Generate 5 samples
python generate_offline_samples.py --num-samples 5

# View statistics
python demo_preprocessing_samples.py --mode stats

# View comparisons
python demo_preprocessing_samples.py --mode grid
```

### Ablation Study
```bash
# Full preprocessing
python generate_offline_samples.py --num-samples 10 --output-dir outputs/full_preprocessing

# Enhancement only
python generate_offline_samples.py --num-samples 10 \
    --no-document-detection \
    --no-perspective-correction \
    --output-dir outputs/enhancement_only

# Compare results
python demo_preprocessing_samples.py --sample-dir outputs/full_preprocessing --mode stats
python demo_preprocessing_samples.py --sample-dir outputs/enhancement_only --mode stats
```

### Integration Testing
```bash
# Generate samples for integration testing
python generate_offline_samples.py --num-samples 20 --output-dir outputs/integration_test

# Use in training validation
# (Modify your training scripts to load from outputs/integration_test/processed/)
```

## Performance Notes

- **Processing Speed**: ~2-3 samples/second on typical hardware
- **Memory Usage**: Minimal (processes images individually)
- **Success Rate**: Typically 95%+ on well-formed document images
- **Document Detection**: May fail on complex backgrounds or irregular documents

## Troubleshooting

### No Images Found
- Ensure `data/datasets/images/` contains image files
- Check file extensions (supports jpg, jpeg, png, bmp, tiff)

### Document Detection Failing
- Many images may not have clear document boundaries
- This is expected behavior - preprocessing gracefully continues
- Check metadata to see which steps were applied

### Memory Issues
- Reduce `--num-samples` for limited memory systems
- Images are processed individually, not loaded all at once

### Visualization Issues
- Ensure matplotlib and PIL/Pillow are installed
- For headless systems, save plots instead of displaying:
  ```python
  plt.savefig('comparison.png')  # Instead of plt.show()
  ```