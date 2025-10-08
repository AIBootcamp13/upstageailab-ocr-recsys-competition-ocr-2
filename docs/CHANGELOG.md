# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - 2025-10-09

#### Data Pipeline Performance Optimization

**Major Refactoring: Offline Pre-processing System**

Replaced on-the-fly polygon caching with an offline pre-processing system that generates probability and threshold maps once and loads them during training.

**Performance Impact:**
- 5-8x faster validation epochs
- Eliminated polygon cache key collision issues
- Reduced memory overhead during training
- Simplified collate function logic

**New Components:**

1. **Pre-processing Script** (`scripts/preprocess_maps.py`)
   - Generates and saves `.npz` files containing probability and threshold maps
   - Supports Hydra configuration for consistency with training pipeline
   - Includes sanity checks and validation
   - Filters degenerate polygons to ensure stable pyclipper operations

2. **Enhanced Dataset** (`ocr/datasets/base.py`)
   - Modified `OCRDataset.__getitem__` to load pre-processed maps from `.npz` files
   - Automatic fallback to on-the-fly generation if maps are missing
   - Maintains backward compatibility

3. **Simplified Collate Function** (`ocr/datasets/db_collate_fn.py`)
   - Removed polygon cache logic
   - Now primarily stacks pre-loaded maps into batches
   - Fallback to on-the-fly generation when needed
   - Removed unused `cache` parameter

4. **Documentation**
   - Added comprehensive [preprocessing guide](docs/preprocessing_guide.md)
   - Updated [README.md](README.md) with preprocessing workflow
   - Includes troubleshooting and maintenance instructions

**Configuration Changes:**

- Removed `polygon_cache` section from `configs/data/base.yaml`
- Removed `cache` parameter from `collate_fn` configuration
- Cleaned up cache-related test configurations
- Deleted obsolete `configs/data/cache.yaml`

**Code Cleanup:**

- Removed `ocr/datasets/polygon_cache.py` (obsolete caching implementation)
- Removed `tests/performance/test_polygon_caching.py` (obsolete tests)
- Removed cache instantiation from `ocr/lightning_modules/ocr_pl.py`

**Migration Guide:**

To migrate existing projects to use pre-processing:

1. Run the pre-processing script:
   ```bash
   uv run python scripts/preprocess_maps.py
   ```

2. Verify output directories are created:
   - `data/datasets/images/train_maps/`
   - `data/datasets/images_val_canonical_maps/`

3. Training will automatically use pre-processed maps when available

4. To regenerate maps after dataset/config changes, simply re-run the script

**Technical Details:**

- Map files are saved as compressed `.npz` format (~50-100 MB per 1000 samples)
- Each `.npz` file contains `prob_map` and `thresh_map` arrays with shape `(1, H, W)`
- Maps are generated using the same DBNet algorithm as before (shrink_ratio=0.4)
- Degenerate polygon filtering prevents pyclipper crashes

**Related Files:**
- Implementation Plan: `logs/2025-10-08_02_refactor_performance_features/description/polygon-preprocessing-implementation-plan.md`
- Unit Tests: `tests/test_preprocess_maps.py`

### Changed - 2025-10-09

- **`ocr/datasets/base.py`**: Added map loading logic to `__getitem__` method
- **`ocr/datasets/db_collate_fn.py`**: Simplified to use pre-loaded maps, removed caching
- **`ocr/lightning_modules/ocr_pl.py`**: Removed polygon cache instantiation from `_build_collate_fn`
- **`configs/data/base.yaml`**: Removed polygon_cache configuration
- **`configs/performance_test.yaml`**: Removed polygon_cache test flag
- **`configs/cache_performance_test.yaml`**: Removed polygon_cache test flag

### Removed - 2025-10-09

- **`ocr/datasets/polygon_cache.py`**: Obsolete caching implementation
- **`tests/performance/test_polygon_caching.py`**: Obsolete cache performance tests
- **`configs/data/cache.yaml`**: Obsolete cache configuration file

## [0.1.0] - 2025-09-23

### Added
- Initial project structure
- DBNet baseline implementation
- PyTorch Lightning training pipeline
- Hydra configuration management
- CLEval metric integration
- Basic data loading and augmentation
- Command builder UI
- Evaluation viewer UI
- Process monitor utility
- Comprehensive test suite

### Documentation
- Architecture overview
- API reference
- Coding standards
- Testing guide
- Process management guide

---

## Release Notes

### Version 0.1.0 (Baseline)
- First stable release
- Competition-ready baseline with H-mean 0.8818 on public test set
- Complete training/validation/inference pipeline
- UI tools for experiment management

### Upcoming Features
- [x] Offline preprocessing system (Phase 1-3 complete)
- [ ] Parallelized preprocessing (Phase 5)
- [ ] WebDataset or RAM caching (Phase 6)
- [ ] NVIDIA DALI integration (Phase 7)
