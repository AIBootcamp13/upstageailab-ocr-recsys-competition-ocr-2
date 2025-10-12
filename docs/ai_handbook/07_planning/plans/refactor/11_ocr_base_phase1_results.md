# Phase 1: Preparation & Analysis - Baseline Results

## Performance Baseline
============================================================
IMAGE LOADING BENCHMARK
============================================================
pil_loading         : avg=0.0004s ± 0.0004s (min=0.0003s, max=0.0031s)
optimized_loading   : avg=0.0147s ± 0.0017s (min=0.0113s, max=0.0187s)

============================================================
TRANSFORM INTERPOLATION BENCHMARK
============================================================
linear_transform    : avg=0.0235s ± 0.0065s (min=0.0134s, max=0.0374s)
cubic_transform     : avg=0.0182s ± 0.0058s (min=0.0109s, max=0.0321s)

============================================================
FULL PIPELINE BENCHMARK
============================================================
full_pipeline       : avg=0.0536s ± 0.0112s (min=0.0400s, max=0.0957s)

============================================================
SPEEDUP ANALYSIS
============================================================
Image Loading Speedup: 0.03x
Transform Speedup (Cubic → Linear): 0.77x

## Test Suite Status
- **Total Tests**: 487
- **Passed**: 484
- **Failed**: 0
- **Skipped**: 2
- **XFailed**: 1
- **Test Coverage**: TBD (pytest-cov not available)

## API Usage Analysis

### OCRDataset (Legacy) Usage:
11 imports found in:
./debug/verification/verify_cache_implementation.py:    from ocr.datasets.base import OCRDataset
./debug/verification/test_load_maps_disabled.py:from ocr.datasets.base import OCRDataset
./tests/debug/data_analyzer.py:from ocr.datasets.base import OCRDataset
./tests/test_ocr_dataset_and_collate.py:from ocr.datasets.base import OCRDataset
./tests/unit/test_dataset.py:from ocr.datasets.base import OCRDataset

### ValidatedOCRDataset Usage:
19 production imports found in:
./ocr/datasets/__init__.py:from .base import ValidatedOCRDataset  # noqa: F401
./ocr/datasets/base.py:class ValidatedOCRDataset(Dataset):
./scripts/preprocess_data.py:Hydra-driven workflow. It instantiates a ``ValidatedOCRDataset`` directly from a
./scripts/preprocess_data.py:from ocr.datasets import ValidatedOCRDataset
./scripts/preprocess_data.py:def _materialise_dataset(dataset: ValidatedOCRDataset, *, limit: int | None) -> int:

## Dependency Analysis - ValidatedOCRDataset.__getitem__

### External Dependencies:
- **CacheManager**: get_cached_tensor(), get_hit_count(), get_miss_count(), set_cached_tensor(), get_cached_maps(), set_cached_maps()
- **Image Utils**: load_pil_image(), pil_to_numpy(), safe_get_image_size(), ensure_rgb(), prenormalize_imagenet()
- **Orientation Utils**: normalize_pil_image(), orientation_requires_rotation(), polygons_in_canonical_frame(), remap_polygons()
- **Polygon Utils**: ensure_polygon_array(), filter_degenerate_polygons(), validate_map_shapes()
- **Schemas**: DataItem, ImageData, ImageMetadata, PolygonData, TransformInput, MapData

### Method Flow:
1. **Cache Check**: CacheManager.get_cached_tensor()
2. **Image Loading**: _load_image_data() → image_utils functions
3. **Polygon Processing**: orientation_utils + polygon_utils
4. **Transformation**: TransformInput assembly
5. **Post-processing**: polygon_utils filtering
6. **Map Loading**: polygon_utils validation
7. **Final Assembly**: DataItem creation
8. **Cache Storage**: CacheManager.set_cached_tensor()

### Extraction Candidates:
- ✅ **CacheManager**: Already extracted (ocr/utils/cache_manager.py)
- 🔄 **Image Utils**: Partially extracted, needs consolidation
- 🔄 **Polygon Utils**: Partially extracted, needs consolidation
- 🔄 **Orientation Utils**: Already exist (ocr/utils/orientation.py)

## Success Criteria Status
- [x] All existing tests pass (baseline: 484/487)
- [x] Performance metrics documented
- [x] Complete dependency map created
- [ ] Test coverage > 90% for base.py (TBD)

## Next Steps
Phase 1 complete. Ready to proceed to Phase 2: CacheManager Extraction (already done) or Phase 3: Image Utilities Extraction.
