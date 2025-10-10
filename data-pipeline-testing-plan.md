
# Data Pipeline Testing Implementation Roadmap

**Objective**: Comprehensive testing implementation to prevent BUG-2025-004 style catastrophic failures through systematic validation of data shape contracts and type safety across the OCR pipeline.

**Timeline**: 4-week implementation across 2 sprints (2 weeks each)
**Total Effort**: ~80 developer hours
**Risk Level**: Medium (incremental changes with rollback plans)

---

## 📅 Phase 1: Critical Path Fixes (Week 1-2) 🔴 HIGH PRIORITY

**Focus**: Address immediate production risks that could cause silent training failures or crashes.

### 1.1 Collate Function Polygon Shape Handling
**Location**: `ocr/datasets/db_collate_fn.py`, `ocr/datasets/craft_collate_fn.py`
**Effort**: 12 hours
**Dependencies**: None
**Risk**: Medium (affects batch processing)

#### Tasks:
- [ ] **Task 1.1.1**: Create `test_db_collate_polygon_shapes.py` (2h)
  - Test polygon normalization: `(N, 2)` → `(1, N, 2)` conversion
  - Test mixed batch inputs with different polygon shapes
  - Acceptance: All shape conversions work correctly

- [ ] **Task 1.1.2**: Add polygon validation in collate functions (3h)
  - Add `_validate_batch_polygons()` method to `DBCollateFN`
  - Handle edge cases: empty polygons, single-point polygons
  - Acceptance: Invalid polygons logged and skipped, valid ones processed

- [ ] **Task 1.1.3**: Integration test collate → model forward pass (4h)
  - Create `test_collate_integration.py` with real batch data
  - Test end-to-end: dataset → collate → model → loss computation
  - Acceptance: No crashes, loss values computed correctly

- [ ] **Task 1.1.4**: Documentation update (1h)
  - Update docstrings in collate functions with shape contracts
  - Add shape validation comments
  - Acceptance: Clear documentation of expected input/output shapes

#### Success Metrics:
- ✅ All polygon shapes normalized correctly in batches
- ✅ No crashes during batch processing
- ✅ Integration tests pass with real data
- ✅ Performance regression < 5% (benchmark vs baseline)

#### Rollback Plan:
- Revert collate function changes
- Use original unvalidated version with warning logs

---

### 1.2 Image Type Confusion (PIL vs NumPy)
**Location**: `ocr/datasets/base.py` (lines 325, 174, 278)
**Effort**: 10 hours
**Dependencies**: None
**Risk**: High (caused validation crashes in production)

#### Tasks:
- [ ] **Task 1.2.1**: Create `test_image_type_handling.py` (2h)
  - Test PIL Image `.size` vs NumPy `.size` behavior
  - Test canonical size extraction for both types
  - Acceptance: Correct size extraction regardless of image type

- [ ] **Task 1.2.2**: Add type guards in image loading pipeline (3h)
  - Implement `safe_get_image_size()` function in `base.py`
  - Update lines 325, 174, 278 to use type-safe size extraction
  - Acceptance: No AttributeError on `.size` access

- [ ] **Task 1.2.3**: Test PIL → NumPy conversion metadata preservation (3h)
  - Test that image metadata survives type conversion
  - Test orientation and EXIF data handling
  - Acceptance: All metadata preserved through pipeline

- [ ] **Task 1.2.4**: Integration test with different image sources (2h)
  - Test images from different loaders (PIL, OpenCV, cached)
  - Test canonical vs regular image paths
  - Acceptance: Consistent behavior across all image sources

#### Success Metrics:
- ✅ No type confusion errors in image processing
- ✅ Canonical size extraction works for all image types
- ✅ Metadata preservation verified
- ✅ Performance impact < 2%

#### Rollback Plan:
- Revert to original size extraction logic
- Add try/catch blocks as temporary mitigation

---

### 1.3 Polygon Filtering Logic
**Location**: `ocr/datasets/base.py` (lines 470-520)
**Effort**: 8 hours
**Dependencies**: Task 1.2 (image type handling)
**Risk**: Medium (causes silent training failures)

#### Tasks:
- [ ] **Task 1.3.1**: Create `test_polygon_filtering.py` (2h)
  - Test filtering with different `min_side` thresholds
  - Test edge case: images with all polygons filtered
  - Acceptance: Filtering behavior predictable and configurable

- [ ] **Task 1.3.2**: Add filtering metrics and logging (3h)
  - Add polygon count tracking before/after filtering
  - Log filtering statistics per image
  - Acceptance: Clear visibility into filtering impact

- [ ] **Task 1.3.3**: Parameterize filtering thresholds (2h)
  - Make `min_side` configurable via dataset config
  - Test different thresholds on sample data
  - Acceptance: Configurable filtering without code changes

- [ ] **Task 1.3.4**: Regression test for filtering impact (1h)
  - Add assertion: minimum polygons per epoch > 0
  - Test with known good dataset samples
  - Acceptance: No false positive filtering failures

#### Success Metrics:
- ✅ Filtering behavior configurable and predictable
- ✅ Clear metrics on filtering impact
- ✅ No silent polygon removal in training
- ✅ Dataset size stability (±10% with same config)

#### Rollback Plan:
- Revert threshold changes to original values
- Keep metrics/logging as they add visibility without risk

---

## 📅 Phase 2: Pipeline Integration (Week 3-4) 🟡 MEDIUM PRIORITY

**Focus**: Ensure data contracts are maintained across pipeline stages.

### 2.1 Transform Pipeline Data Contracts
**Location**: `ocr/datasets/transforms.py`, Albumentations integration
**Effort**: 12 hours
**Dependencies**: Phase 1 complete
**Risk**: Low (defensive changes)

#### Tasks:
- [ ] **Task 2.1.1**: Create `test_transform_pipeline_contracts.py` (3h)
  - Test keypoint/polygon preservation through transforms
  - Test inverse transform matrix accuracy
  - Acceptance: Geometric transformations mathematically correct

- [ ] **Task 2.1.2**: Add shape contract validation (4h)
  - Implement `_validate_transform_contracts()` in `DBTransforms`
  - Test input/output shape consistency
  - Acceptance: Clear error messages for contract violations

- [ ] **Task 2.1.3**: End-to-end transform testing (4h)
  - Test full pipeline: load → transform → collate → model
  - Test with real dataset samples
  - Acceptance: No shape-related crashes in full pipeline

- [ ] **Task 2.1.4**: Performance regression testing (1h)
  - Benchmark transform speed before/after changes
  - Ensure < 5% performance impact
  - Acceptance: Transform performance maintained

#### Success Metrics:
- ✅ All transform contracts validated
- ✅ End-to-end pipeline works with real data
- ✅ No geometric transformation errors
- ✅ Performance regression < 5%

#### Rollback Plan:
- Revert contract validation (keep as optional debug mode)
- Use original transform logic

---

### 2.2 Map Loading and Caching
**Location**: `ocr/datasets/base.py`, `scripts/preprocess_maps.py`
**Effort**: 10 hours
**Dependencies**: Phase 1 complete
**Risk**: Low (affects caching, not core functionality)

#### Tasks:
- [ ] **Task 2.2.1**: Create `test_map_loading_caching.py` (3h)
  - Test loading corrupted/missing `.npz` files
  - Test fallback map generation
  - Acceptance: Graceful handling of map loading failures

- [ ] **Task 2.2.2**: Add map integrity validation (3h)
  - Implement `_validate_map_shapes()` in dataset loader
  - Check map dimensions match image dimensions
  - Acceptance: Invalid maps detected and regenerated

- [ ] **Task 2.2.3**: Test cached vs generated map consistency (3h)
  - Compare cached maps vs on-the-fly generation
  - Test numerical accuracy of cached data
  - Acceptance: < 1% difference between cached and generated

- [ ] **Task 2.2.4**: Update map preprocessing script (1h)
  - Add validation to `preprocess_maps.py`
  - Document map format requirements
  - Acceptance: Clear map format specifications

#### Success Metrics:
- ✅ Map loading robust to corruption
- ✅ Cached/generated maps numerically equivalent
- ✅ Clear error messages for map issues
- ✅ Caching performance maintained

#### Rollback Plan:
- Disable caching temporarily
- Use on-the-fly generation as fallback

---

## 📅 Phase 3: Advanced Validation (Ongoing) 🟠 LOW PRIORITY

**Focus**: Comprehensive validation for long-term stability.

### 3.1 Post-processing Shape Handling
**Location**: `ocr/models/head/db_postprocess.py`
**Effort**: 8 hours
**Dependencies**: Phase 2 complete
**Risk**: Low (defensive validation)

#### Tasks:
- [ ] **Task 3.1.1**: Create `test_postprocessing_shapes.py` (2h)
  - Test with different batch sizes and tensor shapes
  - Test edge cases: empty batches, malformed inputs
  - Acceptance: Robust shape handling in post-processing

- [ ] **Task 3.1.2**: Add tensor shape validation (3h)
  - Implement shape checks in post-processing functions
  - Add clear error messages for shape mismatches
  - Acceptance: Early detection of shape issues

- [ ] **Task 3.1.3**: Integration test model → post-processing (3h)
  - Test full pipeline: model output → post-processing → predictions
  - Test with various input shapes
  - Acceptance: Consistent prediction output formats

#### Success Metrics:
- ✅ Post-processing handles all valid input shapes
- ✅ Clear error messages for invalid inputs
- ✅ Prediction format consistency
- ✅ No runtime crashes in post-processing

---

### 3.2 Dataset Iterator and Batching
**Location**: PyTorch DataLoader integration
**Effort**: 6 hours
**Dependencies**: Phase 2 complete
**Risk**: Low (infrastructure testing)

#### Tasks:
- [x] **Task 3.2.1**: Create `test_dataloader_batching.py` (2h)
  - Test variable polygon counts per image
  - Test batch collation with empty images
  - Acceptance: Robust batching for all data variations

- [x] **Task 3.2.2**: Memory efficiency testing (2h)
  - Test memory usage with large batches
  - Profile DataLoader performance
  - Acceptance: Memory usage scales appropriately

- [x] **Task 3.2.3**: Stress testing (2h)
  - Test maximum batch size limits
  - Test with extreme data variations
  - Acceptance: System handles edge cases gracefully

#### Success Metrics:
- ✅ DataLoader handles all data variations
- ✅ Memory usage efficient and predictable
- ✅ No crashes under stress conditions
- ✅ Performance scales with batch size

---

## 📊 Progress Tracking & Milestones

### Daily Progress Checklist Template:
```markdown
## Day X: [Phase.Task] - [Task Description]
- [ ] Code changes implemented
- [ ] Unit tests written and passing
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] Performance benchmark completed
- [ ] Peer review completed
- [ ] Merged to main branch
```

### Sprint Milestones:
- **End of Sprint 1**: Phases 1.1-1.3 complete, critical path secured
- **End of Sprint 2**: Phases 2.1-2.2 complete, pipeline integration tested
- **End of Phase 3**: Full validation suite operational

### Dependencies Map:
```
Phase 1.1 (Collate) → Phase 1.2 (Images) → Phase 1.3 (Filtering)
                                      ↓
Phase 2.1 (Transforms) ←──────────────┘
Phase 2.2 (Maps) ←─────────────────────┘
                                      ↓
Phase 3.1 (Post-processing) ←─────────┘
Phase 3.2 (DataLoader) ←──────────────┘
```

---

## 🧪 Testing Artifacts Created

### Test Files:
- `tests/ocr/datasets/test_db_collate_polygon_shapes.py`
- `tests/ocr/datasets/test_image_type_handling.py`
- `tests/ocr/datasets/test_polygon_filtering.py`
- `tests/ocr/datasets/test_transform_pipeline_contracts.py`
- `tests/ocr/datasets/test_map_loading_caching.py`
- `tests/ocr/models/test_postprocessing_shapes.py`
- `tests/integration/test_dataloader_batching.py`
- `tests/integration/test_collate_integration.py`

### Scripts & Tools:
- `scripts/validate_pipeline_contracts.py` - Pipeline validation utility
- `scripts/benchmark_pipeline_performance.py` - Performance regression testing
- `scripts/generate_test_data_samples.py` - Test data generation

### Documentation Updates:
- `docs/pipeline/data_contracts.md` - Data shape contracts documentation
- `docs/testing/pipeline_validation.md` - Testing strategy guide
- `docs/troubleshooting/shape_issues.md` - Shape-related troubleshooting

---

## 📈 Performance Benchmarks

### Baseline Metrics (to maintain):
- Training hmean: ≥ 0.8 (vs working commit 8252600)
- Transform throughput: ≥ 100 images/sec
- Memory usage: < 8GB per GPU during training
- Map loading time: < 50ms per batch

### Regression Tests:
- Run full training epoch after each phase
- Compare metrics against baseline
- Alert if performance drops > 5%

---

## 🚨 Risk Mitigation

### High-Risk Changes:
1. **Image type handling** (Phase 1.2): Could affect image loading performance
   - **Mitigation**: Feature flag to enable/disable new logic
   - **Rollback**: Revert to original with try/catch

2. **Polygon filtering** (Phase 1.3): Could change dataset size
   - **Mitigation**: Configurable thresholds, metrics logging
   - **Rollback**: Revert thresholds to original values

### Monitoring:
- Daily performance regression tests
- Error rate monitoring in training logs
- Data quality metrics (polygons per image, etc.)

### Contingency Plans:
- **Phase failure**: Skip to next phase, address failed phase in next sprint
- **Performance regression**: Rollback changes, investigate root cause
- **Data quality issues**: Use original data loading logic as fallback

---

## ✅ Success Criteria

### Phase 1 (Critical Path):
- [ ] No crashes in training pipeline
- [ ] All shape-related errors have clear messages
- [ ] Performance regression < 5%
- [ ] Training hmean ≥ 0.8 maintained

### Phase 2 (Integration):
- [ ] End-to-end pipeline tests passing
- [ ] Data contracts documented and validated
- [ ] All integration points tested
- [ ] Caching performance maintained

### Phase 3 (Advanced):
- [ ] Comprehensive test coverage for edge cases
- [ ] Robust error handling throughout pipeline
- [ ] Performance benchmarks established
- [ ] Documentation complete and accurate

### Overall Project:
- [ ] Zero BUG-2025-004 style failures in production
- [ ] Clear debugging path for future shape issues
- [ ] Maintainable test suite for ongoing validation
- [ ] Performance and accuracy metrics stable

---

**Implementation Start**: October 11, 2025
**Phase 1 Deadline**: October 18, 2025
**Phase 2 Deadline**: October 25, 2025
**Phase 3 Deadline**: November 1, 2025

**Total Effort**: 66 hours across 3 weeks
**Daily Target**: 3-4 hours of focused implementation work

Made changes.
