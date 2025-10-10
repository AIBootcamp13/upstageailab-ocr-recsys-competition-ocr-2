# Comprehensive Pipeline Debug Analysis
**Date**: 2025-10-10
**Session**: Pipeline Debug & Fix

---

## Executive Summary

The pipeline has **4 unit test failures** and **1 test error**. Analysis shows:

1. **BUG-2025-002** (PIL Image crash) - ✅ **ALREADY FIXED** according to bug reports
2. **BUG-2025-003** (Albumentations contract) - ✅ **ALREADY FIXED** according to bug reports
3. **New Issues** - ❌ Test failures suggest fixes are incomplete or tests are outdated

---

## Failure Analysis

### 1. CRITICAL: test_getitem ValueError (tests/unit/test_dataset.py:103)

**Error**: `ValueError: could not convert string to float: 't'`
**Location**: [ocr/datasets/base.py:347](ocr/datasets/base.py#L347)

```python
poly_array = self._ensure_polygon_array(np.asarray(poly, dtype=np.float32))
# ValueError: could not convert string to float: 't'
```

**Root Cause**: The transform mock is returning strings instead of proper polygon arrays.

**Issue**: In the test, the mock transform returns:
```python
transform.return_value = {
    "image": "transformed_image",
    "polygons": "transformed_polygons",  # ← String, not list!
    "inverse_matrix": "inverse_matrix",
}
```

When the code tries to iterate over `"transformed_polygons"` string, it gets characters 't', 'r', 'a'... and fails to convert to float.

**Fix**: Update test mock to return proper data structures:
```python
transform.return_value = {
    "image": np.zeros((100, 100, 3), dtype=np.uint8),
    "polygons": [np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32)],
    "inverse_matrix": np.eye(3),
}
```

---

### 2. MEDIUM: test_dataset_with_annotations shape mismatch (tests/unit/test_dataset.py:72)

**Error**: `AssertionError: assert (4, 2) == (1, 4, 2)`
**Location**: [tests/unit/test_dataset.py:72](tests/unit/test_dataset.py#L72)

```python
assert polygons[0].shape == (1, 4, 2)  # Expected
# But got: (4, 2)
```

**Root Cause**: Test expectation mismatch

**Analysis**:
- Annotations are stored at [base.py:101](ocr/datasets/base.py#L101) as 2D arrays: `np.array(points)` → shape `(4, 2)`
- The `_ensure_polygon_array()` method adds batch dimension when needed
- BUT this only happens **after transforms**, not in raw annotation storage
- The test is checking `dataset.anns["image1.jpg"]` which is the **raw storage**, not transformed

**Fix Options**:
1. **Update test expectation** (Correct): `assert polygons[0].shape == (4, 2)`
2. **Change storage format** (Not recommended): Would break existing code

**Recommendation**: Fix test expectation - raw annotations should be `(4, 2)`, batch dimension added during processing.

---

### 3. HIGH: test_albumentations_wrapper KeyError (tests/unit/test_preprocessing.py:212)

**Error**: `KeyError: 'You have to pass data to augmentations as named arguments'`
**Location**: [tests/unit/test_preprocessing.py:212](tests/unit/test_preprocessing.py#L212)

```python
result = wrapper(sample_image)  # ← Positional argument!
# Should be: wrapper(image=sample_image)
```

**Root Cause**: Test is calling Albumentations transform incorrectly

**Analysis**:
- BUG-2025-003 was supposedly fixed to inherit from `A.ImageOnlyTransform`
- BUT the test is still using positional arguments instead of keyword arguments
- Albumentations **requires** keyword arguments: `transform(image=img)`

**Verification Needed**:
- Check if [ocr/datasets/preprocessing/pipeline.py](ocr/datasets/preprocessing/pipeline.py) actually has the fix
- Check if the test needs updating

**Fix**: Update test to use keyword arguments:
```python
result = wrapper(image=sample_image)  # Correct way
```

---

### 4. MEDIUM: test_transform_init_args AssertionError (tests/unit/test_preprocessing.py:225)

**Error**: `AssertionError: assert False (isinstance(('preprocessor',), list))`
**Location**: [tests/unit/test_preprocessing.py:225](tests/unit/test_preprocessing.py#L225)

**Root Cause**: `get_transform_init_args_names()` returns tuple, test expects list

**Analysis**:
Looking at BUG-2025-003 fix (from bug report):
```python
def get_transform_init_args_names(self) -> tuple[str, ...]:
    return ("preprocessor",)  # Returns tuple
```

Test expects:
```python
assert isinstance(args, list)  # Expects list!
```

**Fix**: Update test to accept tuple OR update implementation to return list:
```python
# Option 1: Fix test
assert isinstance(args, (list, tuple))

# Option 2: Fix implementation
def get_transform_init_args_names(self) -> tuple[str, ...]:
    return ["preprocessor"]  # Return list instead
```

**Recommendation**: Check Albumentations documentation to see what type is expected.

---

### 5. LOW: test_override_pattern fixture error (tests/unit/test_hydra_overrides.py:21)

**Error**: `fixture 'config_name' not found`

**Root Cause**: Test is missing proper pytest fixture setup

**Fix**: Not related to pipeline malfunction - separate test infrastructure issue.

---

## Current State Assessment

### Performance Features Status (from configs/data/base.yaml)

All performance features are **DISABLED**:
- ❌ `preload_maps: false` - NPZ map preloading disabled
- ❌ `load_maps: false` - NPZ map loading disabled
- ❌ `preload_images: false` - Image RAM caching disabled
- ❌ `cache_transformed_tensors: false` - Tensor caching disabled

**This is good for debugging** - eliminates caching as a variable.

### Bug Fix Status (from bug reports)

According to bug reports, both bugs are **FIXED**:
- ✅ BUG-2025-002: PIL Image type mismatch - Fixed in [ocr/datasets/base.py](ocr/datasets/base.py) and [ocr/datasets/transforms.py](ocr/datasets/transforms.py)
- ✅ BUG-2025-003: Albumentations contract - Fixed in [ocr/datasets/preprocessing/pipeline.py](ocr/datasets/preprocessing/pipeline.py)

BUT unit tests are still failing! **Discrepancy suggests**:
1. Fixes were applied but tests weren't updated, OR
2. Fixes are incomplete, OR
3. Tests have bugs

---

## Investigation Priority

### Phase 1: Verify Fixes (CURRENT)
1. ✅ Read bug reports - DONE
2. ⏳ Read actual source code to verify fixes were applied
3. ⏳ Update tests if fixes are correct
4. ⏳ Re-run tests

### Phase 2: Fix Remaining Issues
1. Fix test mocks to use proper data structures
2. Fix Albumentations test to use keyword arguments
3. Fix polygon shape test expectations
4. Fix transform init args type check

### Phase 3: Integration Testing
1. Run small training test (800/100 images, 1 epoch)
2. Verify h-mean > 0.25
3. Document results

---

## Next Steps

1. **Verify BUG-2025-002 fix in source code**
   - Check [ocr/datasets/base.py:232](ocr/datasets/base.py#L232) for PIL conversion removal
   - Check [ocr/datasets/transforms.py:42](ocr/datasets/transforms.py#L42) for defensive type check

2. **Verify BUG-2025-003 fix in source code**
   - Check [ocr/datasets/preprocessing/pipeline.py](ocr/datasets/preprocessing/pipeline.py) for `A.ImageOnlyTransform` inheritance

3. **Fix unit tests**
   - Update test mocks with proper data structures
   - Update Albumentations test to use keyword args
   - Update polygon shape assertions
   - Update init args type check

4. **Run integration test**
   - Small dataset (800/100)
   - 1 epoch
   - Verify training completes

---

## Questions to Answer

1. **Why do tests fail if fixes are already applied?**
   - Tests outdated?
   - Fixes incomplete?
   - Tests have bugs?

2. **Can we safely enable performance features?**
   - After tests pass?
   - Need additional testing?

3. **What features can be used together?**
   - Create compatibility matrix
   - Document constraints

---

## Files to Check Next

- [ocr/datasets/base.py](ocr/datasets/base.py) - Verify BUG-2025-002 fix
- [ocr/datasets/transforms.py](ocr/datasets/transforms.py) - Verify defensive checks
- [ocr/datasets/preprocessing/pipeline.py](ocr/datasets/preprocessing/pipeline.py) - Verify BUG-2025-003 fix
- [tests/unit/test_dataset.py](tests/unit/test_dataset.py) - Update mocks and expectations
- [tests/unit/test_preprocessing.py](tests/unit/test_preprocessing.py) - Update Albumentations tests
