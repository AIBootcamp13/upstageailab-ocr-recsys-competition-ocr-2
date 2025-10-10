# Initial Unit Test Run - 2025-10-10

## Test Summary
- **Total Tests**: 89
- **Passed**: 83
- **Failed**: 4
- **Errors**: 1
- **Xfailed**: 1

## Failures Analysis

### 1. test_dataset_with_annotations (tests/unit/test_dataset.py:72)
**Location**: [tests/unit/test_dataset.py:72](tests/unit/test_dataset.py#L72)

**Error**:
```
AssertionError: assert (4, 2) == (1, 4, 2)
Expected polygon shape: (1, 4, 2) - One polygon with 4 points
Actual shape: (4, 2)
```

**Issue**: Polygon shape mismatch - missing batch dimension

---

### 2. test_getitem (tests/unit/test_dataset.py:103)
**Location**: [tests/unit/test_dataset.py:103](tests/unit/test_dataset.py#L103)
**Failure Location**: [ocr/datasets/base.py:347](ocr/datasets/base.py#L347)

**Error**:
```
ValueError: could not convert string to float: 't'
```

**Issue**: String being passed where float array expected - likely polygon data corruption

---

### 3. test_albumentations_wrapper (tests/unit/test_preprocessing.py:212)
**Location**: [tests/unit/test_preprocessing.py:212](tests/unit/test_preprocessing.py#L212)

**Error**:
```
KeyError: 'You have to pass data to augmentations as named arguments, for example: aug(image=image)'
```

**Issue**: Albumentations contract violation - relates to BUG-2025-003

---

### 4. test_transform_init_args (tests/unit/test_preprocessing.py:225)
**Location**: [tests/unit/test_preprocessing.py:225](tests/unit/test_preprocessing.py#L225)

**Error**:
```
AssertionError: assert False
  +  where False = isinstance(('preprocessor',), list)
```

**Issue**: Transform initialization returning tuple instead of list

---

### 5. test_override_pattern (tests/unit/test_hydra_overrides.py:21) - ERROR
**Location**: [tests/unit/test_hydra_overrides.py:21](tests/unit/test_hydra_overrides.py#L21)

**Error**:
```
fixture 'config_name' not found
```

**Issue**: Test fixture configuration problem - not related to pipeline malfunction

---

## Priority Order
1. **HIGH**: Failures 2 & 3 - Dataset and Albumentations issues block pipeline
2. **MEDIUM**: Failures 1 & 4 - Shape and initialization issues
3. **LOW**: Error 5 - Test fixture configuration

## Next Steps
1. Investigate dataset polygon handling in [ocr/datasets/base.py:347](ocr/datasets/base.py#L347)
2. Review Albumentations wrapper implementation
3. Check polygon shape expectations throughout codebase
