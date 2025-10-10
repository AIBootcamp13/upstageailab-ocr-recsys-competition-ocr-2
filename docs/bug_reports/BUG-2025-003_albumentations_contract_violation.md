---

## 🐛 Bug Report

**Bug ID:** BUG-2025-003
**Date:** October 10, 2025
**Reporter:** Development Team
**Severity:** High (Training pipeline crash)
**Status:** Identified - Fix Pending

### Summary
`IndexError: only integers, slices (:), ellipsis (...), numpy.newaxis (None) and integer or boolean arrays are valid indices` occurs in Albumentations `get_shape()` function when using `LensStylePreprocessorAlbumentations`. The preprocessor incorrectly returns a numpy array instead of a dictionary, breaking Albumentations' internal data flow.

### Environment
- **Pipeline Version:** Phase 6E with preprocessing enabled
- **Components:** LensStylePreprocessorAlbumentations, Albumentations Compose, DBTransforms
- **Configuration:** `data=canonical` with document preprocessing enabled

### Steps to Reproduce
1. Enable document preprocessing with `LensStylePreprocessorAlbumentations`
2. Run training with preprocessing transforms:
   ```yaml
   val_transform:
     transforms:
       - ${lens_style_preprocessor}  # ← Problem here
       - albumentations.Normalize
   ```
3. Observe crash in Albumentations composition:
   ```
   IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`)
              and integer or boolean arrays are valid indices
   ```

### Expected Behavior
- `LensStylePreprocessorAlbumentations` should work seamlessly with Albumentations Compose
- Transforms should pass processed images through the pipeline correctly
- Training should complete without IndexError

### Actual Behavior
```python
File "ocr/datasets/transforms.py", line 59, in __call__
  transformed = self.transform(image=image, keypoints=keypoints)

File "albumentations/core/composition.py", line 213, in __call__
  data = self._check_data_post_transform(data)

File "albumentations/core/composition.py", line 222, in _check_data_post_transform
  rows, cols = get_shape(data["image"])

IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`)
           and integer or boolean arrays are valid indices
```

### Root Cause Analysis

**Incorrect Implementation of Albumentations Transform Contract**

The bug is in how `LensStylePreprocessorAlbumentations` integrates with Albumentations:

#### Broken Implementation (Current)

```python
# ocr/datasets/preprocessing/pipeline.py:257-267
class LensStylePreprocessorAlbumentations:
    """Albumentations-compatible wrapper for the document preprocessor."""

    def __init__(self, preprocessor: DocumentPreprocessor):
        self.preprocessor = preprocessor

    def __call__(self, image, **kwargs):  # type: ignore[override]
        result = self.preprocessor(image)
        # Return just the processed image for Albumentations compatibility
        return result["image"]  # ← BUG: Returns numpy array, not dict!

    def get_transform_init_args_names(self):
        return []
```

#### Why This Breaks

1. **Albumentations Contract Violation**:
   - Albumentations transforms must either:
     - Inherit from `BasicTransform` and implement `apply(img, **params)` method
     - Return a dictionary with at least `{"image": ...}` structure
   - `LensStylePreprocessorAlbumentations` does neither correctly

2. **Data Flow Breakdown**:
   ```
   DBTransforms.transform(image=np_array, keypoints=[...])
     └─> A.Compose([LensStylePreprocessor, Normalize])
           ├─> LensStylePreprocessor.__call__(image=np_array)
           │     └─> returns np_array  # ← Should return dict!
           │
           └─> Albumentations expects dict, gets np_array
                 └─> get_shape(data["image"]) tries to index np_array
                       └─> IndexError!
   ```

3. **What Alb...umentations Expects**:
   ```python
   # Albumentations internal flow
   data = {"image": image, "keypoints": keypoints}

   for transform in transforms:
       data = transform(**data)  # Expects dict → dict
       rows, cols = get_shape(data["image"])  # ← Crashes if data is not dict!
   ```

#### Verification Test

```python
import albumentations as A
import numpy as np

# WRONG way (current implementation)
class WrongTransform:
    def __call__(self, image, **kwargs):
        return image  # Returns numpy array

# Test with Albumentations
test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

try:
    transform = A.Compose([WrongTransform()])
    result = transform(image=test_image)
except AttributeError as e:
    print(f"❌ Error: {e}")  # 'numpy.ndarray' object has no attribute 'items'
```

**Result**: `AttributeError: 'numpy.ndarray' object has no attribute 'items'`

This is the same class of error we're seeing, just manifesting as IndexError in a different part of Albumentations.

### Resolution Strategy

**Option 1: Inherit from A.ImageOnlyTransform** (Recommended)

```python
class LensStylePreprocessorAlbumentations(A.ImageOnlyTransform):
    """Albumentations-compatible wrapper for the document preprocessor."""

    def __init__(self, preprocessor: DocumentPreprocessor, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.preprocessor = preprocessor

    def apply(self, img, **params):
        # Albumentations automatically wraps the returned image in a dict
        result = self.preprocessor(img)
        return result["image"]

    def get_transform_init_args_names(self):
        return ("preprocessor",)
```

**Pros**:
- Correct Albumentations contract
- Automatic parameter handling
- Consistent with other transforms
- No changes to config files

**Cons**:
- Need to handle preprocessor serialization

**Option 2: Return Dictionary in __call__**

```python
def __call__(self, image, **kwargs):
    result = self.preprocessor(image)
    # Return dict to maintain Albumentations contract
    return {"image": result["image"], **kwargs}
```

**Pros**:
- Minimal code change
- Preserves other data (keypoints, etc.)

**Cons**:
- Still not a proper BasicTransform
- May break in future Albumentations versions

**Recommendation**: **Option 1** - Properly inherit from `A.ImageOnlyTransform` for robustness and future compatibility.

### Impact Analysis

**Affected Configurations:**
- ❌ **FAILS**: Any config using `lens_style_preprocessor` in transforms
- ❌ **FAILS**: `data=canonical` (uses preprocessing)
- ✅ Works: Configs without preprocessing

**User Impact:**
- Training crashes immediately during validation sanity check (after BUG-2025-002 fix)
- Unable to use document preprocessing with Albumentations transforms
- Blocks all advanced preprocessing features

**Workaround**:
- Disable preprocessing: `preprocessing.enable_document_detection=false`
- Not viable for production use

### Testing Requirements

**Pre-Fix Verification:**
- [x] Minimal reproduction test created: `test_albumentations_contract.py`
- [x] Confirmed wrong implementation breaks Albumentations
- [x] Verified correct implementation works

**Post-Fix Verification:**
- [ ] Unit test: LensStylePreprocessorAlbumentations with simple image
- [ ] Unit test: Full preprocessing pipeline with Albumentations
- [ ] Integration test: Training run with preprocessing enabled
- [ ] Integration test: Validation with `data=canonical`
- [ ] Regression test: Training without preprocessing still works

### Prevention Measures

1. **Add unit tests** for Albumentations transform contract compliance
2. **Document transform requirements** in code comments
3. **Create base class** for custom Albumentations transforms
4. **Add CI checks** for Albumentations compatibility
5. **Code review checklist** for new transforms

### Related Issues

- BUG-2025-002: PIL Image type mismatch (fixed, exposed this bug)
- Phase 6E: Tensor caching (increased validation runs, revealed bug)
- Document preprocessing feature (root cause)

### Next Steps

1. ✅ Create bug report (this document)
2. ⏳ Implement Option 1 fix (inherit from A.ImageOnlyTransform)
3. ⏳ Add unit tests for transform contract
4. ⏳ Run full training test with preprocessing
5. ⏳ Update documentation with transform best practices
6. ⏳ Commit with reference to BUG-2025-003

---

**Notes:**
- This bug was dormant because preprocessing was recently added
- BUG-2025-002 fix exposed this by making the pipeline progress further
- Root cause is misunderstanding of Albumentations transform contract
- Fix is straightforward but requires careful testing

---

## Technical Deep Dive

### How Albumentations Transforms Work

Albumentations has three main transform types:

1. **ImageOnlyTransform**: Affects only the image
   ```python
   class MyTransform(A.ImageOnlyTransform):
       def apply(self, img, **params):
           return modified_img  # Albumentations wraps in dict
   ```

2. **DualTransform**: Affects image and targets (masks, keypoints, etc.)
   ```python
   class MyDualTransform(A.DualTransform):
       def apply(self, img, **params):
           return modified_img

       def apply_to_keypoint(self, keypoint, **params):
           return modified_keypoint
   ```

3. **BasicTransform**: Base class for custom behavior

### Why Direct __call__ Doesn't Work

When you implement `__call__` directly without inheriting from `BasicTransform`:
- Albumentations can't recognize it as a valid transform
- Type checking fails (see lint error in test)
- Data flow breaks because Albumentations expects dict → dict
- Internal validation fails with IndexError or AttributeError

### The Correct Pattern

```python
# Good: Inherit from ImageOnlyTransform
class GoodTransform(A.ImageOnlyTransform):
    def __init__(self, param, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.param = param

    def apply(self, img, **params):
        # Process image
        return processed_img  # Framework handles dict wrapping

    def get_transform_init_args_names(self):
        return ("param",)
```

---

*Commit Message:*
```
Fix BUG-2025-003: LensStylePreprocessorAlbumentations violates Albumentations contract

- Inherit from A.ImageOnlyTransform instead of plain class
- Implement apply() method instead of __call__()
- Properly handle Albumentations transform lifecycle
- Add unit tests for transform contract compliance

Refs: BUG-2025-003
```
