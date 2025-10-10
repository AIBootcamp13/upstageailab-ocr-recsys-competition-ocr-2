# BUG-2025-002 Fix Summary and Test Findings

**Date:** October 10, 2025
**Bug ID:** BUG-2025-002
**Status:** ✅ **FIXED** (Original AttributeError resolved)
**Secondary Issue:** ⚠️ Discovered IndexError in Albumentations (requires separate investigation)

---

## Executive Summary

**Original Bug:** `AttributeError: 'Image' object has no attribute 'shape'` at `transforms.py:42`
- ✅ **ROOT CAUSE IDENTIFIED**: PIL Images passed to DBTransforms when numpy arrays expected
- ✅ **FIX IMPLEMENTED**: Two-layer defense strategy
- ✅ **UNIT TESTS PASS**: All image types (PIL, uint8, float32) now work correctly
- ⚠️ **INTEGRATION TEST**: Revealed secondary IndexError in Albumentations pipeline

---

## Fix Implementation

### Layer 1: Removed PIL Conversion (Primary Fix)
**File:** `ocr/datasets/base.py:220-228`

**Before:**
```python
if is_normalized:
    image = image_array
else:
    image = Image.fromarray(image_array)  # ← BUG: Creates PIL Image
```

**After:**
```python
# Always use numpy arrays for transforms (Albumentations/DBTransforms require numpy)
# BUG FIX (BUG-2025-002): Previously converted to PIL Image when is_normalized=False,
# causing AttributeError in transforms.py:42 (PIL Image has no .shape attribute)
image = image_array  # Keep as numpy array (uint8 or float32)
```

### Layer 2: Defensive Type Check (Safety Net)
**File:** `ocr/datasets/transforms.py:42-49`

**Added:**
```python
def __call__(self, image, polygons):
    # BUG FIX (BUG-2025-002): Add defensive type check for PIL Images
    # Albumentations/DBTransforms expect numpy arrays, not PIL Images
    from PIL import Image as PILImage

    if isinstance(image, PILImage.Image):
        image = np.array(image)

    height, width = image.shape[:2]
```

---

## Test Results

### Unit Tests (test_bug_fix_verification.py)
```
🔧 BUG-2025-002 FIX VERIFICATION TEST SUITE

======================================================================
TEST 1: DBTransforms with PIL Image
======================================================================
✅ SUCCESS: Transform handled PIL Image correctly (defensive fix)
   Input type: <class 'PIL.Image.Image'>
   Output shape: torch.Size([3, 640, 640])

======================================================================
TEST 2: DBTransforms with numpy uint8
======================================================================
✅ SUCCESS: Transform completed with uint8 numpy array
   Input shape: (600, 800, 3)
   Output shape: torch.Size([3, 640, 640])

======================================================================
TEST 3: DBTransforms with numpy float32
======================================================================
✅ SUCCESS: Transform completed with float32 numpy array
   Input shape: (600, 800, 3)
   Output shape: torch.Size([3, 640, 640])

======================================================================
🎉 ALL UNIT TESTS PASSED - BUG-2025-002 FIX VERIFIED
======================================================================
```

**Conclusion:** ✅ The original AttributeError is completely fixed.

---

## Integration Test Findings

### Command
```bash
HYDRA_FULL_ERROR=1 uv run python runners/train.py \
  exp_name=bug_fix_test \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=5 \
  trainer.limit_val_batches=5 \
  data=canonical \
  model.component_overrides.decoder.name=pan_decoder \
  logger.wandb.enabled=false
```

### Result
✅ **Original bug FIXED**: No more `AttributeError: 'Image' object has no attribute 'shape'`

⚠️ **New error discovered**: `IndexError` in Albumentations pipeline

```python
File "ocr/datasets/transforms.py", line 59, in __call__
  transformed = self.transform(image=image, keypoints=keypoints)

File "albumentations/core/composition.py", line 222, in _check_data_post_transform
  rows, cols = get_shape(data["image"])

IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`)
           and integer or boolean arrays are valid indices
```

**Analysis:**
- This is a DIFFERENT bug from BUG-2025-002
- Occurs deeper in the Albumentations pipeline
- Related to how Albumentations inspects the transformed image shape
- Likely caused by tensor/array type confusion after transforms
- Needs separate investigation and bug report

---

## Impact Assessment

### What Was Fixed
- ✅ PIL Image → numpy array type mismatch in DBTransforms
- ✅ Defensive handling prevents future PIL Image crashes
- ✅ All three image types (PIL, uint8, float32) now work in unit tests
- ✅ Code is more robust with explicit type contracts

### What Remains
- ⚠️ IndexError in Albumentations `get_shape()` function
- ⚠️ Full training pipeline still crashes (but at a different point)
- ⚠️ May be related to ToTensorV2() or other transform operations

---

## Recommendations

### Immediate Actions
1. ✅ **DONE**: Original bug fix verified and committed
2. ⏳ **TODO**: Create BUG-2025-003 for IndexError investigation
3. ⏳ **TODO**: Add debug logging to identify exact transform causing IndexError
4. ⏳ **TODO**: Test with simpler transform pipeline to isolate issue

### Long-term Improvements
1. Add type annotations throughout pipeline
2. Create integration tests for full training pipeline
3. Add assertions at pipeline boundaries
4. Document data type contracts in code comments

---

## Files Modified

```
✅ ocr/datasets/base.py (line 220-228)
   - Removed PIL Image conversion
   - Added BUG-2025-002 reference comment

✅ ocr/datasets/transforms.py (line 42-49)
   - Added defensive PIL Image→numpy conversion
   - Added BUG-2025-002 reference comment

✅ test_bug_reproduction.py (NEW)
   - Minimal reproduction test for original bug

✅ test_bug_fix_verification.py (NEW)
   - Verification test for fix

✅ docs/bug_reports/BUG-2025-002_pil_image_transform_crash.md (NEW)
   - Complete bug report with analysis

✅ docs/bug_reports/BUG-2025-002_fix_findings.md (THIS FILE)
   - Fix implementation and test results
```

---

## Next Steps

1. **Create BUG-2025-003** for the Albumentations IndexError
2. **Add debug logging** to identify which transform causes the IndexError
3. **Test with minimal transforms** (just Resize, no ToTensorV2) to isolate issue
4. **Check Albumentations version** compatibility with our data types
5. **Review ConditionalNormalize** transform (added in Phase 6C) for potential issues

---

## Conclusion

**BUG-2025-002 IS RESOLVED** ✅

The original AttributeError (`'Image' object has no attribute 'shape'`) has been completely fixed with a robust two-layer defense:
1. Primary fix: Always pass numpy arrays from dataset
2. Safety net: Convert PIL Images to numpy in transforms

However, fixing this bug exposed a deeper issue in the Albumentations pipeline that requires separate investigation (BUG-2025-003).

---

**Commit Message:**
```
Fix BUG-2025-002: PIL Image vs numpy array type mismatch in transforms

- Remove PIL Image conversion in base.py when preload_images=True
- Add defensive type check in DBTransforms to handle PIL Images
- All image types (PIL, uint8, float32) now work correctly
- Add unit tests for fix verification

Refs: BUG-2025-002
```
