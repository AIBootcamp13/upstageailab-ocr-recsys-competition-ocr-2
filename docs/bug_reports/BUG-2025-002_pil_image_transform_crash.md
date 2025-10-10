---

## 🐛 Bug Report

**Bug ID:** BUG-2025-002
**Date:** October 10, 2025
**Reporter:** Development Team
**Severity:** High (Training pipeline crash)
**Status:** Identified - Fix Pending

### Summary
`AttributeError: 'Image' object has no attribute 'shape'` occurs in `transforms.py` line 42 when `preload_images=True` but `prenormalize_images=False`. The DBTransforms class expects numpy arrays with `.shape` attribute, but receives PIL Images instead.

### Environment
- **Pipeline Version:** Phase 6E (Tensor Caching)
- **Components:** Dataset loader, DBTransforms, DataLoader workers
- **Configuration:**
  - `preload_images=True` (enabled in configs/data/base.yaml:31)
  - `prenormalize_images=False` (not set, defaults to False)
  - `cache_transformed_tensors=True` (Phase 6E feature)

### Steps to Reproduce
1. Enable RAM caching: `preload_images: true` in dataset config
2. Ensure `prenormalize_images` is not set or set to `false`
3. Run training with canonical data:
   ```bash
   HYDRA_FULL_ERROR=1 uv run python runners/train.py \
     exp_name=pan_resnet18_polygons \
     trainer.max_epochs=15 \
     data=canonical \
     model.component_overrides.decoder.name=pan_decoder \
     logger.wandb.enabled=true \
     trainer.log_every_n_steps=0
   ```
4. Observe crash during validation sanity check:
   ```
   AttributeError: 'Image' object has no attribute 'shape'. Did you mean: 'save'?
   ```

### Expected Behavior
- Transforms should accept both PIL Images and numpy arrays
- Or dataset should consistently pass numpy arrays to transforms
- Training should complete without AttributeError

### Actual Behavior
```python
File "ocr/datasets/transforms.py", line 42, in __call__
  height, width = image.shape[:2]
AttributeError: 'Image' object has no attribute 'shape'
```

**Stack Trace:**
```
File "ocr/datasets/base.py", line 307, in __getitem__
  transformed = self.transform(image=image, polygons=polygons)

File "ocr/datasets/transforms.py", line 42, in __call__
  height, width = image.shape[:2]

AttributeError: 'Image' object has no attribute 'shape'. Did you mean: 'save'?
```

### Root Cause Analysis

**Type Confusion Between PIL Images and Numpy Arrays**

The bug occurs due to inconsistent type handling across the image loading pipeline:

#### Code Flow (Bug Path)

1. **Image Caching** (`base.py:169`):
   ```python
   image_array = np.array(rgb_image)
   self.image_cache[filename] = {
       "image_array": image_array,  # ← Stored as numpy array
       ...
   }
   ```

2. **Cache Retrieval** (`base.py:220-232`):
   ```python
   if image_filename in self.image_cache:
       cached_data = self.image_cache[image_filename]
       image_array = cached_data["image_array"]
       is_normalized = cached_data.get("is_normalized", False)

       if is_normalized:
           image = image_array  # numpy array
       else:
           image = Image.fromarray(image_array)  # ← Converted to PIL Image!
   ```

3. **Transform Call** (`base.py:307`):
   ```python
   transformed = self.transform(image=image, polygons=polygons)
   # ← Passes PIL Image when is_normalized=False
   ```

4. **Transform Processing** (`transforms.py:42`):
   ```python
   def __call__(self, image, polygons):
       height, width = image.shape[:2]  # ← CRASH: PIL Image has no .shape!
   ```

#### The Broken Logic

| Configuration | Cache Type | Retrieved Type | Transform Expectation | Result |
|--------------|------------|----------------|----------------------|--------|
| `prenormalize_images=True` | numpy float32 | numpy array | numpy array | ✅ Works |
| `prenormalize_images=False` | numpy uint8 | **PIL Image** | numpy array | ❌ **Crashes** |
| `preload_images=False` | None | PIL Image | numpy array | ❌ **Should crash but doesn't?** |

**Why line 232 exists:** The code assumes that when images are NOT pre-normalized, they should be converted back to PIL Images to match the behavior of the non-cached path (line 258: `normalized_image` is a PIL Image). However, this assumption is **wrong** because:

1. **Albumentations requires numpy arrays**: The transform library expects numpy arrays, not PIL Images
2. **DBTransforms doesn't handle PIL**: Line 42 directly accesses `.shape[:2]` without type checking
3. **Inconsistent with documentation**: Phase 6B/6E documentation assumes numpy arrays are passed to transforms

### Impact Analysis

**Affected Configurations:**
- ✅ Works: `preload_images=False` (loads PIL, passes PIL... but should fail?)
- ❌ **FAILS**: `preload_images=True, prenormalize_images=False` ← **Current issue**
- ✅ Works: `preload_images=True, prenormalize_images=True`

**User Impact:**
- Training crashes immediately during validation sanity check
- Unable to use RAM caching without pre-normalization
- No workaround except enabling `prenormalize_images=True`

### Resolution Strategy

**Option 1: Fix DBTransforms to handle both types** (Defensive)
```python
# transforms.py:42
def __call__(self, image, polygons):
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    height, width = image.shape[:2]
    # ... rest of code
```

**Pros:**
- Defensive programming
- Handles both PIL and numpy inputs
- No changes to dataset loading logic

**Cons:**
- Adds overhead of type checking
- Doesn't address the root inconsistency

**Option 2: Always pass numpy arrays from dataset** (Preferred)
```python
# base.py:232
# Remove PIL conversion - keep as numpy array
if is_normalized:
    image = image_array  # normalized numpy array
else:
    image = image_array  # uint8 numpy array (don't convert to PIL!)
```

**Pros:**
- Consistent with Albumentations requirements
- No runtime type checking overhead
- Aligns with Phase 6B/6E design intent
- Simpler, cleaner code

**Cons:**
- Need to verify all transforms handle uint8 numpy arrays correctly

**Recommendation:** **Option 2** - Always pass numpy arrays to transforms, as this is what Albumentations expects and what the performance optimizations assume.

### Testing Requirements

**Pre-Fix Verification:**
- [x] Minimal reproduction test created: `test_bug_reproduction.py`
- [x] Confirmed PIL Image crashes in DBTransforms
- [x] Confirmed numpy array works in DBTransforms

**Post-Fix Verification:**
- [ ] Unit test: DBTransforms with uint8 numpy array
- [ ] Unit test: DBTransforms with float32 numpy array
- [ ] Integration test: Full training run with `preload_images=True, prenormalize_images=False`
- [ ] Integration test: Full training run with `preload_images=True, prenormalize_images=True`
- [ ] Regression test: Training without caching still works

### Prevention Measures

1. **Add type checking in DBTransforms** (defensive layer even after fix)
2. **Add unit tests** for transform input types
3. **Document data type contracts**:
   - Dataset `__getitem__` returns: numpy arrays (uint8 or float32)
   - Transforms expect: numpy arrays
   - Collate function expects: tensors
4. **Add assertions** in debug mode to catch type mismatches early
5. **Update Phase 6 documentation** to clarify numpy array contract

### Related Issues

- Phase 6B: RAM caching implementation
- Phase 6C: Pre-normalization feature
- Phase 6E: Tensor caching (exposes this bug due to increased validation runs)

### Next Steps

1. ✅ Create bug report (this document)
2. ⏳ Implement Option 2 fix (remove PIL conversion at line 232)
3. ⏳ Add type assertion at transforms.py:42 (defensive check)
4. ⏳ Run full test suite to verify fix
5. ⏳ Update documentation with type contracts
6. ⏳ Commit with reference to BUG-2025-002

---

**Notes:**
- This bug was introduced during Phase 6B refactoring when adding RAM caching
- The bug was dormant until Phase 6E increased validation frequency
- The root cause is architectural: PIL vs numpy type ambiguity in the pipeline
- Fix is simple but requires careful testing to avoid regressions

---

## Reproduction Test Results

**Test Script:** `test_bug_reproduction.py`

```
🐛 BUG REPRODUCTION TEST SUITE
Testing: AttributeError in transforms.py line 42

======================================================================
TEST 1: DBTransforms with PIL Image (SHOULD FAIL)
======================================================================
✅ EXPECTED FAILURE: 'Image' object has no attribute 'shape'
   Error message: 'Image' object has no attribute 'shape'

======================================================================
TEST 2: DBTransforms with numpy array (SHOULD WORK)
======================================================================
✅ SUCCESS: Transform completed
   Input shape: (600, 800, 3)
   Output shape: torch.Size([3, 640, 640])

======================================================================
ROOT CAUSE ANALYSIS
======================================================================
When preload_images=True but prenormalize_images=False:
1. Images are loaded and cached as numpy arrays (base.py:169)
2. In __getitem__, cached numpy arrays are converted back to PIL Images (base.py:232)
3. PIL Images are passed to DBTransforms.__call__ (base.py:307)
4. DBTransforms tries to access image.shape[:2] (transforms.py:42)
5. PIL Images don't have .shape attribute → AttributeError!
```

**Conclusion:** Bug successfully reproduced. The issue is confirmed as a type mismatch between PIL Images and numpy arrays in the transform pipeline.

---

*Template reference: docs/bug_reports/BUG_REPORT_TEMPLATE.md*
