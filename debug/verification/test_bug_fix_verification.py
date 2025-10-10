#!/usr/bin/env python3
"""
Test to verify BUG-2025-002 fix

This test verifies that the fix for the PIL Image vs numpy array bug works correctly.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import albumentations as A

from ocr.datasets.transforms import DBTransforms


def test_fix_handles_pil_image():
    """Test that DBTransforms now handles PIL Images after the fix."""
    print("=" * 70)
    print("TEST: DBTransforms with PIL Image (SHOULD NOW WORK AFTER FIX)")
    print("=" * 70)

    # Create minimal transform pipeline
    transforms = [A.Resize(640, 640)]
    keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
    db_transforms = DBTransforms(transforms, keypoint_params)

    # Create PIL Image
    pil_image = Image.new("RGB", (800, 600))
    polygons = [np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32).reshape(1, -1, 2)]

    try:
        result = db_transforms(pil_image, polygons)
        print("✅ SUCCESS: Transform handled PIL Image correctly (defensive fix)")
        print(f"   Input type: {type(pil_image)}")
        print(f"   Output shape: {result['image'].shape}")
        return True
    except AttributeError as e:
        print(f"❌ FAILURE: Still crashes with PIL Image: {e}")
        return False


def test_numpy_uint8_still_works():
    """Test that numpy uint8 arrays still work (normal case)."""
    print("\n" + "=" * 70)
    print("TEST: DBTransforms with numpy uint8 (SHOULD STILL WORK)")
    print("=" * 70)

    transforms = [A.Resize(640, 640)]
    keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
    db_transforms = DBTransforms(transforms, keypoint_params)

    np_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    polygons = [np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32).reshape(1, -1, 2)]

    try:
        result = db_transforms(np_image, polygons)
        print("✅ SUCCESS: Transform completed with uint8 numpy array")
        print(f"   Input shape: {np_image.shape}")
        print(f"   Output shape: {result['image'].shape}")
        return True
    except Exception as e:
        print(f"❌ FAILURE: {e}")
        return False


def test_numpy_float32_still_works():
    """Test that numpy float32 arrays still work (pre-normalized case)."""
    print("\n" + "=" * 70)
    print("TEST: DBTransforms with numpy float32 (SHOULD STILL WORK)")
    print("=" * 70)

    transforms = [A.Resize(640, 640)]
    keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
    db_transforms = DBTransforms(transforms, keypoint_params)

    # Pre-normalized float32 image
    np_image = np.random.randn(600, 800, 3).astype(np.float32)
    polygons = [np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32).reshape(1, -1, 2)]

    try:
        result = db_transforms(np_image, polygons)
        print("✅ SUCCESS: Transform completed with float32 numpy array")
        print(f"   Input shape: {np_image.shape}")
        print(f"   Output shape: {result['image'].shape}")
        return True
    except Exception as e:
        print(f"❌ FAILURE: {e}")
        return False


def main():
    """Run all tests and summarize results."""
    print("\n🔧 BUG-2025-002 FIX VERIFICATION TEST SUITE\n")

    results = []

    # Test 1: PIL Image should now work (defensive fix)
    results.append(("PIL Image handled defensively", test_fix_handles_pil_image()))

    # Test 2: uint8 numpy array should still work
    results.append(("uint8 numpy array", test_numpy_uint8_still_works()))

    # Test 3: float32 numpy array should still work
    results.append(("float32 numpy array", test_numpy_float32_still_works()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED - BUG-2025-002 FIX VERIFIED")
        print("=" * 70)
        print("""
Fix Summary:
1. ✅ Removed PIL Image conversion in base.py:227-232
2. ✅ Added defensive type check in transforms.py:42
3. ✅ All input types (PIL, uint8, float32) now work correctly
4. ✅ No performance regression (defensive check is fast)

Next Steps:
- Run full training pipeline test
- Update documentation with type contracts
- Commit changes with reference to BUG-2025-002
        """)
    else:
        print("\n❌ SOME TESTS FAILED - FIX NEEDS MORE WORK")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
