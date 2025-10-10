#!/usr/bin/env python3
"""
Minimal reproduction test for PIL Image vs numpy array bug in transforms.py

Bug: AttributeError: 'Image' object has no attribute 'shape'
Location: ocr/datasets/transforms.py:42
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import albumentations as A

from ocr.datasets.transforms import DBTransforms


def test_pil_image_fails():
    """Test that demonstrates DBTransforms fails with PIL Image input."""
    print("=" * 70)
    print("TEST 1: DBTransforms with PIL Image (SHOULD FAIL)")
    print("=" * 70)

    # Create minimal transform pipeline
    transforms = [A.Resize(640, 640)]
    keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
    db_transforms = DBTransforms(transforms, keypoint_params)

    # Create PIL Image (this is what happens with preload_images=True, prenormalize_images=False)
    pil_image = Image.new("RGB", (800, 600))
    polygons = [np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32).reshape(1, -1, 2)]

    try:
        db_transforms(pil_image, polygons)
        print("❌ UNEXPECTED: Transform succeeded with PIL Image!")
        return False
    except AttributeError as e:
        print(f"✅ EXPECTED FAILURE: {e}")
        print("   Error message: 'Image' object has no attribute 'shape'")
        return True


def test_numpy_array_works():
    """Test that demonstrates DBTransforms works with numpy array input."""
    print("\n" + "=" * 70)
    print("TEST 2: DBTransforms with numpy array (SHOULD WORK)")
    print("=" * 70)

    # Create minimal transform pipeline
    transforms = [A.Resize(640, 640)]
    keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
    db_transforms = DBTransforms(transforms, keypoint_params)

    # Create numpy array (this is what should be passed)
    np_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    polygons = [np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32).reshape(1, -1, 2)]

    try:
        result = db_transforms(np_image, polygons)
        print("✅ SUCCESS: Transform completed")
        print(f"   Input shape: {np_image.shape}")
        print(f"   Output shape: {result['image'].shape}")
        return True
    except Exception as e:
        print(f"❌ UNEXPECTED FAILURE: {e}")
        return False


def test_albumentations_handles_pil():
    """Test that Albumentations can handle PIL Images directly."""
    print("\n" + "=" * 70)
    print("TEST 3: Albumentations with PIL Image (SHOULD WORK)")
    print("=" * 70)

    # Create Albumentations pipeline directly (without DBTransforms wrapper)
    transform = A.Compose([A.Resize(640, 640)])

    pil_image = Image.new("RGB", (800, 600))

    try:
        # Albumentations should handle PIL Images
        result = transform(image=pil_image)
        print("✅ SUCCESS: Albumentations handled PIL Image")
        print(f"   Input type: {type(pil_image)}")
        print(f"   Output type: {type(result['image'])}")
        print(f"   Output shape: {result['image'].shape}")
        return True
    except Exception as e:
        print(f"❌ FAILURE: {e}")
        return False


def main():
    """Run all tests and summarize results."""
    print("\n🐛 BUG REPRODUCTION TEST SUITE")
    print("Testing: AttributeError in transforms.py line 42\n")

    results = []

    # Test 1: PIL Image should fail in DBTransforms
    results.append(("PIL Image in DBTransforms", test_pil_image_fails()))

    # Test 2: Numpy array should work in DBTransforms
    results.append(("Numpy array in DBTransforms", test_numpy_array_works()))

    # Test 3: Albumentations handles PIL Images
    results.append(("PIL Image in Albumentations", test_albumentations_handles_pil()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)
    print("""
When preload_images=True but prenormalize_images=False:
1. Images are loaded and cached as numpy arrays (base.py:169)
2. In __getitem__, cached numpy arrays are converted back to PIL Images (base.py:232)
3. PIL Images are passed to DBTransforms.__call__ (base.py:307)
4. DBTransforms tries to access image.shape[:2] (transforms.py:42)
5. PIL Images don't have .shape attribute → AttributeError!

The fix: DBTransforms should convert PIL Images to numpy arrays before accessing .shape,
or base.py should always pass numpy arrays to transforms.
    """)

    all_passed = all(passed for _, passed in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
