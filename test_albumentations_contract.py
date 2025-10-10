#!/usr/bin/env python3
"""Test Albumentations custom transform contract to understand the bug"""

import albumentations as A
import numpy as np

print("Testing Albumentations custom transform contract\n")
print("=" * 70)


# Example 1: CORRECT - Albumentations-style transform (returns dict or modifies in-place)
class CorrectTransform(A.ImageOnlyTransform):
    def apply(self, img, **params):
        # Just return the modified image, Albumentations wraps it
        return img * 0.9  # Darken slightly

    def get_transform_init_args_names(self):
        return []


# Example 2: WRONG - Returning image directly in __call__
class WrongTransform:
    def __call__(self, image, **kwargs):
        # This bypasses Albumentations and returns raw image
        return image * 0.9  # ← This is what LensStylePreprocessor does!

    def get_transform_init_args_names(self):
        return []


# Test with Albumentations Compose
test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

print("Test 1: Using A.ImageOnlyTransform (CORRECT)")
try:
    transform = A.Compose([CorrectTransform()])
    result = transform(image=test_image)
    print(f"✅ Result type: {type(result)}")
    print(f"✅ Result keys: {list(result.keys())}")
    print(f"✅ Image shape: {result['image'].shape}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 70 + "\n")
print("Test 2: Using custom __call__ that returns image (WRONG)")
try:
    transform = A.Compose([WrongTransform()])
    result = transform(image=test_image)
    print(f"Result type: {type(result)}")
    print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict!'}")
    print(f"Image value: {result.get('image', 'No image key!')}")
except Exception as e:
    print(f"❌ Expected Error: {type(e).__name__}: {str(e)[:100]}")

print("\n" + "=" * 70)
print("\n🔍 DIAGNOSIS:")
print("LensStylePreprocessorAlbumentations.__call__ returns result['image'] (numpy array)")
print("But Albumentations expects the transform to return dict OR use apply() method")
print("This causes get_shape() to receive a numpy array instead of proper dict structure!")
