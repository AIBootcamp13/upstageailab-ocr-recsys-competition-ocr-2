#!/usr/bin/env python3
"""
Demonstration of the canonical_size bug and how it was fixed.
"""

import numpy as np
from PIL import Image


def demonstrate_bug():
    """Show the difference between PIL Image.size and numpy array.size"""
    print("=== The canonical_size Bug Demonstration ===\n")

    # Create a sample image
    sample_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil_image = Image.fromarray(sample_array)

    print("Original PIL Image:")
    print(f"  pil_image.size = {pil_image.size} (type: {type(pil_image.size)})")
    print("  This is correct: (width, height) tuple\n")

    print("When converted to numpy array (like in Phase 6B caching):")
    numpy_array = np.array(pil_image)
    print(f"  numpy_array.shape = {numpy_array.shape} (correct shape tuple)")
    print(f"  numpy_array.size = {numpy_array.size} (type: {type(numpy_array.size)})")
    print("  This is WRONG: total number of elements, not (width, height)!\n")

    print("The Bug Flow:")
    print("1. Phase 6B: Images cached as numpy arrays")
    print("2. Phase 6C: prenormalize_images=True → is_normalized=True")
    print("3. __getitem__: image = image_array (numpy array)")
    print("4. org_shape = image.size → gets integer instead of tuple")
    print("5. Collate: canonical_sizes = [integer, ...]")
    print("6. Lightning: tuple(integer) → TypeError: 'int' object is not iterable\n")

    print("The Fix:")
    print("Ensure shape is always (width, height) tuple:")
    print("  if isinstance(image, np.ndarray):")
    print("      org_shape = (image.shape[1], image.shape[0])  # (width, height)")
    print("  else:")
    print("      org_shape = image.size  # (width, height) for PIL")


if __name__ == "__main__":
    demonstrate_bug()
