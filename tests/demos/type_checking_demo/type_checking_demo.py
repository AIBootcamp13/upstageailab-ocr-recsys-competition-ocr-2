#!/usr/bin/env python3
"""
Demonstration: How type checking would have prevented the canonical_size bug
"""

import numpy as np
from PIL import Image


# Current (buggy) code - no type hints
def get_shape_current(image) -> tuple:
    """Current implementation - no type safety"""
    return image.size  # Could be int or tuple!


# Properly typed version
def get_shape_typed(image: Image.Image | np.ndarray) -> tuple[int, int]:
    """
    Properly typed version that would catch the bug at type-check time.

    This function expects to return a (width, height) tuple, but:
    - PIL.Image.size returns (width, height) ✓
    - np.ndarray.size returns total_elements ✗

    Pylance would flag this as a type error!
    """
    if isinstance(image, np.ndarray):
        # Correct: use .shape for numpy arrays
        height, width = image.shape[:2]  # Note: numpy is (height, width)
        return (width, height)  # Convert to (width, height) convention
    else:
        # PIL Image
        return image.size  # (width, height)


def demonstrate_type_checking():
    """Show how type checking reveals the bug"""

    print("=== Type Checking Would Have Caught This Bug ===\n")

    # Create test data
    pil_img = Image.new("RGB", (224, 224))
    np_array = np.array(pil_img)

    print("PIL Image:")
    print(f"  pil_img.size = {pil_img.size} (type: {type(pil_img.size)})")
    print(f"  get_shape_current(pil_img) = {get_shape_current(pil_img)} ✓")
    print(f"  get_shape_typed(pil_img) = {get_shape_typed(pil_img)} ✓")

    print("\nNumpy Array:")
    print(f"  np_array.shape = {np_array.shape} (correct shape)")
    print(f"  np_array.size = {np_array.size} (wrong for our use case!)")
    print(f"  get_shape_current(np_array) = {get_shape_current(np_array)} ✗ BUG!")
    print(f"  get_shape_typed(np_array) = {get_shape_typed(np_array)} ✓ FIXED!")

    print("\n=== What Pylance Would Have Flagged ===")

    # This is what the type checker would see:
    print("In the original code:")
    print("  def get_shape_current(image) -> tuple[int, int]:")
    print("      return image.size  # Type checker: 'image' could be np.ndarray!")
    print("  # np.ndarray.size returns int, but function promises tuple[int, int]")
    print("  # PYLANCE ERROR: Incompatible return type!")

    print("\nWith proper typing:")
    print("  def get_shape_typed(image: Union[Image.Image, np.ndarray]) -> tuple[int, int]:")
    print("      if isinstance(image, np.ndarray):")
    print("          return (image.shape[1], image.shape[0])  # Correct!")
    print("      else:")
    print("          return image.size")
    print("  # Type checker: All paths return tuple[int, int] ✓")


if __name__ == "__main__":
    demonstrate_type_checking()
