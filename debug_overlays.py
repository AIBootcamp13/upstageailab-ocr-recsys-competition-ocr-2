#!/usr/bin/env python3
"""
Debug script for preprocessing overlays.

Tests overlay drawing functionality directly to identify issues.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def draw_detection_overlay_debug(image: np.ndarray, corners: np.ndarray | None, method: str | None, debug_info: bool = True) -> np.ndarray:
    """Draw detection overlay with debug information."""
    overlay = image.copy()

    if debug_info:
        print(f"Input image shape: {image.shape}")
        print(f"Input image dtype: {image.dtype}")
        print(f"Corners: {corners}")
        print(f"Method: {method}")

    if corners is not None:
        print(f"Corners shape: {corners.shape}")
        print(f"Corners dtype: {corners.dtype}")
        print(f"Corners values:\n{corners}")

        if len(corners) == 4:
            # Draw corners as circles
            for i, corner in enumerate(corners):
                x, y = int(corner[0]), int(corner[1])
                print(f"Drawing corner {i}: ({x}, {y})")
                cv2.circle(overlay, (x, y), 12, (0, 255, 0), -1)
                cv2.putText(overlay, str(i), (x + 15, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw edges as lines
            edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
            for i, j in edges:
                pt1 = (int(corners[i][0]), int(corners[i][1]))
                pt2 = (int(corners[j][0]), int(corners[j][1]))
                print(f"Drawing edge {i}->{j}: {pt1} -> {pt2}")
                cv2.line(overlay, pt1, pt2, (255, 0, 0), 3)

            # Add method label
            if method:
                cv2.putText(overlay, f"Method: {method}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                cv2.putText(overlay, f"Method: {method}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        else:
            print(f"ERROR: Expected 4 corners, got {len(corners)}")
    else:
        print("No corners to draw")
        cv2.putText(overlay, "NO CORNERS DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(overlay, "NO CORNERS DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    return overlay


def test_overlay_with_sample_data():
    """Test overlay drawing with sample data from the test run."""

    # Load a sample image
    image_path = Path("LOW_PERFORMANCE_IMGS/drp.en_ko.in_house.selectstar_003949.jpg")
    if not image_path.exists():
        print(f"Sample image not found: {image_path}")
        return

    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    image_array = np.array(image)
    print(f"Image shape: {image_array.shape}")

    # Test cases from the actual test run
    test_cases = [
        {
            "name": "bounding_box_corners",
            "corners": np.array([[0.0, 0.0], [719.0, 0.0], [719.0, 1279.0], [0.0, 1279.0]]),
            "method": "bounding_box",
        },
        {"name": "camscanner_corners", "corners": np.array([[161, 10], [553, 43], [604, 1242], [84, 1235]]), "method": "camscanner"},
        {"name": "no_corners", "corners": None, "method": None},
    ]

    for test_case in test_cases:
        print(f"\n=== Testing {test_case['name']} ===")

        overlay = draw_detection_overlay_debug(image_array, test_case["corners"], test_case["method"], debug_info=True)

        # Save debug overlay
        output_path = Path(f"debug_overlay_{test_case['name']}.jpg")
        cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved debug overlay: {output_path}")


def test_corner_coordinate_system():
    """Test if corners are in the right coordinate system."""

    # Create a simple test image
    test_image = np.full((100, 200, 3), 128, dtype=np.uint8)

    # Test corners in different formats
    test_corners = [
        ("top-left rectangle", np.array([[10, 10], [190, 10], [190, 90], [10, 90]])),
        ("full image", np.array([[0, 0], [199, 0], [199, 99], [0, 99]])),
        ("normalized coords", np.array([[0.05, 0.1], [0.95, 0.1], [0.95, 0.9], [0.05, 0.9]]) * np.array([200, 100])),
    ]

    for name, corners in test_corners:
        print(f"\n=== Testing {name} ===")
        print(f"Corners: {corners}")

        overlay = draw_detection_overlay_debug(test_image, corners, name, debug_info=False)

        output_path = Path(f"debug_coords_{name.replace(' ', '_')}.jpg")
        cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    print("=== Overlay Debug Script ===\n")

    test_overlay_with_sample_data()
    test_corner_coordinate_system()

    print("\n=== Debug Complete ===")
    print("Check the generated debug_overlay_*.jpg files to see what's being drawn.")
