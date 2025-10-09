"""
Test for a simulated OCR postprocessing component.

This demonstrates how we might test an actual OCR-related utility function
that could be part of the project.
"""

import numpy as np
import pytest


def apply_threshold(probability_map: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Apply threshold to probability map to get binary map.

    Args:
        probability_map: 2D array of probabilities
        threshold: Threshold value (0-1)

    Returns:
        Binary map with 0s and 1s
    """
    return (probability_map > threshold).astype(np.uint8)


def extract_connected_components(binary_map: np.ndarray) -> list:
    """
    Extract connected components from binary map using simple flood fill.

    Args:
        binary_map: 2D binary array

    Returns:
        List of bounding boxes for connected components
    """
    visited = np.zeros_like(binary_map, dtype=bool)
    components = []

    def flood_fill(start_r, start_c):
        """Simple flood fill to find connected components."""
        stack = [(start_r, start_c)]
        component_pixels = []

        while stack:
            r, c = stack.pop()
            if r < 0 or r >= binary_map.shape[0] or c < 0 or c >= binary_map.shape[1] or visited[r, c] or binary_map[r, c] == 0:
                continue

            visited[r, c] = True
            component_pixels.append((r, c))

            # Add neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((r + dr, c + dc))

        return component_pixels

    for r in range(binary_map.shape[0]):
        for c in range(binary_map.shape[1]):
            if binary_map[r, c] != 0 and not visited[r, c]:
                pixels = flood_fill(r, c)
                if pixels:
                    rows, cols = zip(*pixels, strict=False)
                    bbox = [min(cols), min(rows), max(cols), max(rows)]  # [x1, y1, x2, y2]
                    components.append(bbox)

    return components


def postprocess_text_map(probability_map: np.ndarray, threshold: float = 0.3) -> list:
    """
    Postprocess a text probability map to extract bounding boxes.

    Args:
        probability_map: 2D array of text probability values
        threshold: Probability threshold

    Returns:
        List of bounding boxes [x1, y1, x2, y2]
    """
    binary_map = apply_threshold(probability_map, threshold)
    return extract_connected_components(binary_map)


def test_threshold_function():
    """Test the threshold function."""
    prob_map = np.array([[0.1, 0.4, 0.2], [0.6, 0.3, 0.8], [0.3, 0.2, 0.7]])
    binary_map = apply_threshold(prob_map, threshold=0.35)

    expected = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]])  # 0s and 1s based on threshold
    np.testing.assert_array_equal(binary_map, expected)


def test_extract_connected_components():
    """Test connected component extraction."""
    binary_map = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]])

    components = extract_connected_components(binary_map)

    # Should have 2 components: a 2x2 block and a single pixel
    assert len(components) == 2

    # Check if one component is the 2x2 block at top left
    expected_bbox1 = [1, 1, 2, 2]  # [x1, y1, x2, y2] - inclusive coordinates
    assert expected_bbox1 in components

    # Check if the other component is the single pixel
    expected_bbox2 = [3, 3, 3, 3]  # [x1, y1, x2, y2] - single pixel
    assert expected_bbox2 in components


def test_postprocess_text_map():
    """Test the full postprocessing pipeline."""
    # Create a test probability map with 2 text regions
    prob_map = np.zeros((10, 10))

    # First text region
    prob_map[2:4, 2:6] = 0.7  # A 2x4 region of high probability

    # Second text region
    prob_map[6:8, 3:7] = 0.8  # A 2x4 region of high probability

    # Some noise below threshold
    prob_map[1, 1] = 0.2  # Should be filtered out

    bboxes = postprocess_text_map(prob_map, threshold=0.5)

    assert len(bboxes) == 2  # Should detect 2 text regions

    # Check that bounding boxes are reasonable
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        assert x1 <= x2 and y1 <= y2  # Valid bounding box
        assert 0 <= x1 <= 9 and 0 <= x2 <= 9  # Within bounds
        assert 0 <= y1 <= 9 and 0 <= y2 <= 9  # Within bounds


@pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_postprocess_with_different_thresholds(threshold):
    """Test postprocessing with different thresholds."""
    prob_map = np.random.rand(20, 20)

    bboxes = postprocess_text_map(prob_map, threshold=threshold)

    # All returned bounding boxes should be valid
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        assert x1 <= x2 and y1 <= y2  # Valid bounding box
        assert 0 <= x1 <= 19 and 0 <= x2 <= 19  # Within bounds
        assert 0 <= y1 <= 19 and 0 <= y2 <= 19  # Within bounds


def test_postprocess_edge_cases():
    """Test edge cases for postprocessing."""
    # Empty map (all zeros)
    empty_map = np.zeros((10, 10))
    bboxes = postprocess_text_map(empty_map)
    assert len(bboxes) == 0

    # Full map (all ones)
    full_map = np.ones((10, 10))
    bboxes = postprocess_text_map(full_map, threshold=0.5)
    # Should detect one large component
    assert len(bboxes) == 1
    bbox = bboxes[0]
    # The bounding box should encompass most of the map
    assert bbox[0] <= 1 and bbox[1] <= 1  # x1, y1 close to 0
    assert bbox[2] >= 8 and bbox[3] >= 8  # x2, y2 close to 9


if __name__ == "__main__":
    # Run tests directly if executed as script
    test_threshold_function()
    test_extract_connected_components()
    test_postprocess_text_map()
    test_postprocess_edge_cases()

    # Run parametrized test for a few thresholds
    for threshold in [0.1, 0.5, 0.9]:
        test_postprocess_with_different_thresholds(threshold)

    print("All OCR postprocessing tests passed!")
