"""
Sample unit tests for OCR project components.

This file demonstrates common testing patterns used in the OCR project:
- Testing individual components
- Using fixtures for setup/teardown
- Testing with mock data
"""

from unittest.mock import Mock

import pytest
import torch


# Example test for tensor operations (common in OCR models)
def test_tensor_operations():
    """Test basic tensor operations that might be used in OCR models."""
    # Create a sample tensor
    data = torch.randn(2, 3, 224, 224)  # batch, channels, height, width
    assert data.shape == (2, 3, 224, 224)

    # Test normalization
    normalized = (data - data.mean()) / (data.std() + 1e-8)
    assert torch.allclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)

    # Test reshape
    reshaped = data.view(2, -1)
    assert reshaped.shape == (2, 3 * 224 * 224)


def test_dummy_ocr_component():
    """Test a dummy OCR component to demonstrate common patterns."""

    # Simulate detection of text in an image
    def detect_text_bounding_boxes(image_tensor):
        # Simulated function that returns bounding boxes
        # Format: [x1, y1, x2, y2] for each detected text
        return [[10, 10, 50, 30], [60, 20, 100, 40], [20, 60, 80, 90]]

    image = torch.rand(3, 224, 224)
    boxes = detect_text_bounding_boxes(image)

    assert len(boxes) == 3  # Expect 3 text boxes
    for box in boxes:
        assert len(box) == 4  # Each box has 4 coordinates
        x1, y1, x2, y2 = box
        assert x1 < x2 and y1 < y2  # Valid bounding box
        assert all(0 <= coord <= 224 for coord in box)  # Within image bounds


# Example of using fixtures for setup
@pytest.fixture
def sample_image():
    """Provides a sample image tensor for testing."""
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def mock_model():
    """Provides a mock model for testing."""
    model = Mock()
    model.forward = Mock(return_value=torch.randn(1, 5, 256, 256))  # Example output
    return model


def test_with_fixtures(sample_image, mock_model):
    """Test using fixtures."""
    # This simulates passing an image through a model
    output = mock_model.forward(sample_image)

    # Assertions about the output
    assert output.shape[0] == 1  # Batch size
    assert output.shape[1] == 5  # Number of output channels
    assert output.shape[2] == 256  # Height
    assert output.shape[3] == 256  # Width

    # Verify the mock was called once
    mock_model.forward.assert_called_once_with(sample_image)


# Parametrized test for different scenarios
@pytest.mark.parametrize(
    "input_size,expected_shape",
    [
        ((1, 3, 224, 224), (1, 3, 224, 224)),
        ((2, 1, 512, 512), (2, 1, 512, 512)),
        ((4, 3, 128, 128), (4, 3, 128, 128)),
    ],
)
def test_different_input_sizes(input_size, expected_shape):
    """Test tensor operations with different input sizes."""
    tensor = torch.randn(*input_size)
    assert tensor.shape == expected_shape


# Example of an expected failure test
@pytest.mark.xfail
def test_known_buggy_feature():
    """Test for a feature that is known to have issues (will fail but won't break the test suite)."""
    # This simulates a known issue that needs to be fixed later
    assert False, "This is a known issue that needs fixing"


# Example test for utility function
def test_confidence_filtering():
    """Test confidence filtering utility."""

    def filter_by_confidence(boxes, scores, threshold=0.5):
        """Filter bounding boxes by confidence score."""
        return [(box, score) for box, score in zip(boxes, scores, strict=False) if score >= threshold]

    boxes = [[10, 10, 50, 30], [60, 20, 100, 40], [20, 60, 80, 90]]
    scores = [0.8, 0.3, 0.9]

    filtered = filter_by_confidence(boxes, scores, threshold=0.5)

    assert len(filtered) == 2  # Only 2 scores >= 0.5
    assert filtered[0][1] == 0.8  # First kept item has score 0.8
    assert filtered[1][1] == 0.9  # Second kept item has score 0.9


if __name__ == "__main__":
    # Run tests directly if executed as script
    test_tensor_operations()
    test_dummy_ocr_component()
    test_different_input_sizes((1, 3, 224, 224), (1, 3, 224, 224))
    test_confidence_filtering()
    print("All tests passed!")
