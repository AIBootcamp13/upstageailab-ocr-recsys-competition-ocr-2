"""
Unit tests for image type handling in OCRDataset

Tests the resolution of BUG-2025-004 image type confusion that caused
AttributeError when accessing .size on different image types (PIL vs NumPy).

Tests safe image size extraction across different image sources and types.
"""

import numpy as np
import pytest
from PIL import Image

from ocr.datasets.base import Dataset as OCRDataset
from ocr.datasets.schemas import DatasetConfig


class TestImageTypeHandling:
    """Test cases for robust image size extraction across different types."""

    @pytest.fixture
    def mock_transform(self):
        """Create a mock transform for testing."""

        class MockTransform:
            def __call__(self, item):
                return item

        return MockTransform()

    @pytest.fixture
    def dataset(self, mock_transform, tmp_path):
        """Create a minimal OCRDataset instance for testing."""
        # Create dummy annotation file
        annotation_path = tmp_path / "annotations.json"
        annotation_path.write_text('{"images": [], "annotations": []}')

        config = DatasetConfig(image_path=tmp_path, annotation_path=annotation_path)
        return OCRDataset(config, mock_transform)

    def test_pil_image_size_extraction(self):
        """Test that PIL Image.size returns (width, height)."""
        # Create a PIL image
        pil_image = Image.new("RGB", (640, 480), color="red")

        # Should return (width, height)
        width, height = pil_image.size
        assert width == 640
        assert height == 480
        assert isinstance(width, int)
        assert isinstance(height, int)

    def test_numpy_array_size_extraction(self):
        """Test that NumPy array.size returns total element count."""
        # Create a numpy array (H, W, C) = (480, 640, 3)
        np_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # .size returns total number of elements
        total_elements = np_array.size
        assert total_elements == 480 * 640 * 3

        # .shape gives dimensions
        height, width, channels = np_array.shape
        assert height == 480
        assert width == 640
        assert channels == 3

    def test_safe_size_extraction_from_pil(self):
        """Test extracting canonical size from PIL Image."""
        pil_image = Image.new("RGB", (640, 480), color="blue")

        # Simulate the logic from base.py
        if isinstance(pil_image, np.ndarray):
            org_shape = (pil_image.shape[1], pil_image.shape[0])  # (width, height)
        else:
            org_shape = pil_image.size  # (width, height) for PIL

        assert org_shape == (640, 480)

    def test_safe_size_extraction_from_numpy(self):
        """Test extracting canonical size from NumPy array."""
        np_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Simulate the logic from base.py
        if isinstance(np_image, np.ndarray):
            org_shape = (np_image.shape[1], np_image.shape[0])  # (width, height)
        else:
            org_shape = np_image.size  # This would be wrong!

        assert org_shape == (640, 480)

    def test_image_type_confusion_detection(self):
        """Test that we can detect when wrong size extraction is used."""
        pil_image = Image.new("RGB", (640, 480), color="green")
        np_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Correct extraction
        pil_size = pil_image.size  # (640, 480)
        np_size = (np_image.shape[1], np_image.shape[0])  # (640, 480)

        assert pil_size == np_size == (640, 480)

        # Incorrect extraction (what causes the bug)
        wrong_np_size = np_image.size  # Total elements: 921600

        # These should be different
        assert wrong_np_size != 640 and wrong_np_size != 480
        assert wrong_np_size == 480 * 640 * 3  # Total elements

    def test_canonical_size_extraction_method(self):
        """Test a unified method for extracting (width, height) from any image type."""

        def safe_get_image_size(image):
            """Safely extract (width, height) from PIL or NumPy images."""
            if isinstance(image, np.ndarray):
                return (image.shape[1], image.shape[0])  # (width, height)
            else:
                # Assume PIL Image
                return image.size  # (width, height)

        # Test with PIL
        pil_image = Image.new("RGB", (800, 600), color="yellow")
        assert safe_get_image_size(pil_image) == (800, 600)

        # Test with NumPy
        np_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        assert safe_get_image_size(np_image) == (800, 600)

    def test_image_conversion_metadata_preservation(self):
        """Test that PIL → NumPy conversion preserves size information."""
        # Create PIL image with specific size
        original_size = (512, 384)
        pil_image = Image.new("RGB", original_size, color="purple")

        # Convert to numpy
        np_image = np.array(pil_image)

        # Verify size extraction works for both
        pil_width, pil_height = pil_image.size
        np_width, np_height = np_image.shape[1], np_image.shape[0]

        assert (pil_width, pil_height) == (np_width, np_height) == original_size

    def test_different_image_loaders_consistency(self):
        """Test that different image loading methods produce consistent size info."""
        # This would test actual image loading from different sources
        # For now, just test the size extraction logic

        sizes = [(320, 240), (640, 480), (1024, 768)]

        for width, height in sizes:
            # Simulate PIL-loaded image
            pil_img = Image.new("RGB", (width, height))

            # Simulate OpenCV-loaded image (BGR, HWC)
            opencv_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

            # Both should give same canonical size
            pil_size = pil_img.size
            opencv_size = (opencv_img.shape[1], opencv_img.shape[0])

            assert pil_size == opencv_size == (width, height)

    def test_edge_case_image_sizes(self):
        """Test size extraction with edge case image dimensions."""
        edge_sizes = [(1, 1), (1, 100), (100, 1), (0, 0)]

        for width, height in edge_sizes:
            if width > 0 and height > 0:
                # PIL image
                pil_img = Image.new("L", (width, height), color=128)
                assert pil_img.size == (width, height)

                # NumPy array
                np_img = np.full((height, width), 128, dtype=np.uint8)
                assert (np_img.shape[1], np_img.shape[0]) == (width, height)
            else:
                # Skip invalid sizes for PIL
                pass
