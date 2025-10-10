"""
Tests for Microsoft Lens-style document preprocessing.
"""

import numpy as np
import pytest

from ocr.datasets.preprocessing import DOCTR_AVAILABLE, DocumentPreprocessor, LensStylePreprocessorAlbumentations


class TestDocumentPreprocessor:
    """Test cases for DocumentPreprocessor class."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def preprocessor(self):
        """Create a DocumentPreprocessor instance."""
        return DocumentPreprocessor(
            enable_document_detection=True,
            enable_perspective_correction=True,
            enable_enhancement=True,
            target_size=(640, 640),
        )

    def test_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor.enable_document_detection is True
        assert preprocessor.enable_perspective_correction is True
        assert preprocessor.enable_enhancement is True
        assert preprocessor.target_size == (640, 640)

    def test_preprocessing_pipeline(self, preprocessor, sample_image):
        """Test the full preprocessing pipeline."""
        result = preprocessor(sample_image)

        # Check that result contains expected keys
        assert "image" in result
        assert "metadata" in result

        # Check that processed image has correct shape
        processed_image = result["image"]
        assert processed_image.shape == (640, 640, 3)

        # Check metadata structure
        metadata = result["metadata"]
        assert "original_shape" in metadata
        assert "processing_steps" in metadata
        assert "final_shape" in metadata

    def test_document_detection(self, preprocessor, sample_image):
        """Test document boundary detection."""
        corners = preprocessor._detect_document_boundaries(sample_image)

        # For random noise image, detection might fail (which is expected)
        # We just check that the method doesn't crash
        assert corners is None or isinstance(corners, np.ndarray)

    @pytest.mark.skipif(not DOCTR_AVAILABLE, reason="python-doctr not installed")
    def test_doctr_perspective_toggle(self, sample_image, monkeypatch):
        """Ensure docTR geometry is used when configured."""
        preprocessor = DocumentPreprocessor(
            enable_document_detection=True,
            enable_perspective_correction=True,
            enable_enhancement=False,
            use_doctr_geometry=True,
        )

        height, width = sample_image.shape[:2]
        rectangle = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )

        monkeypatch.setattr(
            preprocessor.detector,
            "detect",
            lambda _image: (rectangle, "mock"),
        )

        result = preprocessor(sample_image)
        metadata = result["metadata"]
        assert isinstance(metadata, dict)

        assert metadata.get("perspective_method") == "doctr_rcrop"
        assert "perspective_correction" in metadata.get("processing_steps", [])

    @pytest.mark.skipif(not DOCTR_AVAILABLE, reason="python-doctr not installed")
    def test_orientation_correction_metadata(self, monkeypatch):
        """Orientation correction should log metadata when applied."""
        image = np.full((256, 256, 3), 200, dtype=np.uint8)
        skewed_corners = np.array(
            [
                [20, 40],
                [220, 10],
                [230, 210],
                [15, 230],
            ],
            dtype=np.float32,
        )
        axis_aligned = np.array(
            [
                [0, 0],
                [255, 0],
                [255, 255],
                [0, 255],
            ],
            dtype=np.float32,
        )

        preprocessor = DocumentPreprocessor(
            enable_document_detection=True,
            enable_perspective_correction=False,
            enable_enhancement=False,
            enable_orientation_correction=True,
            orientation_angle_threshold=0.5,
        )

        call_state = {"count": 0}

        def fake_detect(_image: np.ndarray) -> tuple[np.ndarray | None, str | None]:
            call_state["count"] += 1
            if call_state["count"] == 1:
                return skewed_corners, "mock_initial"
            return axis_aligned, "mock_redetect"

        monkeypatch.setattr(preprocessor.detector, "detect", fake_detect)

        result = preprocessor(image)
        metadata = result["metadata"]
        assert isinstance(metadata, dict)
        orientation_meta = metadata.get("orientation")

        assert orientation_meta is not None
        assert orientation_meta.get("redetection_success") is True
        assert abs(orientation_meta.get("angle_correction", 0.0)) > 0
        assert "orientation_correction" in metadata.get("processing_steps", [])

    @pytest.mark.skipif(not DOCTR_AVAILABLE, reason="python-doctr not installed")
    def test_padding_cleanup_step(self):
        """Padding cleanup should be reflected in metadata when enabled."""
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        image[20:100, 20:100] = 255

        preprocessor = DocumentPreprocessor(
            enable_document_detection=False,
            enable_perspective_correction=False,
            enable_enhancement=False,
            enable_padding_cleanup=True,
        )

        result = preprocessor(image)
        metadata = result["metadata"]
        assert isinstance(metadata, dict)

        assert "padding_cleanup" in metadata.get("processing_steps", [])

    def test_image_enhancement(self, preprocessor, sample_image):
        """Test image enhancement functionality."""
        enhanced, applied = preprocessor._enhance_image(sample_image)

        assert enhanced.shape == sample_image.shape
        assert isinstance(applied, list)
        assert len(applied) > 0

    def test_resize_to_target(self, preprocessor):
        """Test image resizing to target dimensions."""
        # Create a smaller test image
        small_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

        resized = preprocessor._resize_to_target(small_image)

        assert resized.shape == (640, 640, 3)

    def test_order_corners(self, preprocessor):
        """Test corner ordering functionality."""
        # Create test corners
        corners = np.array(
            [
                [100, 100],  # top-left
                [200, 100],  # top-right
                [200, 200],  # bottom-right
                [100, 200],  # bottom-left
            ]
        )

        ordered = preprocessor._order_corners(corners)

        assert ordered.shape == (4, 2)


class TestLensStylePreprocessorAlbumentations:
    """Test cases for Albumentations wrapper."""

    def test_albumentations_wrapper(self):
        """Test Albumentations-compatible wrapper."""
        # Create a numpy array image for testing
        sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        preprocessor = DocumentPreprocessor()
        wrapper = LensStylePreprocessorAlbumentations(preprocessor)

        # BUG FIX (BUG-2025-003): Albumentations transforms require keyword arguments
        # Test the apply method (which is what Albumentations calls internally)
        result = wrapper.apply(sample_image)

        # Should return just the processed image (as numpy array)
        assert isinstance(result, np.ndarray)
        assert result.shape[-1] == 3  # RGB image

    def test_transform_init_args(self):
        """Test transform initialization arguments."""
        preprocessor = DocumentPreprocessor()
        wrapper = LensStylePreprocessorAlbumentations(preprocessor)

        args = wrapper.get_transform_init_args_names()

        # BUG FIX: get_transform_init_args_names() returns tuple in Albumentations convention
        assert isinstance(args, list | tuple)
        assert len(args) > 0


class TestPreprocessingIntegration:
    """Integration tests for preprocessing components."""

    @pytest.fixture
    def preprocessor(self):
        """Create a DocumentPreprocessor instance for integration tests."""
        return DocumentPreprocessor(
            enable_document_detection=True,
            enable_perspective_correction=True,
            enable_enhancement=True,
            target_size=(640, 640),
        )

    def test_preprocessing_with_real_image(self):
        """Test preprocessing with a more realistic document-like image."""
        # Create a more document-like image (white background with some content)
        image = np.full((480, 640, 3), 255, dtype=np.uint8)

        # Add some "text" regions (darker areas)
        image[100:150, 100:500] = [100, 100, 100]  # horizontal text line
        image[200:250, 100:400] = [80, 80, 80]  # another text line

        preprocessor = DocumentPreprocessor(
            enable_document_detection=False,  # Skip detection for this test
            enable_perspective_correction=False,  # Skip correction for this test
            enable_enhancement=True,
        )

        result = preprocessor(image)

        processed = result["image"]
        assert isinstance(processed, np.ndarray)
        assert processed.shape == (640, 640, 3)
        assert "image_enhancement" in result["metadata"]["processing_steps"]

    def test_error_handling(self, preprocessor):
        """Test error handling in preprocessing pipeline."""
        # Test with invalid input (shouldn't crash)
        invalid_image = np.array([])  # Empty array

        # Should handle gracefully and return fallback result
        result = preprocessor(invalid_image)

        # Should still return a valid result structure
        assert "image" in result
        assert "metadata" in result
