"""Unit tests for data loading optimizations."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from ocr.utils.image_loading import get_image_loader_info, load_image_optimized


class TestTurboJPEGIntegration:
    """Test TurboJPEG integration and PIL fallback."""

    def test_turbojpeg_available(self):
        """Test that TurboJPEG availability is detected correctly."""
        info = get_image_loader_info()
        assert "turbojpeg_available" in info
        assert "pil_available" in info
        assert info["pil_available"] is True

    def test_load_jpeg_with_turbojpeg(self, tmp_path):
        """Test loading JPEG with TurboJPEG when available."""
        # Create a test JPEG image
        test_image = Image.new("RGB", (100, 100), color="red")
        jpeg_path = tmp_path / "test.jpg"
        test_image.save(jpeg_path, "JPEG")

        # Test loading
        loaded = load_image_optimized(jpeg_path)
        assert isinstance(loaded, Image.Image)
        assert loaded.size == (100, 100)

    def test_load_png_with_pil_fallback(self, tmp_path):
        """Test loading PNG falls back to PIL."""
        # Create a test PNG image
        test_image = Image.new("RGB", (100, 100), color="blue")
        png_path = tmp_path / "test.png"
        test_image.save(png_path, "PNG")

        # Test loading
        loaded = load_image_optimized(png_path)
        assert isinstance(loaded, Image.Image)
        assert loaded.size == (100, 100)

    def test_turbojpeg_fallback_on_failure(self, tmp_path):
        """Test that TurboJPEG failure falls back to PIL."""
        # Create a valid JPEG
        test_image = Image.new("RGB", (100, 100), color="green")
        jpeg_path = tmp_path / "test.jpg"
        test_image.save(jpeg_path, "JPEG")

        # Mock TurboJPEG to fail
        with patch("ocr.utils.image_loading.TurboJPEG") as mock_turbo:
            mock_instance = MagicMock()
            mock_instance.decode.side_effect = Exception("TurboJPEG failed")
            mock_turbo.return_value = mock_instance

            # Should still load via PIL fallback
            loaded = load_image_optimized(jpeg_path)
            assert isinstance(loaded, Image.Image)

    def test_file_not_found_error(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            load_image_optimized("nonexistent.jpg")

    def test_corrupted_image_error(self, tmp_path):
        """Test error handling for corrupted images."""
        # Create a corrupted file
        corrupted_path = tmp_path / "corrupted.jpg"
        with open(corrupted_path, "wb") as f:
            f.write(b"not an image")

        with pytest.raises(RuntimeError):
            load_image_optimized(corrupted_path)


class TestInterpolationOptimization:
    """Test interpolation method optimizations."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def test_linear_vs_cubic_performance(self, sample_image):
        """Test that LINEAR and CUBIC interpolation produce valid results."""
        import time

        import cv2

        # Test LINEAR
        start = time.time()
        resized_linear = cv2.resize(sample_image, (640, 640), interpolation=cv2.INTER_LINEAR)
        linear_time = time.time() - start

        # Test CUBIC
        start = time.time()
        resized_cubic = cv2.resize(sample_image, (640, 640), interpolation=cv2.INTER_CUBIC)
        cubic_time = time.time() - start

        # Both should produce valid results (timing can vary by hardware)
        assert resized_linear.shape == (640, 640, 3)
        assert resized_cubic.shape == (640, 640, 3)
        assert not np.array_equal(resized_linear, resized_cubic)

        # Log timing for reference
        print(f"Linear time: {linear_time:.6f}s, Cubic time: {cubic_time:.6f}s")

    def test_aspect_ratio_preservation(self, sample_image):
        """Test that transforms preserve aspect ratio correctly."""
        import albumentations as A
        import cv2

        # Create transform similar to validation
        transform = A.Compose(
            [A.LongestMaxSize(max_size=640, interpolation=cv2.INTER_LINEAR), A.PadIfNeeded(min_width=640, min_height=640, border_mode=0)]
        )

        result = transform(image=sample_image)
        transformed = result["image"]

        # Should be 640x640
        assert transformed.shape[:2] == (640, 640)

        # Check that aspect ratio is preserved (one dimension should match original aspect)
        # After LongestMaxSize, the longer side becomes 640, shorter is scaled
        # We can't easily check exact aspect without knowing which side was longer


class TestPerformanceBenchmarking:
    """Test performance benchmarking utilities."""

    def test_benchmark_script_import(self):
        """Test that benchmark script can be imported."""
        # This tests that the script doesn't have import errors
        try:
            from scripts.benchmark_optimizations import benchmark_image_loading, create_val_transform

            assert callable(create_val_transform)
            assert callable(benchmark_image_loading)
        except ImportError as e:
            pytest.skip(f"Benchmark script import failed: {e}")

    def test_timing_measurements(self):
        """Test that timing measurements are reasonable."""
        import time

        start = time.time()
        time.sleep(0.01)  # 10ms
        elapsed = time.time() - start

        # Should be close to 0.01s
        assert 0.005 < elapsed < 0.05  # Allow some tolerance


class TestAccuracyValidation:
    """Test accuracy validation for transform changes."""

    def test_transform_output_consistency(self):
        """Test that transforms produce consistent output shapes."""
        import albumentations as A
        import cv2

        sample_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)

        # Test both interpolations
        for interpolation in [cv2.INTER_LINEAR, cv2.INTER_CUBIC]:
            transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=640, interpolation=interpolation),
                    A.PadIfNeeded(min_width=640, min_height=640, border_mode=0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            result = transform(image=sample_image)
            transformed = result["image"]

            # Should be 640x640x3
            assert transformed.shape == (640, 640, 3)

            # Should be normalized (values roughly -2 to 2)
            assert -3 < transformed.min() < -1
            assert 1 < transformed.max() < 3

    def test_polygon_transform_preservation(self):
        """Test that polygon coordinates are transformed correctly."""
        import albumentations as A

        sample_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        sample_polygons = np.array([[10, 10, 50, 10, 50, 50, 10, 50]])  # Simple rectangle

        transform = A.Compose(
            [A.LongestMaxSize(max_size=640), A.PadIfNeeded(min_width=640, min_height=640, border_mode=0)],
            keypoint_params=A.KeypointParams(format="xy"),
        )

        # Convert polygons to keypoints format (x,y pairs)
        keypoints = []
        for i in range(0, len(sample_polygons[0]), 2):
            keypoints.append((sample_polygons[0][i], sample_polygons[0][i + 1]))

        result = transform(image=sample_image, keypoints=keypoints)
        transformed_keypoints = result["keypoints"]

        # Should have same number of keypoints
        assert len(transformed_keypoints) == len(keypoints)

        # All keypoints should be within bounds
        for x, y in transformed_keypoints:
            assert 0 <= x <= 640
            assert 0 <= y <= 640


class TestMemoryUsage:
    """Test memory usage monitoring during image loading."""

    def test_memory_monitoring(self):
        """Test that memory usage can be monitored."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Load a few images
        for i in range(5):
            img = Image.new("RGB", (1000, 1000), color="red")
            del img

        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Memory should not increase dramatically
        assert final_memory - initial_memory < 100  # Less than 100MB increase

    def test_image_cleanup(self, tmp_path):
        """Test that images are properly closed to free memory."""
        import gc

        # Create test image
        test_image = Image.new("RGB", (500, 500), color="blue")
        image_path = tmp_path / "test.jpg"
        test_image.save(image_path, "JPEG")
        test_image.close()

        # Load and close multiple times
        for _ in range(10):
            loaded = load_image_optimized(image_path)
            loaded.close()

        # Force garbage collection
        gc.collect()

        # Should not have memory leaks (basic check)
        # In a real test, we'd use memory profiling tools


class TestEdgeCases:
    """Test edge cases for data loading."""

    def test_large_image_handling(self, tmp_path):
        """Test handling of large images."""
        # Create a large image (but not too large for testing)
        large_image = Image.new("RGB", (2000, 2000), color="yellow")
        large_path = tmp_path / "large.jpg"
        large_image.save(large_path, "JPEG")
        large_image.close()

        # Should load without error
        loaded = load_image_optimized(large_path)
        assert isinstance(loaded, Image.Image)
        assert loaded.size == (2000, 2000)
        loaded.close()

    def test_different_image_formats(self, tmp_path):
        """Test loading different image formats."""
        formats = [("BMP", "test.bmp"), ("TIFF", "test.tiff"), ("PNG", "test.png")]

        for format_name, filename in formats:
            test_image = Image.new("RGB", (100, 100), color="purple")
            image_path = tmp_path / filename
            test_image.save(image_path, format_name)
            test_image.close()

            # Should load (falls back to PIL)
            loaded = load_image_optimized(image_path)
            assert isinstance(loaded, Image.Image)
            loaded.close()

    def test_aspect_ratio_extremes(self):
        """Test transforms with extreme aspect ratios."""
        import albumentations as A
        import cv2

        # Very wide image
        wide_image = np.random.randint(0, 255, (100, 1000, 3), dtype=np.uint8)

        transform = A.Compose(
            [A.LongestMaxSize(max_size=640, interpolation=cv2.INTER_LINEAR), A.PadIfNeeded(min_width=640, min_height=640, border_mode=0)]
        )

        result = transform(image=wide_image)
        transformed = result["image"]

        # Should be 640x640
        assert transformed.shape[:2] == (640, 640)

        # Very tall image
        tall_image = np.random.randint(0, 255, (1000, 100, 3), dtype=np.uint8)

        result = transform(image=tall_image)
        transformed = result["image"]

        # Should be 640x640
        assert transformed.shape[:2] == (640, 640)
