"""Comprehensive pytest test suite for the ValidatedOCRDataset class."""

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import the refactored dataset class and schemas
from ocr.datasets.base import ValidatedOCRDataset
from ocr.datasets.schemas import CacheConfig, DatasetConfig, ImageLoadingConfig


def create_mock_dataset_config(image_path, annotation_path=None, cache_transformed_tensors=False):
    """Helper function to create a proper DatasetConfig for testing."""
    cache_config = CacheConfig(cache_transformed_tensors=cache_transformed_tensors, cache_images=True, cache_maps=True)
    image_loading_config = ImageLoadingConfig(use_turbojpeg=False, turbojpeg_fallback=False)
    return DatasetConfig(
        image_path=image_path, annotation_path=annotation_path, cache_config=cache_config, image_loading_config=image_loading_config
    )


class TestValidatedOCRDatasetInitialization:
    """Test ValidatedOCRDataset initialization and basic attributes."""

    @pytest.fixture
    def mock_transform(self):
        """Fixture providing a mock transform function."""
        return Mock()

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Fixture providing a mock dataset configuration."""
        return create_mock_dataset_config(tmp_path, cache_transformed_tensors=False)

    def test_initialization_with_config(self, mock_config, mock_transform):
        """Test that ValidatedOCRDataset initializes with provided config and transform."""
        dataset = ValidatedOCRDataset(config=mock_config, transform=mock_transform)

        assert dataset.config == mock_config
        assert dataset.transform == mock_transform
        assert isinstance(dataset.logger, logging.Logger)
        assert hasattr(dataset, "anns")
        assert hasattr(dataset, "cache_manager")

    def test_initial_empty_annotations(self, mock_config, mock_transform, tmp_path):
        """Test that annotations dictionary is initialized empty."""
        dataset = ValidatedOCRDataset(config=mock_config, transform=mock_transform)

        assert len(dataset.anns) == 0

    def test_annotations_loaded_from_config(self, mock_config, mock_transform, tmp_path):
        """Test that annotations are loaded according to configuration."""
        # Create a sample image file
        sample_image = tmp_path / "sample.jpg"
        sample_image.write_bytes(b"fake image data")

        dataset = ValidatedOCRDataset(config=mock_config, transform=mock_transform)

        # Since no annotation path was provided, should contain the sample image
        assert len(dataset.anns) == 1
        assert "sample.jpg" in dataset.anns


class TestValidatedOCRDatasetLength:
    """Test ValidatedOCRDataset __len__ method."""

    @pytest.fixture
    def mock_transform(self):
        """Fixture providing a mock transform function."""
        return Mock()

    @pytest.fixture
    def mock_config_with_images(self, tmp_path):
        """Fixture providing a mock config with sample images."""
        # Create sample image files
        (tmp_path / "image1.jpg").write_bytes(b"fake image 1")
        (tmp_path / "image2.png").write_bytes(b"fake image 2")

        return create_mock_dataset_config(tmp_path, cache_transformed_tensors=False)

    def test_dataset_length(self, mock_config_with_images, mock_transform):
        """Test that dataset length matches number of annotations."""
        dataset = ValidatedOCRDataset(config=mock_config_with_images, transform=mock_transform)

        assert len(dataset) == 2  # Two images created in fixture


class TestValidatedOCRDatasetGetItem:
    """Test ValidatedOCRDataset __getitem__ method."""

    @pytest.fixture
    def mock_transform(self):
        """Fixture providing a mock transform function that returns a valid response."""

        def transform_func(transform_input):
            # Mock return value that matches expected transform output
            return {
                "image": np.random.rand(3, 100, 100),
                "polygons": [np.array([[0, 0], [10, 0], [10, 10], [0, 10]])],
                "inverse_matrix": np.eye(3),
                "metadata": {"test": "metadata"},
            }

        return transform_func

    @pytest.fixture
    def mock_config_with_single_image(self, tmp_path):
        """Fixture providing a mock config with a single sample image."""
        # Create a single sample image file - will be mocked during loading
        sample_image_path = tmp_path / "sample.jpg"
        # Create a minimal file that exists so the path check passes
        sample_image_path.write_bytes(b"fake image content")

        return create_mock_dataset_config(tmp_path, cache_transformed_tensors=False)

    def test_getitem_returns_valid_sample(self, mock_config_with_single_image, mock_transform):
        """Test that __getitem__ returns a valid sample with expected structure."""
        import numpy as np

        from ocr.datasets.schemas import ImageData

        # Mock the _load_image_data method to return a pre-built ImageData object
        with patch.object(ValidatedOCRDataset, "_load_image_data") as mock_load_image_data:
            # Create a mock ImageData object with valid data
            mock_image_data = ImageData(
                image_array=np.random.rand(100, 100, 3).astype(np.uint8), raw_width=100, raw_height=100, orientation=1, is_normalized=False
            )
            mock_load_image_data.return_value = mock_image_data

            # The dataset initialization will trigger annotation loading
            # which may try to access files before our mock is in place
            # So we need to ensure the image file exists or mock the annotation loading too
            dataset = ValidatedOCRDataset(config=mock_config_with_single_image, transform=mock_transform)

            # Get the first (and only) item
            sample = dataset[0]

            # Check that sample has expected keys
            assert "image" in sample
            assert "polygons" in sample
            assert "inverse_matrix" in sample
            assert "metadata" in sample

            # Check that image is a numpy array
            assert isinstance(sample["image"], np.ndarray)

            # Check that polygons is a list
            assert isinstance(sample["polygons"], list)

    def test_getitem_with_tensor_caching(self, mock_config_with_single_image, mock_transform):
        """Test that __getitem__ works with tensor caching enabled."""
        import numpy as np

        from ocr.datasets.schemas import ImageData

        # Create config with caching enabled
        config = create_mock_dataset_config(mock_config_with_single_image.image_path, cache_transformed_tensors=True)

        # Mock the _load_image_data method to return a pre-built ImageData object
        with patch.object(ValidatedOCRDataset, "_load_image_data") as mock_load_image_data:
            # Create a mock ImageData object with valid data
            mock_image_data = ImageData(
                image_array=np.random.rand(100, 100, 3).astype(np.uint8), raw_width=100, raw_height=100, orientation=1, is_normalized=False
            )
            mock_load_image_data.return_value = mock_image_data

            dataset = ValidatedOCRDataset(config=config, transform=mock_transform)

            # Get the first item - this should cache it
            first_sample = dataset[0]

            # Get the same item again - should come from cache
            second_sample = dataset[0]

            # Both samples should be equivalent
            assert np.array_equal(first_sample["image"], second_sample["image"])


class TestValidatedOCRDatasetAnnotations:
    """Test annotation loading functionality."""

    @pytest.fixture
    def mock_transform(self):
        """Fixture providing a mock transform function."""
        return Mock()

    def test_load_annotations_no_annotation_file(self, tmp_path, mock_transform):
        """Test loading annotations when no annotation file is provided."""
        # Create sample images
        (tmp_path / "img1.jpg").write_bytes(b"fake image 1")
        (tmp_path / "img2.png").write_bytes(b"fake image 2")

        config = create_mock_dataset_config(tmp_path, annotation_path=None, cache_transformed_tensors=False)

        dataset = ValidatedOCRDataset(config=config, transform=mock_transform)

        # Should load all images with None annotations
        assert len(dataset.anns) == 2
        assert dataset.anns["img1.jpg"] is None
        assert dataset.anns["img2.png"] is None

    def test_load_annotations_with_annotation_file(self, tmp_path, mock_transform):
        """Test loading annotations from an annotation file."""
        # Create sample images
        (tmp_path / "img1.jpg").write_bytes(b"fake image 1")

        # Create annotation file
        annotation_file = tmp_path / "annotations.json"
        annotation_data = {"images": {"img1.jpg": {"words": {"word1": {"points": [[10, 10], [20, 10], [20, 20], [10, 20]]}}}}}
        import json

        with open(annotation_file, "w") as f:
            json.dump(annotation_data, f)

        config = create_mock_dataset_config(tmp_path, annotation_path=annotation_file, cache_transformed_tensors=False)

        dataset = ValidatedOCRDataset(config=config, transform=mock_transform)

        # Should load annotations from file
        assert len(dataset.anns) == 1
        assert "img1.jpg" in dataset.anns
        # The annotation should be a list of polygons
        assert dataset.anns["img1.jpg"] is not None
        assert len(dataset.anns["img1.jpg"]) == 1


class TestValidatedOCRDatasetCaching:
    """Test caching functionality."""

    @pytest.fixture
    def mock_transform(self):
        """Fixture providing a mock transform function."""

        def transform_func(transform_input):
            return {
                "image": np.random.rand(3, 100, 100),
                "polygons": [np.array([[0, 0], [10, 0], [10, 10], [0, 10]])],
                "inverse_matrix": np.eye(3),
                "metadata": {"test": "metadata"},
            }

        return transform_func

    @pytest.fixture
    def mock_config_with_caching(self, tmp_path):
        """Fixture providing a mock config with caching enabled."""
        # Create a sample image
        (tmp_path / "sample.jpg").write_bytes(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00"
            b"\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f"
        )

        return create_mock_dataset_config(tmp_path, cache_transformed_tensors=True)

    def test_tensor_caching_mechanism(self, mock_config_with_caching, mock_transform):
        """Test that tensor caching works correctly."""
        import numpy as np

        from ocr.datasets.schemas import ImageData

        # Mock the _load_image_data method to return a pre-built ImageData object
        with patch.object(ValidatedOCRDataset, "_load_image_data") as mock_load_image_data:
            # Create a mock ImageData object with valid data
            mock_image_data = ImageData(
                image_array=np.random.rand(100, 100, 3).astype(np.uint8), raw_width=100, raw_height=100, orientation=1, is_normalized=False
            )
            mock_load_image_data.return_value = mock_image_data

            dataset = ValidatedOCRDataset(config=mock_config_with_caching, transform=mock_transform)

            # First access should cache the result
            first_result = dataset[0]
            initial_cache_size = len(dataset.cache_manager.tensor_cache)

            # Second access should retrieve from cache
            second_result = dataset[0]
            final_cache_size = len(dataset.cache_manager.tensor_cache)

            # Cache should have exactly one item
            assert final_cache_size == 1
            assert initial_cache_size == final_cache_size  # Size should not change on second access

            # Results should be equivalent
            assert np.array_equal(first_result["image"], second_result["image"])


class TestValidatedOCRDatasetErrorHandling:
    """Test error handling in ValidatedOCRDataset."""

    @pytest.fixture
    def mock_transform(self):
        """Fixture providing a mock transform function."""
        return Mock()

    def test_missing_image_path(self, mock_transform):
        """Test that dataset handles missing image path gracefully."""
        # Use a non-existent path
        non_existent_path = Path("/non/existent/path")

        # Create config without annotation file to avoid error
        config = create_mock_dataset_config(non_existent_path, annotation_path=None, cache_transformed_tensors=False)

        dataset = ValidatedOCRDataset(config=config, transform=mock_transform)

        # Should have no annotations since path doesn't exist
        assert len(dataset.anns) == 0

    def test_invalid_annotation_file(self, tmp_path, mock_transform):
        """Test that dataset handles invalid annotation files."""
        # Create an invalid annotation file
        annotation_file = tmp_path / "invalid_annotations.json"
        annotation_file.write_text("invalid json content")

        config = create_mock_dataset_config(tmp_path, annotation_path=annotation_file, cache_transformed_tensors=False)

        # Should raise an error when trying to load invalid JSON
        with pytest.raises(RuntimeError):  # Invalid JSON should raise RuntimeError
            ValidatedOCRDataset(config=config, transform=mock_transform)
