import pytest
import json
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
from unittest.mock import Mock

from ocr.datasets.base import OCRDataset


class TestOCRDataset:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_annotations(self):
        """Create sample annotation data."""
        return {
            "images": {
                "image1.jpg": {
                    "words": {
                        "word1": {
                            "points": [[10, 10], [50, 10], [50, 30], [10, 30]]
                        },
                        "word2": {
                            "points": [[60, 10], [100, 10], [100, 30], [60, 30]]
                        }
                    }
                },
                "image2.jpg": {
                    "words": {}
                }
            }
        }

    @pytest.fixture
    def create_sample_images(self, temp_dir):
        """Create sample images for testing."""
        # Create sample images
        img1_path = temp_dir / "image1.jpg"
        img2_path = temp_dir / "image2.jpg"
        img3_path = temp_dir / "image3.png"  # Extra image without annotation

        # Create dummy images
        img = Image.new('RGB', (100, 100), color='red')
        img.save(img1_path)
        img.save(img2_path)
        img.save(img3_path)

        return [img1_path, img2_path, img3_path]

    def test_dataset_with_annotations(self, temp_dir, sample_annotations, create_sample_images):
        """Test dataset loading with annotation file."""
        # Create annotation file
        annotation_path = temp_dir / "annotations.json"
        with open(annotation_path, 'w') as f:
            json.dump(sample_annotations, f)

        # Create dataset
        transform = Mock()
        dataset = OCRDataset(temp_dir, annotation_path, transform)

        # Verify dataset properties
        assert len(dataset) == 2  # image1.jpg and image2.jpg
        assert "image1.jpg" in dataset.anns
        assert "image2.jpg" in dataset.anns
        assert "image3.png" not in dataset.anns  # No annotation for image3

        # Verify polygons for image1
        polygons = dataset.anns["image1.jpg"]
        assert len(polygons) == 2
        assert isinstance(polygons[0], np.ndarray)
        assert polygons[0].shape == (1, 4, 2)  # One polygon with 4 points

        # Verify image2 has no polygons
        assert dataset.anns["image2.jpg"] is None

    def test_dataset_without_annotations(self, temp_dir, create_sample_images):
        """Test dataset loading without annotation file."""
        transform = Mock()
        dataset = OCRDataset(temp_dir, None, transform)

        # Should include all images
        assert len(dataset) == 3
        assert all(ann is None for ann in dataset.anns.values())

    def test_getitem(self, temp_dir, sample_annotations, create_sample_images):
        """Test __getitem__ method."""
        # Create annotation file
        annotation_path = temp_dir / "annotations.json"
        with open(annotation_path, 'w') as f:
            json.dump(sample_annotations, f)

        transform = Mock()
        transform.return_value = {"image": "transformed_image", "polygons": "transformed_polygons", "inverse_matrix": "inverse_matrix"}

        dataset = OCRDataset(temp_dir, annotation_path, transform)

        # Get first item
        item = dataset[0]

        # Verify transform was called
        transform.assert_called_once()

        # Verify the call arguments
        call_args, call_kwargs = transform.call_args
        assert 'image' in call_kwargs
        assert isinstance(call_kwargs['image'], np.ndarray)  # numpy array
        assert call_kwargs['image'].shape == (100, 100, 3)

    def test_missing_image_file(self, temp_dir):
        """Test behavior when annotation references non-existent image."""
        annotations = {
            "images": {
                "missing.jpg": {
                    "words": {
                        "word1": {"points": [[0, 0], [10, 10], [10, 0]]}
                    }
                }
            }
        }

        annotation_path = temp_dir / "annotations.json"
        with open(annotation_path, 'w') as f:
            json.dump(annotations, f)

        transform = Mock()
        dataset = OCRDataset(temp_dir, annotation_path, transform)

        # Should be empty since image doesn't exist
        assert len(dataset) == 0

    def test_empty_words_in_annotation(self, temp_dir, create_sample_images):
        """Test handling of words with empty points."""
        annotations = {
            "images": {
                "image1.jpg": {
                    "words": {
                        "word1": {"points": []},  # Empty points
                        "word2": {"points": [[0, 0], [10, 10]]},  # Valid points
                        "word3": {"points": []}  # Another empty
                    }
                }
            }
        }

        annotation_path = temp_dir / "annotations.json"
        with open(annotation_path, 'w') as f:
            json.dump(annotations, f)

        transform = Mock()
        dataset = OCRDataset(temp_dir, annotation_path, transform)

        # Should have only one polygon (word2)
        polygons = dataset.anns["image1.jpg"]
        assert len(polygons) == 1

    def test_image_extensions(self, temp_dir):
        """Test that dataset recognizes different image extensions."""
        # Create images with different extensions
        extensions = ['jpg', 'jpeg', 'png']
        for ext in extensions:
            img_path = temp_dir / f"test.{ext}"
            img = Image.new('RGB', (50, 50), color='blue')
            img.save(img_path)

        transform = Mock()
        dataset = OCRDataset(temp_dir, None, transform)

        # Should find all three images
        assert len(dataset) == 3
        expected_files = {f"test.{ext}" for ext in extensions}
        assert set(dataset.anns.keys()) == expected_files

    def test_case_insensitive_extension_matching(self, temp_dir):
        """Test that extension matching is case insensitive."""
        # Create image with uppercase extension
        img_path = temp_dir / "test.JPG"
        img = Image.new('RGB', (50, 50), color='green')
        img.save(img_path)

        transform = Mock()
        dataset = OCRDataset(temp_dir, None, transform)

        # Should find the image
        assert len(dataset) == 1
        assert "test.JPG" in dataset.anns