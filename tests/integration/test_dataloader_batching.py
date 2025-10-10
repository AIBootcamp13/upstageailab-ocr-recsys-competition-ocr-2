"""
Integration tests for DataLoader batching robustness.

Tests DataLoader behavior with variable polygon counts, empty images,
and edge cases to ensure robust batch processing across different data variations.

This addresses part of Phase 3.2 in the data pipeline testing plan.
"""

import os
import tempfile
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image


class TestDataLoaderBatching:
    """Test DataLoader batching with various data variations."""

    @pytest.fixture
    def mock_transform(self):
        """Create a mock transform that returns consistent output."""
        transform = Mock()
        transform.return_value = {"image": torch.rand(3, 224, 224), "polygons": [], "inverse_matrix": np.eye(3)}
        return transform

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary directory with test images and annotations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create image directory
            img_dir = os.path.join(temp_dir, "images")
            os.makedirs(img_dir)

            # Create test images
            for i in range(5):
                img_path = os.path.join(img_dir, f"test_{i}.jpg")
                img = Image.new("RGB", (100, 100))
                img.save(img_path)

            yield temp_dir

    def create_annotation_file(self, temp_dir, annotations):
        """Create an annotation file with the given annotations."""
        ann_path = os.path.join(temp_dir, "annotations.json")
        with open(ann_path, "w") as f:
            f.write(str(annotations).replace("'", '"'))
        return ann_path
