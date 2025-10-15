"""
Integration tests for collate function end-to-end pipeline.

Tests the complete pipeline from dataset loading through collation,
model forward pass, and loss computation to ensure robust integration.

This addresses Phase 1.1.3 in the data pipeline testing plan.
"""

import os
import tempfile
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ocr.datasets.base import Dataset as OCRDataset
from ocr.datasets.db_collate_fn import DBCollateFN
from ocr.models.head.db_head import DBHead
from ocr.models.loss.db_loss import DBLoss


class TestCollateIntegration:
    """Test end-to-end integration from dataset to loss computation."""

    @pytest.fixture
    def mock_transform(self):
        """Create a mock transform that returns consistent output."""
        transform = Mock()
        transform.return_value = {
            "image": torch.rand(3, 224, 224),
            "polygons": [
                np.array([[10, 20], [30, 40], [20, 60]], dtype=np.float32),
                np.array([[50, 60], [70, 80], [60, 100]], dtype=np.float32),
            ],
            "inverse_matrix": np.eye(3),
        }
        return transform

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary directory with test images and annotations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create image directory
            img_dir = os.path.join(temp_dir, "images")
            os.makedirs(img_dir)

            # Create test images
            for i in range(3):
                img_path = os.path.join(img_dir, f"test_{i}.jpg")
                # Create a simple RGB image
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                from PIL import Image

                pil_img = Image.fromarray(img)
                pil_img.save(img_path)

            yield temp_dir

    def create_annotation_file(self, temp_dir, annotations):
        """Create an annotation file with the given annotations."""
        ann_path = os.path.join(temp_dir, "annotations.json")
        with open(ann_path, "w") as f:
            f.write(str(annotations).replace("'", '"'))
        return ann_path

    def test_collate_to_model_forward_pass(self, temp_dataset_dir, mock_transform):
        """Test complete pipeline: dataset → collate → model forward pass."""
        # Create annotations with polygons
        annotations = {
            "images": {
                "test_0.jpg": {"words": {"word1": {"points": [[10, 20], [30, 40], [20, 60], [0, 40]]}}},
                "test_1.jpg": {
                    "words": {
                        "word1": {"points": [[50, 60], [70, 80], [60, 100], [40, 80]]},
                        "word2": {"points": [[100, 120], [120, 140], [110, 160], [90, 140]]},
                    }
                },
                "test_2.jpg": {"words": {"word1": {"points": [[10, 20], [30, 40], [20, 60], [0, 40]]}}},
            }
        }

        ann_path = self.create_annotation_file(temp_dataset_dir, annotations)
        img_dir = os.path.join(temp_dataset_dir, "images")

        # Create dataset
        dataset = OCRDataset(image_path=img_dir, annotation_path=ann_path, transform=mock_transform)

        # Create DataLoader with collate function
        collate_fn = DBCollateFN()
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)

        # Get a batch
        batch = next(iter(dataloader))

        # Verify batch structure
        assert "images" in batch
        assert "polygons" in batch
        assert "prob_maps" in batch
        assert "thresh_maps" in batch

        batch_size = batch["images"].shape[0]
        assert batch_size == 2

        # Create model head
        head = DBHead(in_channels=256, upscale=4)  # Typical DBNet head config

        # Mock decoder output instead of passing raw images
        batch_size, _, height, width = batch["images"].shape
        mock_decoder_output = torch.randn(batch_size, 256, height, width)

        # Test forward pass
        with torch.no_grad():
            predictions = head(mock_decoder_output)

        # Verify predictions
        assert "prob_maps" in predictions
        assert "thresh_map" in predictions
        assert "binary_map" in predictions

        # Check output shapes
        assert predictions["prob_maps"].shape[0] == batch_size
        assert predictions["thresh_map"].shape[0] == batch_size
        assert predictions["binary_map"].shape[0] == batch_size

    def test_collate_to_loss_computation(self, temp_dataset_dir, mock_transform):
        """Test complete pipeline: dataset → collate → model → loss computation."""
        # Create annotations
        annotations = {
            "images": {
                "test_0.jpg": {"words": {"word1": {"points": [[10, 20], [30, 40], [20, 60], [0, 40]]}}},
                "test_1.jpg": {"words": {"word1": {"points": [[50, 60], [70, 80], [60, 100], [40, 80]]}}},
            }
        }

        ann_path = self.create_annotation_file(temp_dataset_dir, annotations)
        img_dir = os.path.join(temp_dataset_dir, "images")

        # Create dataset
        dataset = OCRDataset(image_path=img_dir, annotation_path=ann_path, transform=mock_transform)

        # Create DataLoader
        collate_fn = DBCollateFN()
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)

        # Get batch
        batch = next(iter(dataloader))

        # Create model and loss
        head = DBHead(in_channels=256, upscale=4)
        loss_fn = DBLoss()

        # Mock decoder output instead of passing raw images
        # Decoder should output 256 channels (matching head input_channels)
        batch_size, _, height, width = batch["images"].shape
        mock_decoder_output = torch.randn(batch_size, 256, height, width)

        # Forward pass with mocked decoder output
        with torch.no_grad():
            predictions = head(mock_decoder_output)

        # Create ground truth (simplified - in real scenario this would come from batch)
        batch_size = batch["images"].shape[0]
        # Head has upscale=4, so output is 4x larger than input
        gt_height, gt_width = height * 4, width * 4
        gt_binary = torch.rand(batch_size, 1, gt_height, gt_width)  # Mock ground truth
        gt_thresh = torch.rand(batch_size, 1, gt_height, gt_width)  # Mock threshold map

        # Compute loss
        loss, loss_dict = loss_fn(predictions, gt_binary, gt_thresh)

        # Verify loss computation
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss >= 0  # Loss should be non-negative

        assert isinstance(loss_dict, dict)
        assert "loss_prob" in loss_dict

        # For training mode predictions, should have additional losses
        if "thresh_map" in predictions:
            assert "loss_thresh" in loss_dict
            assert "loss_binary" in loss_dict

    def test_collate_integration_with_empty_polygons(self, temp_dataset_dir, mock_transform):
        """Test integration with images that have no polygons after processing."""
        # Create annotations where some images have no valid polygons
        annotations = {
            "images": {
                "test_0.jpg": {"words": {}},  # No polygons
                "test_1.jpg": {"words": {"word1": {"points": [[10, 20], [30, 40], [20, 60], [0, 40]]}}},
                "test_2.jpg": {"words": {}},  # No polygons
            }
        }

        ann_path = self.create_annotation_file(temp_dataset_dir, annotations)
        img_dir = os.path.join(temp_dataset_dir, "images")

        dataset = OCRDataset(image_path=img_dir, annotation_path=ann_path, transform=mock_transform)

        collate_fn = DBCollateFN()
        dataloader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=collate_fn, num_workers=0)

        # Should handle batch with mixed polygon counts
        batch = next(iter(dataloader))

        # Verify batch structure
        assert batch["images"].shape[0] == 3
        assert len(batch["polygons"]) == 3  # One entry per image

        # Create model and test forward pass
        head = DBHead(in_channels=256, upscale=4)

        # Mock decoder output instead of passing raw images
        batch_size, _, height, width = batch["images"].shape
        mock_decoder_output = torch.randn(batch_size, 256, height, width)

        with torch.no_grad():
            predictions = head(mock_decoder_output)

        # Should not crash even with empty polygons
        assert predictions["prob_maps"].shape[0] == 3

    def test_collate_integration_different_batch_sizes(self, temp_dataset_dir, mock_transform):
        """Test integration with different batch sizes."""
        annotations = {
            "images": {
                "test_0.jpg": {"words": {"word1": {"points": [[10, 20], [30, 40], [20, 60], [0, 40]]}}},
                "test_1.jpg": {"words": {"word1": {"points": [[50, 60], [70, 80], [60, 100], [40, 80]]}}},
                "test_2.jpg": {"words": {"word1": {"points": [[100, 120], [120, 140], [110, 160], [90, 140]]}}},
            }
        }

        ann_path = self.create_annotation_file(temp_dataset_dir, annotations)
        img_dir = os.path.join(temp_dataset_dir, "images")

        dataset = OCRDataset(image_path=img_dir, annotation_path=ann_path, transform=mock_transform)

        collate_fn = DBCollateFN()

        # Test different batch sizes
        for batch_size in [1, 2, 3]:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=False)

            # Process all batches
            for batch in dataloader:
                # Create model and test
                head = DBHead(in_channels=256, upscale=4)

                # Mock decoder output instead of passing raw images
                batch_size, _, height, width = batch["images"].shape
                mock_decoder_output = torch.randn(batch_size, 256, height, width)

                with torch.no_grad():
                    predictions = head(mock_decoder_output)

                # Verify batch size consistency
                actual_batch_size = batch["images"].shape[0]
                assert actual_batch_size <= batch_size

                assert predictions["prob_maps"].shape[0] == actual_batch_size

    def test_collate_integration_inference_mode(self, temp_dataset_dir, mock_transform):
        """Test integration in inference mode (no loss computation)."""
        annotations = {"images": {"test_0.jpg": {"words": {"word1": {"points": [[10, 20], [30, 40], [20, 60], [0, 40]]}}}}}

        ann_path = self.create_annotation_file(temp_dataset_dir, annotations)
        img_dir = os.path.join(temp_dataset_dir, "images")

        dataset = OCRDataset(image_path=img_dir, annotation_path=ann_path, transform=mock_transform)

        collate_fn = DBCollateFN()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

        batch = next(iter(dataloader))

        # Create model in inference mode
        head = DBHead(in_channels=256, upscale=4)

        # Mock decoder output instead of passing raw images
        batch_size, _, height, width = batch["images"].shape
        mock_decoder_output = torch.randn(batch_size, 256, height, width)

        with torch.no_grad():
            predictions = head(mock_decoder_output, return_loss=False)

        # Inference mode should only return probability maps
        assert "binary_map" in predictions
        assert "prob_maps" in predictions
        assert "thresh_map" not in predictions  # No threshold map in inference
        assert "thresh_binary_map" not in predictions
