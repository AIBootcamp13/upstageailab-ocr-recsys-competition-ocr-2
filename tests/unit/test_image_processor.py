"""Unit tests for the ImageProcessor utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from ocr.lightning_modules.processors import ImageProcessor


def test_tensor_to_pil_image_rgb_tensor():
    """tensor_to_pil_image converts a simple tensor into an RGB image."""
    tensor = torch.ones((3, 2, 2), dtype=torch.float32) * 0.5
    pil_image = ImageProcessor.tensor_to_pil_image(tensor)
    assert isinstance(pil_image, Image.Image)
    assert pil_image.size == (2, 2)
    assert pil_image.mode == "RGB"


def test_tensor_to_pil_image_applies_statistics():
    """Normalization statistics are applied when supplied."""
    tensor = torch.zeros((3, 1, 1), dtype=torch.float32)
    mean = np.array([0.5, 0.25, 0.75], dtype=np.float32)
    std = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    pil_image = ImageProcessor.tensor_to_pil_image(tensor, mean=mean, std=std)
    pixels = np.array(pil_image) / 255.0

    # After normalization: tensor * std + mean = 0 * std + mean = mean
    # Then clipped to [0, 1] and converted to uint8 and back to float
    # So the expected value is mean after clipping
    expected = np.clip(mean, 0.0, 1.0)

    # The actual result after denormalization should be close to the mean values
    # Increased tolerance to account for precision loss from uint8 conversion
    assert np.allclose(pixels.reshape(-1, 3)[0], expected, atol=5e-3)


def test_tensor_to_pil_image_invalid_shape_raises():
    """Unsupported tensor shapes trigger a ValueError."""
    tensor = torch.randn(2, 2)
    with pytest.raises(ValueError):
        ImageProcessor.tensor_to_pil_image(tensor)


def test_prepare_wandb_image_resizes_and_preserves_mode():
    """prepare_wandb_image resizes large images and ensures RGB output."""
    pil_image = Image.new("L", (1024, 512), color=128)
    processed = ImageProcessor.prepare_wandb_image(pil_image, max_side=256)
    assert processed.size[0] <= 256 and processed.size[1] <= 256
    assert processed.mode == "RGB"


def test_prepare_wandb_image_no_resize_when_smaller():
    """Images smaller than max_side are returned as-is (or RGB-cloned)."""
    pil_image = Image.new("RGB", (64, 32), color=(10, 20, 30))
    processed = ImageProcessor.prepare_wandb_image(pil_image, max_side=256)
    assert processed.size == (64, 32)
    assert processed.mode == "RGB"
