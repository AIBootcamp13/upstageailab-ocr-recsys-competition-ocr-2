from __future__ import annotations

"""Image preprocessing helpers."""

import logging
from collections.abc import Callable
from typing import Any

import cv2

from .config_loader import PreprocessSettings
from .dependencies import torch, transforms

LOGGER = logging.getLogger(__name__)


def build_transform(settings: PreprocessSettings):
    """Create a torchvision transform pipeline from preprocessing settings."""
    if transforms is None:
        raise RuntimeError("Torchvision transforms are not available. Install the vision extras.")

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(settings.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=settings.normalization.mean, std=settings.normalization.std),
        ]
    )


def preprocess_image(image: Any, transform: Callable[[Any], Any]) -> Any:
    """Apply preprocessing transform to an image and return a batched tensor."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = transform(image_rgb)
    if torch is None:
        raise RuntimeError("Torch is not available to create inference batches.")
    return tensor.unsqueeze(0)
