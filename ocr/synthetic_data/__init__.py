# Synthetic Data Generation Package
"""
Modular synthetic data generation for OCR training.

This package provides tools for generating synthetic images with text
for training and augmenting OCR models. It's designed to be modular
and extensible for future enhancements.
"""

from .dataset import SyntheticDatasetGenerator
from .generators import BackgroundGenerator, TextGenerator, TextRenderer
from .models import SyntheticImage, TextRegion
from .utils import (
    augment_existing_dataset,
    create_synthetic_dataset,
    setup_augmentation_pipeline,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "SyntheticDatasetGenerator",
    "TextGenerator",
    "BackgroundGenerator",
    "TextRenderer",
    # Data models
    "SyntheticImage",
    "TextRegion",
    # Utilities
    "create_synthetic_dataset",
    "augment_existing_dataset",
    "setup_augmentation_pipeline",
]
