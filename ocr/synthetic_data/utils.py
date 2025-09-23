# Synthetic Data Utilities
"""
Utility functions and augmentation setup for synthetic data generation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

try:
    import albumentations as A

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None

from ocr.utils.logging import logger

from .dataset import SyntheticDatasetGenerator


def create_synthetic_dataset(
    num_images: int = 1000,
    output_dir: str = "data/synthetic",
    config: Optional[DictConfig] = None,
) -> List[Dict[str, Any]]:
    """Convenience function to create synthetic dataset.

    Args:
        num_images: Number of images to generate
        output_dir: Output directory
        config: Generation configuration

    Returns:
        Dataset entries
    """
    generator = SyntheticDatasetGenerator(config)
    return generator.generate_dataset(num_images, output_dir)


def augment_existing_dataset(
    source_dir: str,
    output_dir: str,
    augmentation_factor: int = 5,
    config: Optional[DictConfig] = None,
) -> List[Dict[str, Any]]:
    """Convenience function to augment existing dataset.

    Args:
        source_dir: Source dataset directory
        output_dir: Output directory for augmented data
        augmentation_factor: Augmentation factor
        config: Augmentation configuration

    Returns:
        Augmented dataset entries
    """
    # Load source dataset (simplified - would need proper dataset loading)
    source_entries: list[dict] = []  # Would load actual dataset

    generator = SyntheticDatasetGenerator(config)
    return generator.generate_augmented_dataset(
        source_entries, output_dir, augmentation_factor
    )


def setup_augmentation_pipeline(config: Optional[DictConfig] = None) -> Optional[Any]:
    """Setup augmentation pipeline for synthetic data.

    Args:
        config: Augmentation configuration

    Returns:
        Albumentations compose object or None if not available
    """
    if not ALBUMENTATIONS_AVAILABLE:
        logger.warning("Albumentations not available, skipping augmentations")
        return None

    # Default augmentation transforms
    transforms = [
        A.Rotate(limit=10, p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    ]

    # Add custom transforms from config if available
    if config and hasattr(config, "augmentation"):
        aug_config = config.augmentation
        if hasattr(aug_config, "additional_transforms"):
            # Would parse additional transforms from config
            pass

    return A.Compose(transforms)


__all__ = [
    "create_synthetic_dataset",
    "augment_existing_dataset",
    "setup_augmentation_pipeline",
]
