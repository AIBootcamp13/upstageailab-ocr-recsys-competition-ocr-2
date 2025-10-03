# Synthetic Dataset Generation Module
"""
Main orchestrator for synthetic OCR dataset generation.
"""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

try:
    import albumentations as A

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None

from ocr.utils.logging import logger

from .generators import BackgroundGenerator, TextGenerator, TextRenderer
from .models import SyntheticImage


class SyntheticDatasetGenerator:
    """Main generator for synthetic OCR datasets."""

    def __init__(self, config: DictConfig | None = None):
        """Initialize synthetic dataset generator.

        Args:
            config: Configuration for dataset generation
        """
        self.config: DictConfig | dict[str, Any] = config or {}
        self.text_generator = TextGenerator(config)
        self.background_generator = BackgroundGenerator(config)
        self.text_renderer = TextRenderer(config)

        # Setup augmentation pipeline if albumentations is available
        if ALBUMENTATIONS_AVAILABLE:
            self.augmentation_pipeline = self._create_augmentation_pipeline()
        else:
            self.augmentation_pipeline = None
            logger.warning("Albumentations not available, skipping augmentations")

    def _create_augmentation_pipeline(self) -> Any:
        """Create augmentation pipeline."""
        if not ALBUMENTATIONS_AVAILABLE or A is None:
            return None
        transforms = [
            A.Rotate(limit=10, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.1),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ]
        return A.Compose(transforms)

    def generate_single_image(
        self,
        size: tuple[int, int] = (512, 512),
        num_text_regions: int = 3,
        background_type: str = "plain",
    ) -> SyntheticImage:
        """Generate a single synthetic image with text.

        Args:
            size: Image size (width, height)
            num_text_regions: Number of text regions to generate
            background_type: Type of background ('plain', 'gradient', 'noise')

        Returns:
            SyntheticImage object
        """
        # Generate background
        if background_type == "gradient":
            background = self.background_generator.generate_gradient_background(size)
        elif background_type == "noise":
            background = self.background_generator.generate_noise_background(size)
        else:
            background = self.background_generator.generate_plain_background(size)

        text_regions = []

        for _ in range(num_text_regions):
            # Generate text
            text = self.text_generator.generate_text_line()

            # Random position
            width, height = size
            margin_x = min(50, max(0, width // 6))
            margin_y = min(50, max(0, height // 6))
            x_min = margin_x
            x_max = width - margin_x - 1
            if x_max < x_min:
                x_min, x_max = 0, max(width - 1, 0)
            y_min = margin_y
            y_max = height - margin_y - 1
            if y_max < y_min:
                y_min, y_max = 0, max(height - 1, 0)
            x = random.randint(x_min, x_max if x_max >= x_min else x_min)
            y = random.randint(y_min, y_max if y_max >= y_min else y_min)

            # Random font size
            font_size = random.randint(16, 32)

            # Random color (dark colors for contrast)
            color = (
                random.randint(0, 100),
                random.randint(0, 100),
                random.randint(0, 100),
            )

            # Render text
            text_region = self.text_renderer.render_text_region(background.copy(), text, (x, y), font_size, color)

            text_regions.append(text_region)

            # Apply to background
            # Actually, we need to draw on the image
            pil_image = Image.fromarray(background)
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.load_default()
            draw.text((x, y), text, fill=color, font=font)
            background = np.array(pil_image)

        # Apply augmentations if available
        if self.augmentation_pipeline:
            augmented = self.augmentation_pipeline(image=background)
            background = augmented["image"]

        metadata = {
            "size": size,
            "num_text_regions": num_text_regions,
            "background_type": background_type,
            "augmented": self.augmentation_pipeline is not None,
        }

        return SyntheticImage(image=background, text_regions=text_regions, metadata=metadata)

    def generate_dataset(
        self,
        num_images: int,
        output_dir: str | Path,
        image_size: tuple[int, int] = (512, 512),
        image_prefix: str = "synthetic",
        image_format: str = "png",
        annotation_format: str = "json",
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Generate a complete synthetic dataset.

        Args:
            num_images: Number of images to generate
            output_dir: Output directory for dataset
            image_size: Size of generated images
            **kwargs: Additional generation parameters

        Returns:
            List of dataset entries
        """
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        annotations_dir = output_dir / "annotations"

        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)

        dataset_entries = []

        logger.info(f"Generating {num_images} synthetic images...")

        for i in range(num_images):
            # Generate synthetic image
            synthetic_image = self.generate_single_image(size=image_size, **kwargs)

            # Determine filenames
            image_filename = f"{image_prefix}_{i:04d}.{image_format.strip('.')}"
            annotation_filename = f"{image_prefix}_{i:04d}.{annotation_format.strip('.')}"

            # Save image
            image_path = images_dir / image_filename
            Image.fromarray(synthetic_image.image).save(image_path)

            # Create annotation
            annotation = self._create_annotation(synthetic_image, image_filename)
            annotation_path = annotations_dir / annotation_filename

            # Persist annotation as JSON
            with annotation_path.open("w", encoding="utf-8") as fp:
                json.dump(annotation, fp, ensure_ascii=False, indent=2)

            dataset_entries.append(
                {
                    "image_path": str(image_path),
                    "annotation_path": str(annotation_path),
                    "metadata": synthetic_image.metadata,
                }
            )

            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_images} images")

        logger.info(f"Dataset generation complete! Saved to {output_dir}")
        return dataset_entries

    def _create_annotation(self, synthetic_image: SyntheticImage, image_filename: str) -> dict[str, Any]:
        """Create annotation dictionary for synthetic image.

        Args:
            synthetic_image: Synthetic image object
            image_filename: Name of the image file

        Returns:
            Annotation dictionary
        """
        annotation: dict[str, Any] = {
            "image_filename": image_filename,
            "text_regions": [],
        }

        for region in synthetic_image.text_regions:
            annotation["text_regions"].append(
                {
                    "text": region.text,
                    "bbox": region.bbox,
                    "polygon": region.polygon,
                    "font_size": region.font_size,
                    "angle": region.angle,
                    "confidence": region.confidence,
                }
            )

        return annotation

    def generate_augmented_dataset(
        self,
        source_dataset: list[dict[str, Any]],
        output_dir: str | Path,
        augmentation_factor: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate augmented versions of existing dataset.

        Args:
            source_dataset: Source dataset entries
            output_dir: Output directory
            augmentation_factor: Number of augmented versions per image

        Returns:
            List of augmented dataset entries
        """
        output_dir = Path(output_dir)
        augmented_entries = []

        logger.info(f"Generating augmented dataset with factor {augmentation_factor}...")

        for entry in source_dataset:
            image_path = Path(entry["image_path"])

            if not image_path.exists():
                continue

            # Load original image
            original_image = np.array(Image.open(image_path))

            for aug_idx in range(augmentation_factor):
                # Apply augmentations
                if self.augmentation_pipeline:
                    augmented = self.augmentation_pipeline(image=original_image)
                    aug_image = augmented["image"]
                else:
                    aug_image = original_image.copy()

                # Save augmented image
                aug_filename = f"{image_path.stem}_aug_{aug_idx}{image_path.suffix}"
                aug_path = output_dir / "images" / aug_filename
                aug_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(aug_image).save(aug_path)

                # Copy annotation (assuming same text regions)
                aug_entry = entry.copy()
                aug_entry["image_path"] = str(aug_path)
                aug_entry["metadata"]["augmented"] = True
                aug_entry["metadata"]["augmentation_index"] = aug_idx

                augmented_entries.append(aug_entry)

        logger.info(f"Augmented dataset generation complete! Created {len(augmented_entries)} images")
        return augmented_entries


__all__ = ["SyntheticDatasetGenerator"]
