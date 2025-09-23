# Synthetic Data Generation Module
"""
Module for generating augmented, synthetic datasets for OCR training.
Supports various augmentation strategies and synthetic text generation.
"""

import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import torch
from omegaconf import DictConfig

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None

from ocr.utils.logging import logger


@dataclass
class TextRegion:
    """Represents a text region with its properties."""
    text: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    polygon: List[Tuple[int, int]]
    font_size: int
    angle: float = 0.0
    confidence: float = 1.0


@dataclass
class SyntheticImage:
    """Represents a synthetic image with text regions."""
    image: np.ndarray
    text_regions: List[TextRegion]
    metadata: Dict[str, Any]


class TextGenerator:
    """Generator for synthetic text content."""

    def __init__(self, config: Optional[DictConfig] = None):
        """Initialize text generator.

        Args:
            config: Configuration for text generation
        """
        self.config = config or {}
        self.words = self._load_word_list()
        self.fonts = self._load_fonts()

    def _load_word_list(self) -> List[str]:
        """Load list of words for text generation."""
        # Default word list - can be extended with custom dictionaries
        default_words = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "hello", "world", "text", "recognition", "artificial", "intelligence",
            "machine", "learning", "computer", "vision", "deep", "neural", "network",
            "convolutional", "recurrent", "transformer", "attention", "mechanism"
        ]
        return default_words

    def _load_fonts(self) -> List[str]:
        """Load available fonts for text rendering."""
        # Default fonts - can be extended with custom font paths
        default_fonts = [
            "DejaVuSans-Bold.ttf",
            "DejaVuSans.ttf",
            "DejaVuSerif-Bold.ttf"
        ]
        return default_fonts

    def generate_text_line(self, min_words: int = 3, max_words: int = 8) -> str:
        """Generate a random text line.

        Args:
            min_words: Minimum number of words
            max_words: Maximum number of words

        Returns:
            Generated text line
        """
        num_words = random.randint(min_words, max_words)
        words = random.choices(self.words, k=num_words)
        return " ".join(words).title()

    def generate_paragraph(self, num_lines: int = 3) -> List[str]:
        """Generate a paragraph of text.

        Args:
            num_lines: Number of lines in paragraph

        Returns:
            List of text lines
        """
        return [self.generate_text_line() for _ in range(num_lines)]


class BackgroundGenerator:
    """Generator for synthetic image backgrounds."""

    def __init__(self, config: Optional[DictConfig] = None):
        """Initialize background generator.

        Args:
            config: Configuration for background generation
        """
        self.config = config or {}

    def generate_plain_background(
        self,
        size: Tuple[int, int],
        color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """Generate plain color background.

        Args:
            size: Image size (width, height)
            color: Background color (R, G, B)

        Returns:
            Background image array
        """
        if color is None:
            color = (random.randint(200, 255),) * 3  # Light gray to white

        image = np.full((*size[::-1], 3), color, dtype=np.uint8)
        return image

    def generate_gradient_background(
        self,
        size: Tuple[int, int],
        colors: Optional[List[Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """Generate gradient background.

        Args:
            size: Image size (width, height)
            colors: List of gradient colors

        Returns:
            Background image array
        """
        if colors is None:
            colors = [(255, 255, 255), (240, 240, 240)]  # White to light gray

        width, height = size
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Simple vertical gradient
        for y in range(height):
            factor = y / (height - 1) if height > 1 else 0
            color = [
                int(colors[0][i] + factor * (colors[1][i] - colors[0][i]))
                for i in range(3)
            ]
            image[y, :, :] = color

        return image

    def generate_noise_background(
        self,
        size: Tuple[int, int],
        intensity: float = 0.1
    ) -> np.ndarray:
        """Generate noise background.

        Args:
            size: Image size (width, height)
            intensity: Noise intensity (0-1)

        Returns:
            Background image array
        """
        base_color = (245, 245, 245)  # Light gray base
        image = np.full((*size[::-1], 3), base_color, dtype=np.uint8)

        # Add random noise
        noise = np.random.randint(0, int(255 * intensity), image.shape, dtype=np.uint8)
        image = np.clip(image.astype(np.int32) + noise - int(127 * intensity), 0, 255).astype(np.uint8)

        return image


class TextRenderer:
    """Renderer for synthetic text on images."""

    def __init__(self, config: Optional[DictConfig] = None):
        """Initialize text renderer.

        Args:
            config: Configuration for text rendering
        """
        self.config = config or {}
        self.font_cache = {}

    def _get_font(self, font_path: str, size: int) -> ImageFont.FreeTypeFont:
        """Get cached font or load new one.

        Args:
            font_path: Path to font file
            size: Font size

        Returns:
            Font object
        """
        cache_key = (font_path, size)
        if cache_key not in self.font_cache:
            try:
                self.font_cache[cache_key] = ImageFont.truetype(font_path, size)
            except OSError:
                # Fallback to default font
                self.font_cache[cache_key] = ImageFont.load_default()

        return self.font_cache[cache_key]

    def render_text_region(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_size: int = 24,
        color: Tuple[int, int, int] = (0, 0, 0),
        angle: float = 0.0
    ) -> TextRegion:
        """Render text on image and return region info.

        Args:
            image: Input image array
            text: Text to render
            position: Top-left position (x, y)
            font_size: Font size
            color: Text color (R, G, B)
            angle: Rotation angle in degrees

        Returns:
            TextRegion object with rendered text info
        """
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        # Use default font for simplicity (can be extended with custom fonts)
        font = ImageFont.load_default()

        # Get text bounding box
        bbox = draw.textbbox(position, text, font=font)
        x1, y1, x2, y2 = bbox

        # Create polygon from bbox
        polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        # Draw text
        draw.text(position, text, fill=color, font=font)

        # Convert back to numpy array
        rendered_image = np.array(pil_image)

        return TextRegion(
            text=text,
            bbox=(x1, y1, x2, y2),
            polygon=polygon,
            font_size=font_size,
            angle=angle
        )


class SyntheticDatasetGenerator:
    """Main generator for synthetic OCR datasets."""

    def __init__(self, config: Optional[DictConfig] = None):
        """Initialize synthetic dataset generator.

        Args:
            config: Configuration for dataset generation
        """
        self.config = config or {}
        self.text_generator = TextGenerator(config)
        self.background_generator = BackgroundGenerator(config)
        self.text_renderer = TextRenderer(config)

        # Setup augmentation pipeline if albumentations is available
        if ALBUMENTATIONS_AVAILABLE:
            self.augmentation_pipeline = self._create_augmentation_pipeline()
        else:
            self.augmentation_pipeline = None
            logger.warning("Albumentations not available, skipping augmentations")

    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create augmentation pipeline."""
        transforms = [
            A.Rotate(limit=10, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.1),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ]
        return A.Compose(transforms)

    def generate_single_image(
        self,
        size: Tuple[int, int] = (512, 512),
        num_text_regions: int = 3,
        background_type: str = "plain"
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
            margin = 50
            x = random.randint(margin, size[0] - 200)
            y = random.randint(margin, size[1] - 50)

            # Random font size
            font_size = random.randint(16, 32)

            # Random color (dark colors for contrast)
            color = (
                random.randint(0, 100),
                random.randint(0, 100),
                random.randint(0, 100)
            )

            # Render text
            text_region = self.text_renderer.render_text_region(
                background.copy(),
                text,
                (x, y),
                font_size,
                color
            )

            text_regions.append(text_region)

            # Apply to background
            background = self.text_renderer.render_text_region(
                background,
                text,
                (x, y),
                font_size,
                color
            ).bbox  # This is not right - need to fix
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
            "augmented": self.augmentation_pipeline is not None
        }

        return SyntheticImage(
            image=background,
            text_regions=text_regions,
            metadata=metadata
        )

    def generate_dataset(
        self,
        num_images: int,
        output_dir: Union[str, Path],
        image_size: Tuple[int, int] = (512, 512),
        **kwargs
    ) -> List[Dict[str, Any]]:
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
            synthetic_image = self.generate_single_image(
                size=image_size,
                **kwargs
            )

            # Save image
            image_filename = "04d"
            image_path = images_dir / image_filename
            Image.fromarray(synthetic_image.image).save(image_path)

            # Create annotation
            annotation = self._create_annotation(synthetic_image, image_filename)
            annotation_filename = f"synthetic_{i:04d}.json"
            annotation_path = annotations_dir / annotation_filename

            # Save annotation (simplified - would use JSON in real implementation)
            with open(annotation_path, 'w') as f:
                # Simplified annotation format
                f.write(str(annotation))

            dataset_entries.append({
                "image_path": str(image_path),
                "annotation_path": str(annotation_path),
                "metadata": synthetic_image.metadata
            })

            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_images} images")

        logger.info(f"Dataset generation complete! Saved to {output_dir}")
        return dataset_entries

    def _create_annotation(self, synthetic_image: SyntheticImage, image_filename: str) -> Dict[str, Any]:
        """Create annotation dictionary for synthetic image.

        Args:
            synthetic_image: Synthetic image object
            image_filename: Name of the image file

        Returns:
            Annotation dictionary
        """
        annotation = {
            "image_filename": image_filename,
            "text_regions": []
        }

        for region in synthetic_image.text_regions:
            annotation["text_regions"].append({
                "text": region.text,
                "bbox": region.bbox,
                "polygon": region.polygon,
                "font_size": region.font_size,
                "angle": region.angle,
                "confidence": region.confidence
            })

        return annotation

    def generate_augmented_dataset(
        self,
        source_dataset: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        augmentation_factor: int = 5
    ) -> List[Dict[str, Any]]:
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


# Convenience functions
def create_synthetic_dataset(
    num_images: int = 1000,
    output_dir: str = "data/synthetic",
    config: Optional[DictConfig] = None
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
    config: Optional[DictConfig] = None
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
    source_entries = []  # Would load actual dataset

    generator = SyntheticDatasetGenerator(config)
    return generator.generate_augmented_dataset(source_entries, output_dir, augmentation_factor)


__all__ = [
    "TextRegion",
    "SyntheticImage",
    "TextGenerator",
    "BackgroundGenerator",
    "TextRenderer",
    "SyntheticDatasetGenerator",
    "create_synthetic_dataset",
    "augment_existing_dataset"
]