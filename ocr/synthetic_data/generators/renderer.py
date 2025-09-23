# Text Rendering Module
"""
Text rendering on images for synthetic data.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

from ..models import TextRegion


class TextRenderer:
    """Renderer for synthetic text on images."""

    def __init__(self, config: Optional[DictConfig] = None):
        """Initialize text renderer.

        Args:
            config: Configuration for text rendering
        """
        self.config: Union[DictConfig, Dict[str, Any]] = config or {}
        self.font_cache: dict[tuple[str, int], "ImageFont.FreeTypeFont"] = {}

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
        angle: float = 0.0,
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
            angle=angle,
        )


__all__ = ["TextRenderer"]
