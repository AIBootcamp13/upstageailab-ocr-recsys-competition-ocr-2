# Background Generation Module
"""
Background image generation for synthetic data.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig


class BackgroundGenerator:
    """Generator for synthetic image backgrounds."""

    def __init__(self, config: Optional[DictConfig] = None):
        """Initialize background generator.

        Args:
            config: Configuration for background generation
        """
        self.config: Union[DictConfig, Dict[str, Any]] = config or {}

    def generate_plain_background(self, size: Tuple[int, int], color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """Generate plain color background.

        Args:
            size: Image size (width, height)
            color: Background color (R, G, B)

        Returns:
            Background image array
        """
        if color is None:
            gray_value = random.randint(200, 255)
            color = (gray_value, gray_value, gray_value)  # Light gray to white

        image = np.full((*size[::-1], 3), color, dtype=np.uint8)
        return image

    def generate_gradient_background(self, size: Tuple[int, int], colors: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
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
            color = [int(colors[0][i] + factor * (colors[1][i] - colors[0][i])) for i in range(3)]
            image[y, :, :] = color

        return image

    def generate_noise_background(self, size: Tuple[int, int], intensity: float = 0.1) -> np.ndarray:
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


__all__ = ["BackgroundGenerator"]
