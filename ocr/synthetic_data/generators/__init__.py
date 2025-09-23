# Generators Package
"""
Synthetic data generators for text, backgrounds, and rendering.
"""

from .background import BackgroundGenerator
from .renderer import TextRenderer
from .text import TextGenerator

__all__ = ["TextGenerator", "BackgroundGenerator", "TextRenderer"]
