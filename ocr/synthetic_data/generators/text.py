# Text Generation Module
"""
Text content generation for synthetic data.
"""

import random
from typing import Any

from omegaconf import DictConfig


class TextGenerator:
    """Generator for synthetic text content."""

    def __init__(self, config: DictConfig | None = None):
        """Initialize text generator.

        Args:
            config: Configuration for text generation
        """
        self.config: DictConfig | dict[str, Any] = config or {}
        self.words = self._load_word_list()
        self.fonts = self._load_fonts()

    def _load_word_list(self) -> list[str]:
        """Load list of words for text generation."""
        # Default word list - can be extended with custom dictionaries
        default_words = [
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "lazy",
            "dog",
            "hello",
            "world",
            "text",
            "recognition",
            "artificial",
            "intelligence",
            "machine",
            "learning",
            "computer",
            "vision",
            "deep",
            "neural",
            "network",
            "convolutional",
            "recurrent",
            "transformer",
            "attention",
            "mechanism",
        ]
        return default_words

    def _load_fonts(self) -> list[str]:
        """Load available fonts for text rendering."""
        # Default fonts - can be extended with custom font paths
        default_fonts = [
            "DejaVuSans-Bold.ttf",
            "DejaVuSans.ttf",
            "DejaVuSerif-Bold.ttf",
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

    def generate_paragraph(self, num_lines: int = 3) -> list[str]:
        """Generate a paragraph of text.

        Args:
            num_lines: Number of lines in paragraph

        Returns:
            List of text lines
        """
        return [self.generate_text_line() for _ in range(num_lines)]


__all__ = ["TextGenerator"]
