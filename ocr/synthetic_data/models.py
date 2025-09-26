# Synthetic Data Models
"""
Data models and structures for synthetic data generation.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass
class TextRegion:
    """Represents a text region with its properties."""

    text: str
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    polygon: list[tuple[int, int]]
    font_size: int
    angle: float = 0.0
    confidence: float = 1.0


@dataclass
class SyntheticImage:
    """Represents a synthetic image with text regions."""

    image: "np.ndarray"  # Forward reference to avoid numpy import
    text_regions: list[TextRegion]
    metadata: dict[str, Any]


__all__ = ["TextRegion", "SyntheticImage"]
