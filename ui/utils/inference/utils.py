from __future__ import annotations

"""Utility helpers for inference."""

import logging
from collections.abc import Sequence

import numpy as np

from .dependencies import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)


def get_available_checkpoints() -> list[str]:
    outputs_dir = PROJECT_ROOT / "outputs"
    if not outputs_dir.exists():
        return ["No 'outputs' directory found"]

    checkpoints = [str(path.relative_to(PROJECT_ROOT)) for path in outputs_dir.rglob("*.ckpt")]
    return checkpoints or ["No checkpoints found in 'outputs' directory"]


def generate_mock_predictions() -> dict[str, object]:
    LOGGER.info("Generating mock predictions.")
    # Competition format uses space-separated coordinates
    return {
        "polygons": "100 100 300 100 300 180 100 180|350 250 600 250 600 300 350 300",
        "texts": ["Mock Text 1", "Mock Text 2"],
        "confidences": [0.98, 0.95],
    }


def ensure_three_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    return image[..., :3] if image.shape[2] == 4 else image


def format_polygon(points: Sequence[int]) -> str:
    """Format polygon points as space-separated string (competition format)."""
    return " ".join(map(str, points))
