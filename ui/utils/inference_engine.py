from __future__ import annotations

"""Compatibility wrapper for the modular inference toolkit."""

import logging

from .inference import InferenceEngine, get_available_checkpoints, run_inference_on_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

__all__ = [
    "InferenceEngine",
    "run_inference_on_image",
    "get_available_checkpoints",
]
