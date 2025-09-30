"""Padding cleanup component."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .external import doctr_remove_image_padding


class PaddingCleanup:
    """Optional padding cleanup leveraging docTR."""

    def __init__(self, ensure_doctr: Callable[[str], bool]) -> None:
        self.ensure_doctr = ensure_doctr

    def cleanup(self, image: np.ndarray) -> np.ndarray | None:
        if not self.ensure_doctr("padding_cleanup"):
            return None
        if doctr_remove_image_padding is None:
            return None
        cleaned = doctr_remove_image_padding(image)
        return cleaned if cleaned.size != 0 and cleaned.shape != image.shape else None


__all__ = ["PaddingCleanup"]
