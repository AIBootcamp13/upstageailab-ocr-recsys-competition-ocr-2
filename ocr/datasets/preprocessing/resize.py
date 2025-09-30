"""Image resizing utility."""

from __future__ import annotations

import cv2
import numpy as np


class FinalResizer:
    """Resize helper honouring target dimension constraints."""

    def resize(self, image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        target_width, target_height = target_size
        height, width = image.shape[:2]
        scale = min(target_width / width, target_height / height)

        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        pad_left = (target_width - new_width) // 2
        pad_top = (target_height - new_height) // 2
        pad_right = target_width - new_width - pad_left
        pad_bottom = target_height - new_height - pad_top

        return cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )


__all__ = ["FinalResizer"]
