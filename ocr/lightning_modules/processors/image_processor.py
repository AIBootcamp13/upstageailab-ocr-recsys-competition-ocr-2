"""Image processing helpers used by the OCR Lightning module."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


class ImageProcessor:
    """Utility functions for converting tensors to images and resizing outputs."""

    @staticmethod
    def tensor_to_pil_image(
        tensor: torch.Tensor,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> Image.Image:
        """Convert a model tensor into a display-ready PIL Image."""
        if tensor.ndim not in (3, 4):
            raise ValueError("Expected tensor with 3 or 4 dimensions for image conversion.")

        working_tensor = tensor
        if working_tensor.ndim == 4:
            working_tensor = working_tensor.squeeze(0)

        working_tensor = working_tensor.detach().cpu()
        if working_tensor.ndim != 3:
            raise ValueError("Expected a 3D tensor after squeezing for image conversion.")

        if working_tensor.shape[0] in (1, 3):
            array = working_tensor.permute(1, 2, 0).numpy()
        elif working_tensor.shape[2] in (1, 3):
            array = working_tensor.numpy()
        else:
            raise ValueError("Unsupported tensor shape for image conversion.")

        if array.dtype != np.float32:
            array = array.astype(np.float32)

        if isinstance(mean, np.ndarray) and isinstance(std, np.ndarray) and mean.size == std.size:
            if mean.size == array.shape[2]:
                array = array * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)

        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)

        array = np.clip(array, 0.0, 1.0)
        array = (array * 255).astype(np.uint8)
        return Image.fromarray(array)

    @staticmethod
    def prepare_wandb_image(pil_image: Image.Image, max_side: int | None) -> Image.Image:
        """Resize the PIL image for W&B logging without mutating the original."""
        image = pil_image
        if max_side is not None and max_side > 0:
            width, height = pil_image.size
            if width > max_side or height > max_side:
                image = pil_image.copy()
                resampling: Any
                try:
                    resampling = Image.Resampling.LANCZOS
                except AttributeError:  # Pillow<9 compatibility
                    resampling = getattr(Image, "LANCZOS", None)
                    if resampling is None:
                        resampling = Image.BICUBIC
                image.thumbnail((max_side, max_side), resampling)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
