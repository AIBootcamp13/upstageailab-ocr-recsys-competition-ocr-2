"""Image enhancement components."""

from __future__ import annotations

import cv2
import numpy as np


class ImageEnhancer:
    """General image enhancement strategies."""

    def enhance(self, image: np.ndarray, method: str) -> tuple[np.ndarray, list[str]]:
        if method == "office_lens":
            return self._enhance_image_office_lens(image)
        return self._enhance_image(image)

    def _enhance_image(self, image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        enhanced = image.copy()
        applied_enhancements = ["clahe_mild", "bilateral_filter_mild"]

        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        enhanced = cv2.bilateralFilter(enhanced, 5, 25, 25)

        return enhanced, applied_enhancements

    def _enhance_image_office_lens(self, image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        enhanced = image.copy()
        applied_enhancements = [
            "gamma_correction",
            "clahe_lab",
            "saturation_boost",
            "sharpening",
            "noise_reduction",
        ]

        gamma = 1.1
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)

        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
        l_channel = clahe.apply(l)
        lab = cv2.merge([l_channel, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        s_clean = cv2.bilateralFilter(s, 9, 75, 75)
        s = cv2.addWeighted(s, 1.2, s_clean, 0.3, 0)
        s = np.clip(s, 20, 230).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        blurred = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
        final = cv2.bilateralFilter(sharpened, 9, 75, 75)

        return final, applied_enhancements


__all__ = ["ImageEnhancer"]
