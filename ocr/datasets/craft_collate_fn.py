"""Collate function for generating CRAFT-style supervision maps."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

import cv2
import numpy as np
import torch


class CraftCollateFN:
    """Prepare training batches for CRAFT-style text detection.

    This collate function produces region and affinity probability maps that mimic
    the supervision signals used by the CRAFT architecture. Because the competition
    dataset only provides word-level polygons, we approximate the character heatmaps
    by applying Gaussian smoothing and morphological dilation to each polygon.
    """

    def __init__(
        self,
        region_blur_scale: float = 0.35,
        affinity_kernel_ratio: float = 0.25,
        affinity_blur_scale: float = 0.15,
        min_text_area: int = 9,
    ) -> None:
        self.region_blur_scale = region_blur_scale
        self.affinity_kernel_ratio = affinity_kernel_ratio
        self.affinity_blur_scale = affinity_blur_scale
        self.min_text_area = int(max(min_text_area, 1))
        self.inference_mode = False

    def __call__(self, batch: Iterable[OrderedDict]):
        images = [item["image"] for item in batch]
        filenames = [item["image_filename"] for item in batch]
        inverse_matrices = [item["inverse_matrix"] for item in batch]

        collated_batch = OrderedDict(
            images=torch.stack(images, dim=0),
            image_filename=filenames,
            inverse_matrix=inverse_matrices,
        )

        if self.inference_mode:
            return collated_batch

        polygons_batch = [item["polygons"] for item in batch]

        region_maps = []
        affinity_maps = []
        for image, polygons in zip(images, polygons_batch, strict=False):
            region_map, affinity_map = self._generate_maps(image, polygons)
            region_maps.append(torch.from_numpy(region_map).unsqueeze(0))
            affinity_maps.append(torch.from_numpy(affinity_map).unsqueeze(0))

        collated_batch.update(
            polygons=polygons_batch,
            prob_maps=torch.stack(region_maps, dim=0),
            thresh_maps=torch.stack(affinity_maps, dim=0),
            region_maps=torch.stack(region_maps, dim=0),
            affinity_maps=torch.stack(affinity_maps, dim=0),
        )
        return collated_batch

    def _generate_maps(
        self,
        image: torch.Tensor,
        polygons: list[np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        _, height, width = image.shape
        region_map = np.zeros((height, width), dtype=np.float32)
        affinity_map = np.zeros((height, width), dtype=np.float32)

        if not polygons:
            return region_map, affinity_map

        for polygon in polygons:
            if polygon is None:
                continue
            squeezed = polygon.reshape(-1, 2).astype(np.int32)
            if squeezed.shape[0] < 3:
                continue

            bbox_w = max(float(squeezed[:, 0].max() - squeezed[:, 0].min()), 1.0)
            bbox_h = max(float(squeezed[:, 1].max() - squeezed[:, 1].min()), 1.0)
            area = bbox_w * bbox_h
            if area < self.min_text_area:
                continue

            polygon_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(polygon_mask, [squeezed], color=(1,))

            # Region heatmap: apply adaptive Gaussian smoothing inside the polygon
            sigma = max(bbox_w, bbox_h) * self.region_blur_scale
            if sigma > 0:
                region_heatmap = cv2.GaussianBlur(polygon_mask.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
                max_value = float(region_heatmap.max())
                if max_value > 0:
                    region_heatmap = region_heatmap / max_value
            else:
                region_heatmap = polygon_mask.astype(np.float32)
            region_map = np.maximum(region_map, region_heatmap)

            # Affinity heatmap: dilate the polygon to encourage connectivity
            kernel_size = int(max(bbox_w, bbox_h) * self.affinity_kernel_ratio)
            kernel_size = max(kernel_size // 2 * 2 + 1, 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated = cv2.dilate(polygon_mask, kernel)

            if self.affinity_blur_scale > 0:
                sigma_aff = kernel_size * self.affinity_blur_scale
                affinity_heatmap = cv2.GaussianBlur(dilated.astype(np.float32), (0, 0), sigmaX=sigma_aff, sigmaY=sigma_aff)
                max_value = float(affinity_heatmap.max())
                if max_value > 0:
                    affinity_heatmap = affinity_heatmap / max_value
            else:
                affinity_heatmap = dilated.astype(np.float32)
            affinity_map = np.maximum(affinity_map, affinity_heatmap)

        np.clip(region_map, 0.0, 1.0, out=region_map)
        np.clip(affinity_map, 0.0, 1.0, out=affinity_map)
        return region_map, affinity_map
