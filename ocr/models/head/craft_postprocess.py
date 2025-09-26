"""Post-processing utilities for the CRAFT text detector."""

from __future__ import annotations

import cv2
import numpy as np


class CraftPostProcessor:
    """Convert CRAFT score maps into polygons."""

    def __init__(
        self,
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.3,
        min_area: int = 16,
        expand_ratio: float = 1.5,
    ) -> None:
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.min_area = max(int(min_area), 1)
        self.expand_ratio = max(expand_ratio, 1.0)

    def represent(self, batch, pred) -> tuple[list[list[list[int]]], list[list[float]]]:
        region_scores = pred["region_score"] if isinstance(pred, dict) else pred
        affinity_scores = pred.get("affinity_score") if isinstance(pred, dict) else None
        if affinity_scores is None:
            raise KeyError("CRAFT head predictions must include 'affinity_score'.")

        region_scores = region_scores.detach().cpu().numpy()
        affinity_scores = affinity_scores.detach().cpu().numpy()
        inverse_matrices = batch["inverse_matrix"]

        boxes_batch: list[list[list[int]]] = []
        scores_batch: list[list[float]] = []
        for region_score, affinity_score, inverse_matrix in zip(region_scores, affinity_scores, inverse_matrices, strict=False):
            boxes, scores = self._process_single(region_score[0], affinity_score[0], inverse_matrix)
            boxes_batch.append(boxes)
            scores_batch.append(scores)

        return boxes_batch, scores_batch

    def _process_single(
        self,
        region_score: np.ndarray,
        affinity_score: np.ndarray,
        inverse_matrix: np.ndarray,
    ) -> tuple[list[list[int]], list[float]]:
        text_map = region_score.copy()
        link_map = affinity_score.copy()

        # Apply thresholds and connected component analysis
        text_mask = text_map >= self.low_text
        link_mask = link_map >= self.link_threshold
        combined_mask = np.logical_or(text_mask, link_mask).astype(np.uint8)
        if not combined_mask.any():
            return [], []

        num_labels, labels = cv2.connectedComponents(combined_mask, connectivity=4)
        boxes: list[list[int]] = []
        scores: list[float] = []

        for label_idx in range(1, num_labels):
            component_mask = labels == label_idx
            if component_mask.sum() < self.min_area:
                continue

            score = float(text_map[component_mask].mean())
            if score < self.text_threshold:
                continue

            component_uint8 = component_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(component_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            if contour.shape[0] < 3:
                continue

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            if self.expand_ratio != 1.0:
                box = self._expand_box(box, self.expand_ratio)

            box = self._transform_coordinates(box, inverse_matrix)
            boxes.append(box.astype(np.int32).tolist())
            scores.append(score)

        return boxes, scores

    @staticmethod
    def _expand_box(points: np.ndarray, ratio: float) -> np.ndarray:
        center = points.mean(axis=0)
        return (points - center) * ratio + center

    @staticmethod
    def _transform_coordinates(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        transformed = (matrix @ homogeneous.T).T
        transformed /= transformed[:, 2:3]
        return transformed[:, :2]
