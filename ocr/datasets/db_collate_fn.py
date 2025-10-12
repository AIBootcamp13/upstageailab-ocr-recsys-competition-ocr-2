"""
*****************************************************************************************
* Modified from https://github.com/MhLiao/DB/blob/master/data/make_seg_detector_data.py
*
* 참고 논문:
* Real-time Scene Text Detection with Differentiable Binarization
* https://arxiv.org/pdf/1911.08947.pdf
*
* 참고 Repository:
* https://github.com/MhLiao/DB/
*****************************************************************************************
"""

from collections import OrderedDict

import cv2
import numpy as np
import pyclipper
import torch


class DBCollateFN:
    """
    Collate function for DB text detection model batches.

    Handles polygon shape normalization and validation to ensure robust batch processing.
    Supports both pre-computed and on-the-fly probability/threshold map generation.

    Shape Contracts:
    - Input polygons: List[List[np.ndarray]] where each polygon is (N, 2) or (1, N, 2)
    - Output prob_maps: (batch_size, 1, H, W) tensor
    - Output thresh_maps: (batch_size, 1, H, W) tensor
    - Output polygons: List[List[np.ndarray]] with validated (N, 2) polygons

    Validation:
    - Filters invalid polygons (wrong shape, degenerate, zero area)
    - Normalizes polygon shapes to (N, 2) format
    - Logs filtering statistics for monitoring
    """

    def __init__(self, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.inference_mode = False

    def __call__(self, batch):
        """
        Collate a batch of DB detection samples.

        Args:
            batch: List of sample dictionaries with keys:
                - "image": torch.Tensor (C, H, W)
                - "metadata": Optional metadata payload (dict or Pydantic model)
                - "image_filename": str (legacy fallback)
                - "image_path": str (legacy fallback)
                - "inverse_matrix": np.ndarray (3, 3)
                - "polygons": List[np.ndarray] - polygon coordinates
                - "prob_map": np.ndarray (H, W) - optional pre-computed
                - "thresh_map": np.ndarray (H, W) - optional pre-computed

        Returns:
            OrderedDict with collated batch data:
                - "images": torch.Tensor (batch_size, C, H, W)
                - "polygons": List[List[np.ndarray]] - validated polygons
                - "prob_maps": torch.Tensor (batch_size, 1, H, W)
                - "thresh_maps": torch.Tensor (batch_size, 1, H, W)
                - Additional metadata fields
        """
        images = [item["image"] for item in batch]
        metadata_entries = [self._extract_metadata(item) for item in batch]

        filenames = []
        image_paths = []
        raw_sizes = []
        orientations = []
        canonical_sizes = []

        for idx, (item, metadata) in enumerate(zip(batch, metadata_entries, strict=True)):
            # Filename/path primarily come from metadata, fall back to legacy keys
            filename = metadata.get("filename") or item.get("image_filename") or f"sample_{idx}"
            filenames.append(str(filename))

            raw_path = metadata.get("path")
            if raw_path is None:
                raw_path = item.get("image_path", "")
            image_paths.append(str(raw_path))

            raw_sizes.append(metadata.get("raw_size", item.get("raw_size")))
            orientation_value = metadata.get("orientation", item.get("orientation"))
            if orientation_value is None:
                orientation_value = 1
            orientations.append(int(orientation_value))

            # Canonical size: prefer metadata canonical size, otherwise derive from tensor
            canonical_size = metadata.get("canonical_size")
            if canonical_size is None:
                canonical_size = item.get("shape")
            if canonical_size is None:
                tensor = item["image"]
                if isinstance(tensor, torch.Tensor):
                    canonical_size = (int(tensor.shape[-2]), int(tensor.shape[-1]))
                else:
                    canonical_size = None
            canonical_sizes.append(canonical_size)

        inverse_matrix = [item["inverse_matrix"] for item in batch]

        collated_batch = OrderedDict(
            images=torch.stack(images, dim=0),
            image_filename=filenames,
            image_path=image_paths,
            inverse_matrix=inverse_matrix,
            shape=canonical_sizes,
            raw_size=raw_sizes,
            orientation=orientations,
            canonical_size=canonical_sizes,
        )

        collated_batch["metadata"] = metadata_entries

        if self.inference_mode:
            return collated_batch

        # Load pre-processed maps from batch items
        polygons = [item.get("polygons", []) for item in batch]

        # Validate and filter invalid polygons
        polygons = self._validate_batch_polygons(polygons, filenames)

        prob_maps = []
        thresh_maps = []

        # Track map loading statistics
        preloaded_count = 0
        fallback_count = 0

        for i, item in enumerate(batch):
            # Check if pre-processed maps exist in the item
            if item.get("prob_map") is not None and item.get("thresh_map") is not None:
                # Use pre-loaded maps
                prob_map = torch.from_numpy(item["prob_map"]) if isinstance(item["prob_map"], np.ndarray) else item["prob_map"]
                thresh_map = torch.from_numpy(item["thresh_map"]) if isinstance(item["thresh_map"], np.ndarray) else item["thresh_map"]
                preloaded_count += 1
            else:
                # Fallback: generate maps on-the-fly if pre-processed maps are missing
                segmentations = self.make_prob_thresh_map(images[i], polygons[i], filenames[i])
                prob_map = torch.tensor(segmentations["prob_map"])
                thresh_map = torch.tensor(segmentations["thresh_map"])
                fallback_count += 1

            prob_maps.append(prob_map)
            thresh_maps.append(thresh_map)

        # Log map loading statistics (only once per epoch to avoid spam)
        if not hasattr(self, "_logged_stats"):
            from rich.console import Console

            console = Console()
            total_samples = len(batch)
            preloaded_pct = (preloaded_count / total_samples) * 100

            if preloaded_count > 0:
                console.print(
                    f"[green]✓ Using .npz maps (from cache or disk): {preloaded_count}/{total_samples} samples ({preloaded_pct:.1f}%)[/green]"
                )
            if fallback_count > 0:
                console.print(
                    f"[yellow]⚠ Fallback to on-the-fly generation: {fallback_count}/{total_samples} samples ({100 - preloaded_pct:.1f}%)[/yellow]"
                )

            self._logged_stats = True

        collated_batch.update(
            polygons=polygons,
            prob_maps=self._stack_maps_with_channel_dim(prob_maps),  # Ensure (B, 1, H, W) shape
            thresh_maps=self._stack_maps_with_channel_dim(thresh_maps),  # Ensure (B, 1, H, W) shape
        )

        return collated_batch

    @staticmethod
    def _extract_metadata(sample):
        metadata = sample.get("metadata")
        if metadata is None:
            return {}
        if hasattr(metadata, "model_dump"):
            try:
                return metadata.model_dump()
            except Exception:
                return dict(metadata)
        if isinstance(metadata, dict):
            return metadata
        return {}

    @staticmethod
    def _stack_maps_with_channel_dim(maps_list):
        """
        Stack a list of maps ensuring they have the channel dimension.

        Args:
            maps_list: List of tensors with shapes (H, W) or (1, H, W)

        Returns:
            Tensor with shape (B, 1, H, W)
        """
        if not maps_list:
            raise ValueError("Cannot stack empty maps list")

        # Stack the maps first
        stacked = torch.stack(maps_list, dim=0)  # Shape: (B, ...) where ... is H,W or 1,H,W

        # Check if we need to add the channel dimension
        if stacked.ndim == 3:  # Shape is (B, H, W)
            return stacked.unsqueeze(1)  # Add channel dim: (B, 1, H, W)
        elif stacked.ndim == 4 and stacked.shape[1] == 1:  # Shape is (B, 1, H, W)
            return stacked  # Already has channel dimension
        else:
            raise ValueError(f"Unexpected map shape after stacking: {stacked.shape}. Expected (B, H, W) or (B, 1, H, W)")

    def make_prob_thresh_map(self, image, polygons, filename):
        _, h, w = image.shape

        prob_map = np.zeros((h, w), dtype=np.float32)
        thresh_map = np.zeros((h, w), dtype=np.float32)

        for poly in polygons:
            # Ensure polygon is in correct format (N, 2) not (1, N, 2)
            # Some code paths return (N, 2), others return (1, N, 2)
            if poly.ndim == 3 and poly.shape[0] == 1:
                poly = poly[0]  # Remove batch dimension: (1, N, 2) -> (N, 2)

            # Calculate the distance and polygons
            poly = poly.astype(np.int32)
            # Polygon point가 3개 미만이라면 skip
            if poly.size < 3:
                continue

            # https://arxiv.org/pdf/1911.08947.pdf 참고
            area = cv2.contourArea(poly)
            if area <= 0:
                # Degenerate polygon (line or a point) – skip to avoid zero-area artifacts
                continue

            L = cv2.arcLength(poly, True)
            if L <= 0:
                continue

            eps = np.finfo(float).eps
            D = area * (1 - self.shrink_ratio**2) / (L + eps)
            if D <= eps:
                continue
            pco = pyclipper.PyclipperOffset()  # type: ignore[attr-defined]
            # pyclipper expects list of polygons, so wrap in list: [[N, 2]]
            pco.AddPaths([poly], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)  # type: ignore[attr-defined]

            # Probability map 생성
            shrinked = pco.Execute(-D)
            for s in shrinked:
                shrinked_poly = np.array(s)
                cv2.fillPoly(prob_map, [shrinked_poly], 1.0)  # type: ignore[arg-type]

            # Threshold map 생성
            dilated = pco.Execute(D)
            for d in dilated:
                dilated_poly = np.array(d)

                xmin = dilated_poly[:, 0].min()
                xmax = dilated_poly[:, 0].max()
                ymin = dilated_poly[:, 1].min()
                ymax = dilated_poly[:, 1].max()
                width = xmax - xmin + 1
                height = ymax - ymin + 1

                # poly is already (N, 2) at this point, no need to index
                polygon = poly.copy()
                polygon[:, 0] = polygon[:, 0] - xmin
                polygon[:, 1] = polygon[:, 1] - ymin

                xs = np.broadcast_to(
                    np.linspace(0, width - 1, num=width).reshape(1, width),
                    (height, width),
                )
                ys = np.broadcast_to(
                    np.linspace(0, height - 1, num=height).reshape(height, 1),
                    (height, width),
                )

                distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
                for i in range(polygon.shape[0]):
                    j = (i + 1) % polygon.shape[0]
                    absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
                    distance_map[i] = np.clip(absolute_distance / D, 0, 1)
                distance_map = distance_map.min(axis=0)

                xmin_valid = min(max(0, xmin), thresh_map.shape[1] - 1)
                xmax_valid = min(max(0, xmax), thresh_map.shape[1] - 1)
                ymin_valid = min(max(0, ymin), thresh_map.shape[0] - 1)
                ymax_valid = min(max(0, ymax), thresh_map.shape[0] - 1)

                thresh_map[ymin_valid : ymax_valid + 1, xmin_valid : xmax_valid + 1] = np.fmax(
                    1
                    - distance_map[
                        ymin_valid - ymin : ymax_valid - ymax + height,
                        xmin_valid - xmin : xmax_valid - xmax + width,
                    ],  # noqa
                    thresh_map[ymin_valid : ymax_valid + 1, xmin_valid : xmax_valid + 1],
                )

        # Normalize the threshold map
        thresh_map = thresh_map * (self.thresh_max - self.thresh_min) + self.thresh_min

        result = OrderedDict(prob_map=prob_map, thresh_map=thresh_map)

        return result

    def _validate_batch_polygons(self, batch_polygons, filenames):
        """
        Validate polygons in a batch and filter out invalid ones.

        Performs comprehensive validation of polygon data including:
        - Shape normalization: (1, N, 2) → (N, 2)
        - Geometric validation: minimum points, positive area
        - Data integrity: finite values, correct dimensions

        Args:
            batch_polygons: List of polygon lists, one per sample
            filenames: List of filenames for logging

        Returns:
            List of validated polygon lists with invalid polygons removed.
            Each polygon is normalized to (N, 2) shape.
        """
        validated_batch = []

        for sample_idx, (polygons, filename) in enumerate(zip(batch_polygons, filenames, strict=True)):
            validated_polygons = []

            for poly_idx, poly in enumerate(polygons):
                try:
                    # Handle different polygon shapes
                    if poly.ndim == 3 and poly.shape[0] == 1:
                        poly = poly[0]  # Normalize (1, N, 2) -> (N, 2)

                    # Check basic shape requirements
                    if poly.ndim != 2 or poly.shape[1] != 2:
                        print(f"⚠ Skipping invalid polygon shape {poly.shape} in {filename}, polygon {poly_idx}")
                        continue

                    # Check minimum points for a valid polygon
                    if poly.shape[0] < 3:
                        print(f"⚠ Skipping degenerate polygon with {poly.shape[0]} points in {filename}, polygon {poly_idx}")
                        continue

                    # Check for valid numeric values
                    if not np.isfinite(poly).all():
                        print(f"⚠ Skipping polygon with invalid values in {filename}, polygon {poly_idx}")
                        continue

                    # Additional validation: check area is positive
                    if poly.shape[0] >= 3:
                        try:
                            area = cv2.contourArea(poly.astype(np.float32))
                            if area <= 0:
                                print(f"⚠ Skipping polygon with non-positive area {area:.2f} in {filename}, polygon {poly_idx}")
                                continue
                        except Exception as e:
                            print(f"⚠ Error calculating area for polygon in {filename}, polygon {poly_idx}: {e}")
                            continue

                    # Polygon is valid
                    validated_polygons.append(poly)

                except Exception as e:
                    print(f"⚠ Error validating polygon in {filename}, polygon {poly_idx}: {e}")
                    continue

            validated_batch.append(validated_polygons)

            # Log if polygons were filtered
            original_count = len(polygons)
            validated_count = len(validated_polygons)
            if validated_count < original_count:
                print(f"⚠ Filtered {original_count - validated_count} invalid polygons in {filename} ({validated_count} remaining)")

        return validated_batch

    def distance(self, xs, ys, point_1, point_2):
        """
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        """
        height, width = xs.shape[:2]
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1]) + np.finfo(float).eps

        denom = 2 * np.sqrt(square_distance_1 * square_distance_2) + np.finfo(float).eps
        cosin = (square_distance - square_distance_1 - square_distance_2) / denom
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
        # self.extend_line(point_1, point_2, result)
        return result
