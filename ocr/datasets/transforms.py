from collections import OrderedDict
from typing import Any

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from pydantic import ValidationError

from ocr.datasets.schemas import ImageMetadata, PolygonData, TransformInput, TransformOutput
from ocr.utils.geometry_utils import calculate_cropbox, calculate_inverse_transform


class ConditionalNormalize(A.ImageOnlyTransform):
    """
    Normalize image only if it hasn't been pre-normalized.

    This transform checks if the image is already normalized (float32 with values around 0)
    and skips normalization if so. This allows pre-normalized images from cache to skip
    this expensive operation.
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def apply(self, img, **params):
        # Check if image is already normalized (float32 dtype is a good indicator)
        if img.dtype == np.float32 and img.max() < 10.0:
            # Image is already normalized, return as-is
            return img

        # Image is uint8, need to normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return img

    def get_transform_init_args_names(self):
        return ("mean", "std")


class ValidatedDBTransforms:
    def __init__(self, transforms, keypoint_params):
        self.transform = A.Compose([*transforms, ToTensorV2()], keypoint_params=keypoint_params)

    def __call__(self, data: TransformInput | dict[str, Any] | np.ndarray, polygons: list[np.ndarray] | None = None) -> OrderedDict:
        """
        Apply transforms to image and polygons.

        Args:
            data: TransformInput payload or raw numpy image for backwards compatibility
            polygons: Legacy list of polygon arrays when using legacy signature

        Returns:
            OrderedDict with:
                - image: Transformed image tensor
                - polygons: List of transformed polygons with shape (1, N, 2)
                - inverse_matrix: Matrix for coordinate transformation
        """
        transform_input = self._coerce_input(data, polygons)

        image = transform_input.image
        polygon_models = transform_input.polygons or []
        metadata_payload: dict[str, Any] | None = None
        if transform_input.metadata is not None:
            metadata_payload = transform_input.metadata.model_dump()
            if metadata_payload.get("path") is not None:  # type: ignore
                metadata_payload["path"] = str(metadata_payload["path"])  # type: ignore

        height, width = image.shape[:2]

        keypoints = [point for poly in polygon_models for point in poly.points.reshape(-1, 2)]
        keypoints = self.clamp_keypoints(keypoints, width, height)

        # Image transform / Geometric transform의 경우 keypoints를 변환
        transformed = self.transform(image=image, keypoints=keypoints)
        transformed_image = transformed["image"]
        transformed_keypoints = transformed["keypoints"]
        metadata = transformed.get("metadata")

        if metadata is not None and not isinstance(metadata, dict):
            try:
                metadata = dict(metadata)
            except Exception:
                metadata = {"metadata": metadata}

        if metadata_payload is not None:
            if metadata is None:
                metadata = metadata_payload
            else:
                combined_metadata = metadata_payload.copy()
                combined_metadata.update(metadata)
                metadata = combined_metadata

        # Keypoints 재변환을 위한 Matrix 계산
        _, new_height, new_width = transformed_image.shape
        crop_box = calculate_cropbox((width, height), max(new_height, new_width))
        inverse_matrix = calculate_inverse_transform((width, height), (new_width, new_height), crop_box=crop_box)

        # Keypoints 정보를 Polygons 형태로 변환
        # BUG FIX (BUG-2025-004): Correct polygon point count extraction
        transformed_polygons = []
        index = 0
        if polygon_models:
            for polygon_idx, polygon in enumerate(polygon_models):
                num_points = polygon.points.shape[0]

                keypoint_slice = transformed_keypoints[index : index + num_points]
                index += len(keypoint_slice)

                if len(keypoint_slice) < 3:
                    import logging

                    logging.debug(
                        "Skipping degenerate polygon at index %d: only %d points after transform",
                        polygon_idx,
                        len(keypoint_slice),
                    )
                    continue

                polygon_array = np.asarray(keypoint_slice, dtype=np.float32).reshape(1, -1, 2)
                transformed_polygons.append(polygon_array)

        output = OrderedDict(
            image=transformed_image,
            polygons=transformed_polygons,
            inverse_matrix=inverse_matrix,
        )

        if metadata is not None:
            output["metadata"] = metadata

        validated_output = TransformOutput.model_validate(output)

        result: OrderedDict[str, Any] = OrderedDict(
            image=validated_output.image,
            polygons=validated_output.polygons,
            inverse_matrix=validated_output.inverse_matrix,
        )

        if validated_output.metadata is not None:
            result["metadata"] = validated_output.metadata

        return result

    def clamp_keypoints(self, keypoints: list, img_width: int, img_height: int) -> list:
        clamped_keypoints = []
        for kp in keypoints:
            x, y = kp[:2]
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            clamped_keypoints.append((x, y) + tuple(kp[2:]))
        return clamped_keypoints

    def _coerce_input(
        self,
        data: TransformInput | dict[str, Any] | np.ndarray,
        polygons: list[np.ndarray] | None,
    ) -> TransformInput:
        if isinstance(data, TransformInput):
            return data

        if isinstance(data, np.ndarray):
            return self._build_from_legacy(image=data, polygons=polygons)

        try:
            return TransformInput.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"Invalid transform input payload: {exc}") from exc

    def _build_from_legacy(self, image: np.ndarray, polygons: list[np.ndarray] | None) -> TransformInput:
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            image = np.array(image)

        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array or PIL Image, got {type(image)}")

        polygon_models = None
        if polygons is not None:
            if not isinstance(polygons, list):
                raise TypeError(f"polygons must be a list, got {type(polygons)}")
            # Filter out invalid polygons before creating PolygonData objects
            valid_polygons = []
            for idx, poly in enumerate(polygons):
                if not isinstance(poly, np.ndarray):
                    raise TypeError(f"polygon at index {idx} must be numpy array, got {type(poly)}")

                if poly.dtype not in (np.float32, np.float64, np.int32, np.int64):
                    raise TypeError(f"polygon at index {idx} must be numeric array, got dtype {poly.dtype}")

                if poly.ndim not in (2, 3):
                    raise ValueError(f"polygon at index {idx} must be 2D or 3D array, got {poly.ndim}D with shape {poly.shape}")

                if poly.ndim == 2 and poly.shape[1] != 2:
                    raise ValueError(f"polygon at index {idx} must have shape (N, 2), got {poly.shape}")

                if poly.ndim == 3 and (poly.shape[0] != 1 or poly.shape[2] != 2):
                    raise ValueError(f"polygon at index {idx} must have shape (1, N, 2), got {poly.shape}")
                try:
                    # Attempt to create PolygonData - this will validate the polygon
                    valid_polygons.append(PolygonData(points=poly))
                except ValidationError:
                    # Skip invalid polygons that fail validation
                    continue
            polygon_models = valid_polygons

        if image.ndim not in (2, 3):
            raise ValueError(f"Image must be 2D or 3D array, got shape {image.shape}")

        height, width = map(int, image.shape[:2])

        metadata = ImageMetadata(
            original_shape=(height, width),
            dtype=str(image.dtype),
        )

        return TransformInput(image=image, polygons=polygon_models, metadata=metadata)


# Backwards compatibility for existing imports
DBTransforms = ValidatedDBTransforms
