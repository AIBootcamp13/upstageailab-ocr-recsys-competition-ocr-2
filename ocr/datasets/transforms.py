from collections import OrderedDict

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


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


class DBTransforms:
    def __init__(self, transforms, keypoint_params):
        self.transform = A.Compose([*transforms, ToTensorV2()], keypoint_params=keypoint_params)

    def _validate_polygons(self, polygons: list[np.ndarray] | None) -> None:
        """
        Validate polygon shapes and types for transform pipeline.

        Args:
            polygons: List of polygon arrays to validate

        Raises:
            TypeError: If polygons is not a list or contains invalid types
            ValueError: If polygon shapes are invalid
        """
        if polygons is None:
            return

        if not isinstance(polygons, list):
            raise TypeError(f"polygons must be a list, got {type(polygons)}")

        for idx, polygon in enumerate(polygons):
            if not isinstance(polygon, np.ndarray):
                raise TypeError(f"polygon at index {idx} must be numpy array, got {type(polygon)}")

            if polygon.dtype not in (np.float32, np.float64, np.int32, np.int64):
                raise TypeError(f"polygon at index {idx} must be numeric array, got dtype {polygon.dtype}")

            if polygon.ndim not in (2, 3):
                raise ValueError(f"polygon at index {idx} must be 2D or 3D array, got {polygon.ndim}D with shape {polygon.shape}")

            if polygon.ndim == 2:
                # Shape should be (N, 2) where N is number of points
                if polygon.shape[1] != 2:
                    raise ValueError(f"polygon at index {idx} must have shape (N, 2), got {polygon.shape}")
                # Note: Minimum point validation happens in processing loop
            elif polygon.ndim == 3:
                # Shape should be (1, N, 2) for backward compatibility
                if polygon.shape[0] != 1 or polygon.shape[2] != 2:
                    raise ValueError(f"polygon at index {idx} must have shape (1, N, 2), got {polygon.shape}")
                # Note: Minimum point validation happens in processing loop

    def _validate_transform_contracts(self, image: np.ndarray, polygons: list[np.ndarray] | None) -> None:
        """
        Validate input contracts for the transform pipeline.

        Args:
            image: Input image array
            polygons: List of polygon arrays

        Raises:
            ValueError: If input contracts are violated
        """
        # Validate image shape and type
        if image.ndim not in (2, 3):
            raise ValueError(f"Image must be 2D or 3D array, got shape {image.shape}")

        if image.ndim == 3 and image.shape[2] not in (1, 3):
            raise ValueError(f"Image must have 1 or 3 channels, got {image.shape[2]}")

        # Validate polygons if provided
        if polygons is not None:
            self._validate_polygons(polygons)

    def _validate_output_contracts(self, result: OrderedDict) -> None:
        """
        Validate output contracts for the transform pipeline.

        Args:
            result: Output OrderedDict from transform

        Raises:
            ValueError: If output contracts are violated
        """
        required_keys = {"image", "polygons", "inverse_matrix"}
        if not all(key in result for key in required_keys):
            missing = required_keys - set(result.keys())
            raise ValueError(f"Output missing required keys: {missing}")

        # Validate image output
        image = result["image"]
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Output image must be torch.Tensor, got {type(image)}")

        if image.ndim != 3:
            raise ValueError(f"Output image must be 3D tensor (C, H, W), got shape {image.shape}")

        # Validate polygons output
        polygons = result["polygons"]
        if not isinstance(polygons, list):
            raise TypeError(f"Output polygons must be list, got {type(polygons)}")

        for idx, polygon in enumerate(polygons):
            if not isinstance(polygon, np.ndarray):
                raise TypeError(f"Output polygon at index {idx} must be numpy array, got {type(polygon)}")

            if polygon.shape != (1, polygon.shape[1], 2):
                raise ValueError(f"Output polygon at index {idx} must have shape (1, N, 2), got {polygon.shape}")

            if polygon.dtype != np.float32:
                raise ValueError(f"Output polygon at index {idx} must be float32, got {polygon.dtype}")

        # Validate inverse matrix
        inverse_matrix = result["inverse_matrix"]
        if not isinstance(inverse_matrix, np.ndarray):
            raise TypeError(f"Inverse matrix must be numpy array, got {type(inverse_matrix)}")

        if inverse_matrix.shape != (3, 3):
            raise ValueError(f"Inverse matrix must be 3x3, got shape {inverse_matrix.shape}")

        if inverse_matrix.dtype not in (np.float32, np.float64):
            raise ValueError(f"Inverse matrix must be float32 or float64, got {inverse_matrix.dtype}")

    def __call__(self, image: np.ndarray, polygons: list[np.ndarray] | None) -> OrderedDict:
        """
        Apply transforms to image and polygons.

        Args:
            image: RGB image as numpy array with shape (H, W, 3) or PIL Image
            polygons: List of polygon arrays with shape (N, 2) where:
                     - N is number of points (>= 3 for valid polygons)
                     - 2 represents (x, y) coordinates
                     Also accepts (1, N, 2) for backward compatibility

        Returns:
            OrderedDict with:
                - image: Transformed image tensor
                - polygons: List of transformed polygons with shape (1, N, 2)
                - inverse_matrix: Matrix for coordinate transformation
        """
        # BUG FIX (BUG-2025-002): Add defensive type check for PIL Images
        # Albumentations/DBTransforms expect numpy arrays, not PIL Images
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            image = np.array(image)

        # Type validation for image
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array or PIL Image, got {type(image)}")

        if image.ndim not in (2, 3):
            raise ValueError(f"Image must be 2D or 3D array, got shape {image.shape}")

        height, width = image.shape[:2]

        # Validate input contracts
        self._validate_transform_contracts(image, polygons)

        keypoints = []
        if polygons is not None:
            # BUG FIX (BUG-2025-004): Validate polygon shapes before processing
            self._validate_polygons(polygons)

            # Polygons 정보를 Keypoints 형태로 변환
            keypoints = [point for polygon in polygons for point in polygon.reshape(-1, 2)]
            # keypoints가 이미지의 크기를 벗어나지 않도록 제한
            keypoints = self.clamp_keypoints(keypoints, width, height)

        # Image transform / Geometric transform의 경우 keypoints를 변환
        transformed = self.transform(image=image, keypoints=keypoints)
        transformed_image = transformed["image"]
        transformed_keypoints = transformed["keypoints"]
        metadata = transformed.get("metadata")

        # Keypoints 재변환을 위한 Matrix 계산
        _, new_height, new_width = transformed_image.shape
        crop_box = self.calculate_cropbox((width, height), max(new_height, new_width))
        inverse_matrix = self.calculate_inverse_transform((width, height), (new_width, new_height), crop_box=crop_box)

        # Keypoints 정보를 Polygons 형태로 변환
        # BUG FIX (BUG-2025-004): Correct polygon point count extraction
        transformed_polygons = []
        index = 0
        if polygons is not None:
            for polygon_idx, polygon in enumerate(polygons):
                # Get number of points - handle both (N, 2) and (1, N, 2) shapes
                # BUG-2025-004: Must use correct dimension for point count
                if polygon.ndim == 2:
                    # Shape is (N, 2) where N = number of points
                    # shape[0] = N (correct), shape[1] = 2 (wrong - coordinate dimension)
                    num_points = polygon.shape[0] if polygon.shape[0] != 1 else polygon.shape[1]
                elif polygon.ndim == 3:
                    # Shape is (1, N, 2) where N = number of points
                    # shape[1] = N (correct)
                    num_points = polygon.shape[1]
                else:
                    # Invalid polygon dimension - log and skip
                    import logging

                    logging.warning(
                        f"Invalid polygon dimension at index {polygon_idx}: "
                        f"expected 2D or 3D array, got {polygon.ndim}D with shape {polygon.shape}"
                    )
                    continue

                # Extract transformed keypoints for this polygon
                keypoint_slice = transformed_keypoints[index : index + num_points]
                index += len(keypoint_slice)

                # Skip degenerate polygons (< 3 points)
                if len(keypoint_slice) < 3:
                    import logging

                    logging.debug(f"Skipping degenerate polygon at index {polygon_idx}: only {len(keypoint_slice)} points after transform")
                    continue

                # Reshape to standard output format (1, N, 2)
                polygon_array = np.array(keypoint_slice, dtype=np.float32).reshape(1, -1, 2)
                transformed_polygons.append(polygon_array)

        output = OrderedDict(
            image=transformed_image,
            polygons=transformed_polygons,
            inverse_matrix=inverse_matrix,
        )

        if metadata is not None:
            output["metadata"] = metadata

        # Validate output contracts
        self._validate_output_contracts(output)

        return output

    def clamp_keypoints(self, keypoints: list, img_width: int, img_height: int) -> list:
        clamped_keypoints = []
        for kp in keypoints:
            x, y = kp[:2]
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            clamped_keypoints.append((x, y) + tuple(kp[2:]))
        return clamped_keypoints

    @staticmethod
    def calculate_inverse_transform(
        original_size: tuple[int, int], transformed_size: tuple[int, int], crop_box: tuple | None = None
    ) -> np.ndarray:
        ox, oy = original_size
        tx, ty = transformed_size
        cx, cy = 0, 0
        if crop_box:
            cx, cy, tx, ty = crop_box

        # Scale back to the original size
        scale_x = ox / tx
        scale_y = oy / ty
        scale_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])

        # Padding back to the original size
        translation_matrix = np.eye(3)
        translation_matrix[0, 2] = -cx
        translation_matrix[1, 2] = -cy

        inverse_matrix = np.dot(scale_matrix, translation_matrix)
        return inverse_matrix

    @staticmethod
    def calculate_cropbox(original_size: tuple[int, int], target_size: int = 640) -> tuple[int, int, int, int]:
        ox, oy = original_size
        scale = target_size / max(ox, oy)
        new_width, new_height = int(ox * scale), int(oy * scale)
        delta_w = target_size - new_width
        delta_h = target_size - new_height
        x, y = delta_w // 2, delta_h // 2
        w, h = new_width, new_height
        return x, y, w, h
