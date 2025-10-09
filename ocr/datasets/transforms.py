from collections import OrderedDict

import albumentations as A
import numpy as np
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

    def __call__(self, image, polygons):
        height, width = image.shape[:2]

        keypoints = []
        if polygons is not None:
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
        transformed_polygons = []
        index = 0
        if polygons is not None:
            for polygon in polygons:
                num_points = polygon.shape[1]
                keypoint_slice = transformed_keypoints[index : index + num_points]
                index += len(keypoint_slice)

                if len(keypoint_slice) < 3:
                    continue

                polygon_array = np.array(keypoint_slice, dtype=np.float32).reshape(1, -1, 2)
                transformed_polygons.append(polygon_array)

        output = OrderedDict(
            image=transformed_image,
            polygons=transformed_polygons,
            inverse_matrix=inverse_matrix,
        )

        if metadata is not None:
            output["metadata"] = metadata

        return output

    def clamp_keypoints(self, keypoints, img_width, img_height):
        clamped_keypoints = []
        for kp in keypoints:
            x, y = kp[:2]
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            clamped_keypoints.append((x, y) + tuple(kp[2:]))
        return clamped_keypoints

    @staticmethod
    def calculate_inverse_transform(original_size, transformed_size, crop_box=None):
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
    def calculate_cropbox(original_size, target_size=640):
        ox, oy = original_size
        scale = target_size / max(ox, oy)
        new_width, new_height = int(ox * scale), int(oy * scale)
        delta_w = target_size - new_width
        delta_h = target_size - new_height
        x, y = delta_w // 2, delta_h // 2
        w, h = new_width, new_height
        return x, y, w, h
