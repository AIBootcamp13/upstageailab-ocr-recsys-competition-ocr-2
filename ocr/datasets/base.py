import json
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = 108000000
EXIF_ORIENTATION = 274  # Orientation Information: 274


class OCRDataset(Dataset):
    DEFAULT_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"]

    def __init__(self, image_path, annotation_path, transform, image_extensions=None):
        self.image_path = Path(image_path)
        self.transform = transform

        # Allow configurable image extensions, fallback to default if not provided
        if image_extensions is None:
            self.image_extensions = self.DEFAULT_IMAGE_EXTENSIONS
        else:
            self.image_extensions = [ext.lower() for ext in image_extensions]

        self.anns = OrderedDict()

        # annotation_path가 없다면, image_path에서 이미지만 불러오기
        if annotation_path is None:
            for file in self.image_path.glob("*"):
                # Ensure file is a file (not a directory) and has a valid image extension
                if file.is_file() and file.suffix.lower() in self.image_extensions:
                    self.anns[file.name] = None
            return

        import logging

        class AnnotationFileError(Exception):
            pass

        try:
            with open(annotation_path) as f:
                annotations = json.load(f)
        except FileNotFoundError:
            logging.error(f"Annotation file not found: {annotation_path}")
            raise AnnotationFileError(f"Annotation file not found: {annotation_path}")
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in annotation file: {annotation_path}")
            raise AnnotationFileError(f"Invalid JSON in annotation file: {annotation_path}")
        except Exception as e:
            logging.error(f"Error loading annotation file {annotation_path}: {e}")
            raise AnnotationFileError(f"Error loading annotation file {annotation_path}: {e}")

        for filename in annotations.get("images", {}).keys():
            # Image file이 경로에 존재하는지 확인
            if (self.image_path / filename).exists():
                # words 정보를 가지고 있는지 확인
                if "words" in annotations.get("images", {}).get(filename, {}):
                    # Words의 Points 변환
                    gt_words = annotations.get("images", {}).get(filename, {}).get("words", {})
                    polygons = [
                        np.array([np.round(word_data["points"])], dtype=np.int32)
                        for word_data in gt_words.values()
                        if isinstance(word_data.get("points"), list) and len(word_data["points"]) > 0
                    ]
                    self.anns[filename] = polygons or None
                else:
                    self.anns[filename] = None

    def __len__(self):
        return len(self.anns.keys())

    def __getitem__(self, idx):
        image_filename = list(self.anns.keys())[idx]
        image_path = self.image_path / image_filename
        try:
            image = Image.open(image_path).convert("RGB")
        except OSError as e:
            raise RuntimeError(f"Failed to load image {image_filename}: {e}")

        # EXIF정보를 확인하여 이미지 회전
        exif = image.getexif()
        polygons = self.anns[image_filename] or None
        if exif and EXIF_ORIENTATION in exif:
            orientation = exif[EXIF_ORIENTATION]
            if polygons is not None:
                polygons = self.transform_polygons_for_exif(polygons, orientation, image.size)
            image = OCRDataset.rotate_image(image, orientation)
        org_shape = image.size

        item = OrderedDict(image=image, image_filename=image_filename, shape=org_shape)

        if self.transform is None:
            raise ValueError("Transform function is a required value.")

        # Image transform
        transformed = self.transform(image=np.array(image), polygons=polygons)
        item.update(
            image=transformed["image"],
            polygons=transformed["polygons"],
            inverse_matrix=transformed["inverse_matrix"],
        )

        return item

    @staticmethod
    def rotate_image(image, orientation):
        """
        Rotate image based on EXIF orientation.
        Handles orientations 1-8 according to EXIF standard.
        Orientation 1 (normal) requires no rotation.
        """

        try:
            bicubic = Image.Resampling.BICUBIC  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - Pillow<9 fallback
            bicubic = Image.BICUBIC  # type: ignore[attr-defined]

        fillcolor: tuple[int, ...]
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            fillcolor = (0, 0, 0, 0)
        else:
            fillcolor = (255, 255, 255)

        rotate_kwargs: dict[str, Any] = {
            "resample": bicubic,
            "fillcolor": fillcolor,
        }

        if orientation == 2:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            image = image.rotate(180, **rotate_kwargs)
        elif orientation == 4:
            image = image.rotate(180, **rotate_kwargs).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            image = image.rotate(-90, expand=True, **rotate_kwargs).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            image = image.rotate(-90, expand=True, **rotate_kwargs)
        elif orientation == 7:
            image = image.rotate(90, expand=True, **rotate_kwargs).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            image = image.rotate(90, expand=True, **rotate_kwargs)
        # Orientation 1 (normal) and any other values: no rotation needed
        return image

    @staticmethod
    def transform_polygons_for_exif(
        polygons: Sequence[np.ndarray | Sequence | None],
        orientation: int,
        image_size: tuple[int, int],
    ) -> list[np.ndarray]:
        """Apply EXIF-aware rotation to polygons and drop degenerate results.

        Polygons that already fit within the target oriented frame are not
        transformed again to prevent double-rotation, but they are still
        clipped and filtered so downstream consumers receive clean shapes.
        """
        if not polygons:
            return []

        normalised_polygons: list[np.ndarray] = [
            polygon_array
            for polygon in polygons
            for polygon_array in [OCRDataset._ensure_polygon_array(polygon)]
            if polygon_array is not None
        ]

        width, height = image_size
        if orientation in {5, 6, 7, 8}:
            new_width, new_height = height, width
        else:
            new_width, new_height = width, height

        fits_original = OCRDataset._polygons_fit_within(normalised_polygons, width, height)
        fits_rotated = OCRDataset._polygons_fit_within(normalised_polygons, new_width, new_height)

        force_identity = orientation == 1
        skip_transform = force_identity or (not force_identity and not fits_original and fits_rotated)

        transformed: list[np.ndarray] = []

        if skip_transform:
            transformed = [np.array(polygon, copy=True) for polygon in normalised_polygons]
        else:
            for polygon_array in normalised_polygons:
                normalised_shape = polygon_array.shape
                reshaped = np.asarray(polygon_array, dtype=np.float32).reshape(-1, 2)
                rotated_points = OCRDataset._apply_orientation_transform(
                    reshaped,
                    orientation=orientation,
                    width=width,
                    height=height,
                )
                transformed.append(rotated_points.astype(polygon_array.dtype, copy=False).reshape(normalised_shape))

        OCRDataset._clip_polygons_in_place(transformed, new_width, new_height)
        return OCRDataset._filter_degenerate_polygons(transformed)

    @staticmethod
    def _apply_orientation_transform(points: np.ndarray, orientation: int, width: int, height: int) -> np.ndarray:
        """Vectorised orientation transform for polygon points."""
        if orientation == 2:
            return np.stack((width - 1 - points[:, 0], points[:, 1]), axis=-1)
        if orientation == 3:
            return np.stack((width - 1 - points[:, 0], height - 1 - points[:, 1]), axis=-1)
        if orientation == 4:
            return np.stack((points[:, 0], height - 1 - points[:, 1]), axis=-1)
        if orientation == 5:
            return np.stack((points[:, 1], points[:, 0]), axis=-1)
        if orientation == 6:
            return np.stack((height - 1 - points[:, 1], points[:, 0]), axis=-1)
        if orientation == 7:
            return np.stack((height - 1 - points[:, 1], width - 1 - points[:, 0]), axis=-1)
        if orientation == 8:
            return np.stack((points[:, 1], width - 1 - points[:, 0]), axis=-1)
        return np.array(points, copy=True)

    @staticmethod
    def _ensure_polygon_array(polygon: np.ndarray | Sequence | None) -> np.ndarray | None:
        if polygon is None:
            return None
        polygon_array = polygon if isinstance(polygon, np.ndarray) else np.asarray(polygon, dtype=np.float32)

        if polygon_array.size == 0:
            return polygon_array

        if polygon_array.ndim == 1:
            if polygon_array.size % 2 != 0:
                return polygon_array.reshape(1, -1)
            return polygon_array.reshape(1, -1, 2)
        if polygon_array.ndim == 2:
            if polygon_array.shape[0] == 1:
                return polygon_array
            return polygon_array[np.newaxis, ...]
        if polygon_array.ndim == 3 and polygon_array.shape[0] == 1:
            return polygon_array
        return polygon_array.reshape(1, -1, 2)

    @staticmethod
    def _clip_polygons_in_place(polygons: Iterable[np.ndarray | None], width: int, height: int) -> None:
        for polygon in polygons:
            if polygon is None:
                continue
            polygon[..., 0] = np.clip(polygon[..., 0], 0, max(width - 1, 0))
            polygon[..., 1] = np.clip(polygon[..., 1], 0, max(height - 1, 0))

    @staticmethod
    def _filter_degenerate_polygons(
        polygons: Iterable[np.ndarray | None],
        min_side: float = 1.0,
    ) -> list[np.ndarray]:
        filtered: list[np.ndarray] = []
        for polygon in polygons:
            if polygon is None:
                continue
            if polygon.size == 0:
                continue

            reshaped = polygon.reshape(-1, 2)
            if reshaped.shape[0] < 3:
                continue

            width_span = float(reshaped[:, 0].max() - reshaped[:, 0].min())
            height_span = float(reshaped[:, 1].max() - reshaped[:, 1].min())

            if width_span < min_side or height_span < min_side:
                continue

            filtered.append(polygon)

        return filtered

    @staticmethod
    def _polygons_fit_within(polygons: Sequence[np.ndarray], width: int, height: int) -> bool:
        if width <= 0 or height <= 0:
            return False

        for polygon in polygons:
            if polygon.size == 0:
                continue
            points = polygon.reshape(-1, 2)
            if not points.size:
                continue
            xmin = float(points[:, 0].min())
            xmax = float(points[:, 0].max())
            ymin = float(points[:, 1].min())
            ymax = float(points[:, 1].max())
            if xmin < 0 or ymin < 0 or xmax >= width or ymax >= height:
                return False
        return True
