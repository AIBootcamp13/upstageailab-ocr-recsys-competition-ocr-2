import json
import logging
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ocr.utils.orientation import (
    EXIF_ORIENTATION_TAG,
    FLIP_LEFT_RIGHT,
    normalize_pil_image,
    orientation_requires_rotation,
    polygons_in_canonical_frame,
    remap_polygons,
)

Image.MAX_IMAGE_PIXELS = 108000000
EXIF_ORIENTATION = EXIF_ORIENTATION_TAG  # Orientation Information: 274


class OCRDataset(Dataset):
    DEFAULT_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

    def __init__(self, image_path, annotation_path, transform, image_extensions=None):
        self.image_path = Path(image_path)
        self.transform = transform
        self.logger = logging.getLogger(__name__)
        self._canonical_frame_logged = set()

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
            pil_image = Image.open(image_path)
        except OSError as e:
            raise RuntimeError(f"Failed to load image {image_filename}: {e}")

        raw_width, raw_height = pil_image.size
        try:
            normalized_image, orientation = normalize_pil_image(pil_image)
        except Exception as exc:
            pil_image.close()
            raise RuntimeError(f"Failed to normalize image {image_filename}: {exc}") from exc

        if normalized_image.mode != "RGB":
            image = normalized_image.convert("RGB")
        else:
            image = normalized_image.copy()

        if normalized_image is not pil_image:
            normalized_image.close()
        pil_image.close()

        raw_polygons = self.anns[image_filename] or None
        polygons = None
        polygon_frame = "raw"
        if raw_polygons:
            polygons_list = [np.asarray(poly, dtype=np.float32) for poly in raw_polygons]
            if orientation_requires_rotation(orientation):
                if polygons_in_canonical_frame(polygons_list, raw_width, raw_height, orientation):
                    polygons = polygons_list
                    polygon_frame = "canonical"
                    if image_filename not in self._canonical_frame_logged:
                        self.logger.debug(
                            "Skipping EXIF remap for %s; polygons already align with canonical orientation (orientation=%d).",
                            image_filename,
                            orientation,
                        )
                        self._canonical_frame_logged.add(image_filename)
                else:
                    polygons = remap_polygons(polygons_list, raw_width, raw_height, orientation)
                    polygon_frame = "canonical"
            else:
                polygons = polygons_list

        org_shape = image.size

        item = OrderedDict(
            image=image,
            image_filename=image_filename,
            image_path=str(image_path),
            shape=org_shape,
            raw_size=(raw_width, raw_height),
            orientation=orientation,
            polygon_frame=polygon_frame,
        )

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
            image = image.transpose(FLIP_LEFT_RIGHT)
        elif orientation == 3:
            image = image.rotate(180, **rotate_kwargs)
        elif orientation == 4:
            image = image.rotate(180, **rotate_kwargs).transpose(FLIP_LEFT_RIGHT)
        elif orientation == 5:
            image = image.rotate(-90, expand=True, **rotate_kwargs).transpose(FLIP_LEFT_RIGHT)
        elif orientation == 6:
            image = image.rotate(-90, expand=True, **rotate_kwargs)
        elif orientation == 7:
            image = image.rotate(90, expand=True, **rotate_kwargs).transpose(FLIP_LEFT_RIGHT)
        elif orientation == 8:
            image = image.rotate(90, expand=True, **rotate_kwargs)
        # Orientation 1 (normal) and any other values: no rotation needed
        return image

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
