import json
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image

from ocr.datasets.base import EXIF_ORIENTATION
from ocr.datasets.base import ValidatedOCRDataset as Dataset
from ocr.datasets.schemas import DatasetConfig
from ocr.datasets.transforms import DBTransforms
from ocr.utils.orientation import remap_polygons

RAW_POLYGON_POINTS = [
    [10, 20],
    [30, 20],
    [30, 40],
    [10, 40],
]


class IdentityTransform:
    def __call__(self, transform_input):
        image = transform_input.image
        polygons = transform_input.polygons
        # Extract polygon points if they exist
        polygon_arrays = []
        if polygons is not None:
            for poly in polygons:
                polygon_arrays.append(poly.points)

        return {
            "image": image,
            "polygons": polygon_arrays,
            "inverse_matrix": np.eye(3, dtype=np.float32),
        }


def _write_image_with_orientation(path: Path, size: tuple[int, int], orientation: int) -> None:
    image = Image.new("RGB", size, color=(255, 255, 255))
    exif = image.getexif()
    exif[EXIF_ORIENTATION] = orientation
    image.save(path, exif=exif)


def _write_annotations(path: Path, filename: str) -> None:
    annotations = {"images": {filename: {"words": {"word_0": {"points": RAW_POLYGON_POINTS}}}}}
    path.write_text(json.dumps(annotations), encoding="utf-8")


def test_dataset_remaps_polygons_with_orientation(tmp_path: Path):
    image_dir = tmp_path / "images"
    json_dir = tmp_path / "jsons"
    image_dir.mkdir()
    json_dir.mkdir()

    filename = "sample.jpg"
    image_path = image_dir / filename
    json_path = json_dir / "train.json"

    _write_image_with_orientation(image_path, (100, 200), orientation=6)
    _write_annotations(json_path, filename)

    config = DatasetConfig(image_path=image_dir, annotation_path=json_path)
    dataset = Dataset(config=config, transform=IdentityTransform())

    sample = dataset[0]

    canonical = remap_polygons([np.array([RAW_POLYGON_POINTS], dtype=np.float32)], 100, 200, orientation=6)[0]
    # Handle shape differences - canonical might be (1, 4, 2), but we need (4, 2) to match actual
    if canonical.ndim == 3 and canonical.shape[0] == 1:
        expected = canonical[0].astype(np.float32)
    else:
        expected = canonical.astype(np.float32)
    assert isinstance(sample["polygons"], list)
    assert sample["polygons"], "Polygons should not be empty"
    actual = np.asarray(sample["polygons"][0], dtype=np.float32)
    # Handle potential shape differences: (N, 2) vs (1, N, 2)
    if actual.ndim == 3 and actual.shape[0] == 1:
        actual = actual[0]  # Remove batch dimension if present
    np.testing.assert_allclose(actual, expected)


def test_dataset_rotates_image_to_canonical_orientation(tmp_path: Path):
    image_dir = tmp_path / "images"
    json_dir = tmp_path / "jsons"
    image_dir.mkdir()
    json_dir.mkdir()

    filename = "sample.jpg"
    image_path = image_dir / filename
    json_path = json_dir / "train.json"

    _write_image_with_orientation(image_path, (100, 200), orientation=6)
    _write_annotations(json_path, filename)

    config = DatasetConfig(image_path=image_dir, annotation_path=json_path)
    dataset = Dataset(config=config, transform=IdentityTransform())
    sample = dataset[0]

    image_array = np.asarray(sample["image"])
    # Orientation 6 rotates -90 degrees, swapping width/height
    assert image_array.shape[0] == 100
    assert image_array.shape[1] == 200


def test_dataset_albumentations_preserves_polygon_alignment(tmp_path: Path):
    image_dir = tmp_path / "images"
    json_dir = tmp_path / "jsons"
    image_dir.mkdir()
    json_dir.mkdir()

    filename = "sample.jpg"
    image_path = image_dir / filename
    json_path = json_dir / "train.json"

    _write_image_with_orientation(image_path, (100, 200), orientation=6)
    _write_annotations(json_path, filename)

    transform = DBTransforms(
        transforms=[A.HorizontalFlip(p=1.0)],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    config = DatasetConfig(image_path=image_dir, annotation_path=json_path)
    dataset = Dataset(config=config, transform=transform)
    sample = dataset[0]

    assert isinstance(sample["polygons"], list)
    assert sample["polygons"], "Polygons should not be empty after transform"

    image_array = np.asarray(sample["image"])
    width = int(image_array.shape[-1])

    canonical = remap_polygons([np.array([RAW_POLYGON_POINTS], dtype=np.float32)], 100, 200, orientation=6)[0]
    # Handle shape differences - canonical might be (1, 4, 2), but we need (4, 2) to match actual
    if canonical.ndim == 3 and canonical.shape[0] == 1:
        canonical_2d = canonical[0]
    else:
        canonical_2d = canonical
    canonical_points = canonical_2d.reshape(-1, 2)
    flipped_points = canonical_points.copy()
    flipped_points[:, 0] = (width - 1) - flipped_points[:, 0]
    expected = flipped_points.reshape(canonical_2d.shape)

    actual = np.asarray(sample["polygons"][0], dtype=np.float32)
    # Handle potential shape differences: (N, 2) vs (1, N, 2)
    if actual.ndim == 3 and actual.shape[0] == 1:
        actual = actual[0]  # Remove batch dimension if present
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
