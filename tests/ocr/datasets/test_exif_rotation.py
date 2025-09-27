import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ocr.datasets.base import EXIF_ORIENTATION, OCRDataset


class IdentityTransform:
    def __call__(self, **kwargs):
        image = kwargs["image"]
        polygons = kwargs["polygons"]
        return {
            "image": image,
            "polygons": polygons,
            "inverse_matrix": np.eye(3, dtype=np.float32),
        }


def _write_image_with_orientation(path: Path, size: tuple[int, int], orientation: int) -> None:
    image = Image.new("RGB", size, color=(255, 255, 255))
    exif = image.getexif()
    exif[EXIF_ORIENTATION] = orientation
    image.save(path, exif=exif)


def _write_annotations(path: Path, filename: str) -> None:
    annotations = {
        "images": {
            filename: {
                "words": {
                    "word_0": {
                        "points": [
                            [10, 20],
                            [30, 20],
                            [30, 40],
                            [10, 40],
                        ]
                    }
                }
            }
        }
    }
    path.write_text(json.dumps(annotations), encoding="utf-8")


@pytest.mark.parametrize(
    "orientation,expected",
    [
        (
            6,
            np.array([[[179, 10], [179, 30], [159, 30], [159, 10]]], dtype=np.int32),
        ),
        (
            5,
            np.array([[[20, 10], [20, 30], [40, 30], [40, 10]]], dtype=np.int32),
        ),
        (
            8,
            np.array([[[20, 89], [20, 69], [40, 69], [40, 89]]], dtype=np.int32),
        ),
    ],
)
def test_transform_polygons_for_exif_expected_coordinates(orientation, expected):
    polygons = [
        np.array([[[10, 20], [30, 20], [30, 40], [10, 40]]], dtype=np.int32),
    ]
    transformed = OCRDataset.transform_polygons_for_exif(polygons, orientation, image_size=(100, 200))
    np.testing.assert_array_equal(transformed[0], expected)


def test_transform_polygons_handles_list_input():
    polygons = [
        [
            [10, 20],
            [30, 20],
            [30, 40],
            [10, 40],
        ]
    ]

    transformed = OCRDataset.transform_polygons_for_exif(polygons, orientation=6, image_size=(100, 200))

    expected = np.array([[[179, 10], [179, 30], [159, 30], [159, 10]]], dtype=np.float32)
    assert isinstance(transformed[0], np.ndarray)
    np.testing.assert_array_equal(transformed[0], expected)


def test_dataset_applies_exif_transform_to_polygons(tmp_path: Path):
    image_dir = tmp_path / "images"
    json_dir = tmp_path / "jsons"
    image_dir.mkdir()
    json_dir.mkdir()

    filename = "sample.jpg"
    image_path = image_dir / filename
    json_path = json_dir / "train.json"

    _write_image_with_orientation(image_path, (100, 200), orientation=6)
    _write_annotations(json_path, filename)

    dataset = OCRDataset(image_path=image_dir, annotation_path=json_path, transform=IdentityTransform())

    sample = dataset[0]

    expected = np.array([[[179, 10], [179, 30], [159, 30], [159, 10]]], dtype=np.int32)
    assert isinstance(sample["polygons"], list)
    assert sample["polygons"], "Polygons should not be empty"
    np.testing.assert_array_equal(sample["polygons"][0], expected)

    image_array = np.asarray(sample["image"])
    assert image_array.shape[0] == 100
    assert image_array.shape[1] == 200


def test_transform_polygons_filters_degenerate_after_clipping():
    polygons = [
        np.array(
            [
                [165.43, 970.5],
                [192.15, 970.5],
                [192.15, 999.77],
                [165.43, 999.77],
            ],
            dtype=np.float32,
        ).reshape(1, -1, 2)
    ]

    transformed = OCRDataset.transform_polygons_for_exif(polygons, orientation=6, image_size=(1280, 960))

    assert transformed == []
