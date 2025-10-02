import json
from pathlib import Path

import numpy as np
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


def test_dataset_preserves_polygons_when_rotated(tmp_path: Path):
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

    expected = np.array([[[10, 20], [30, 20], [30, 40], [10, 40]]], dtype=np.int32)
    assert isinstance(sample["polygons"], list)
    assert sample["polygons"], "Polygons should not be empty"
    np.testing.assert_array_equal(sample["polygons"][0], expected)


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

    dataset = OCRDataset(image_path=image_dir, annotation_path=json_path, transform=IdentityTransform())
    sample = dataset[0]

    image_array = np.asarray(sample["image"])
    # Orientation 6 rotates -90 degrees, swapping width/height
    assert image_array.shape[0] == 100
    assert image_array.shape[1] == 200
