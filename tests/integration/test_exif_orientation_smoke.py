from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ocr.datasets import ValidatedOCRDataset
from ocr.datasets.base import EXIF_ORIENTATION
from ocr.datasets.schemas import DatasetConfig
from ocr.utils.orientation import normalize_pil_image, remap_polygons
from ui.utils.inference.engine import InferenceEngine
from ui.visualization.helpers import parse_polygon_string

RAW_POLYGON_POINTS = [
    [10, 20],
    [30, 20],
    [30, 40],
    [10, 40],
]


class IdentityTransform:
    def __call__(self, data):
        # Handle both dict and TransformInput
        if hasattr(data, "image"):
            # TransformInput object
            image = data.image
            polygons = data.polygons or []
            # Convert PolygonData objects back to numpy arrays
            if polygons:
                polygons = [poly.points for poly in polygons]
        else:
            # Dict
            image = data["image"]
            polygons = data["polygons"]

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
    annotations = {"images": {filename: {"words": {"word_0": {"points": RAW_POLYGON_POINTS}}}}}
    path.write_text(json.dumps(annotations), encoding="utf-8")


@pytest.mark.parametrize("orientation", [1, 6])
def test_dataset_to_inference_to_viewer_alignment(tmp_path: Path, orientation: int) -> None:
    image_dir = tmp_path / "images"
    json_dir = tmp_path / "jsons"
    image_dir.mkdir()
    json_dir.mkdir()

    filename = "sample.jpg"
    image_path = image_dir / filename
    json_path = json_dir / "train.json"

    _write_image_with_orientation(image_path, (100, 200), orientation)
    _write_annotations(json_path, filename)

    config = DatasetConfig(
        image_path=image_dir,
        annotation_path=json_path,
        preload_maps=False,
        load_maps=False,
        preload_images=False,
        prenormalize_images=False,
    )

    dataset = ValidatedOCRDataset(config=config, transform=IdentityTransform())
    sample = dataset[0]

    canonical_polygon = np.asarray(sample["polygons"][0], dtype=np.float32).reshape(-1, 2)
    image_array = np.asarray(sample["image"], dtype=np.uint8)
    canonical_height, canonical_width = image_array.shape[:2]

    canonical_str = ",".join(str(int(round(coord))) for coord in canonical_polygon.reshape(-1))

    engine = InferenceEngine()
    remapped_result = engine._remap_predictions_if_needed(  # noqa: SLF001
        {"polygons": canonical_str},
        orientation=orientation,
        canonical_width=int(canonical_width),
        canonical_height=int(canonical_height),
    )

    raw_polygons_str = remapped_result.get("polygons", "")
    if orientation != 1:
        assert raw_polygons_str != canonical_str
    else:
        assert raw_polygons_str == canonical_str

    with Image.open(image_path) as raw_image:
        raw_width, raw_height = raw_image.size
        normalized_image, applied_orientation = normalize_pil_image(raw_image)
        if normalized_image is not raw_image:
            normalized_image.close()

    parsed_polygons = parse_polygon_string(raw_polygons_str)
    assert parsed_polygons, "Viewer parser should return polygons"

    np_polygons = [np.asarray(poly, dtype=np.float32).reshape(1, -1, 2) for poly in parsed_polygons]

    if applied_orientation != 1:
        remapped_viewer = remap_polygons(np_polygons, raw_width, raw_height, applied_orientation)
    else:
        remapped_viewer = np_polygons

    canonical_from_viewer = remapped_viewer[0].reshape(-1, 2)
    np.testing.assert_allclose(canonical_from_viewer, canonical_polygon, atol=1e-5, rtol=1e-5)
    assert image_array.shape[1] == canonical_width
    assert image_array.shape[0] == canonical_height
