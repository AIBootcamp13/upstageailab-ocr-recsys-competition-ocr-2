from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ocr.utils.orientation import EXIF_ORIENTATION_TAG
from ui.visualize_annotations import _extract_polygons, _normalize_image_and_polygons


def _save_image_with_orientation(tmp_path: Path, size: tuple[int, int], orientation: int) -> Path:
    image = Image.new("RGB", size, color=(0, 0, 0))
    if orientation != 1:
        exif = image.getexif()
        exif[EXIF_ORIENTATION_TAG] = orientation
        image.save(tmp_path / "test.jpg", format="JPEG", exif=exif.tobytes())
    else:
        image.save(tmp_path / "test.jpg", format="JPEG")
    return tmp_path / "test.jpg"


def test_extract_polygons_filters_invalid_entries() -> None:
    payload = {
        "annotations": [
            {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "text": "valid"},
            {"polygon": [[0, 0], [1, 1]], "text": "degenerate"},
            {"polygon": [], "text": "empty"},
            {"text": "missing"},
        ]
    }

    entries = _extract_polygons(payload)
    assert len(entries) == 1
    assert entries[0]["annotation"]["text"] == "valid"
    np.testing.assert_array_equal(entries[0]["coords"], np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32))


@pytest.mark.parametrize("orientation", [1, 6])
def test_normalize_image_and_polygons_respects_orientation(tmp_path: Path, orientation: int) -> None:
    image_path = _save_image_with_orientation(tmp_path, size=(40, 20), orientation=orientation)

    payload = {
        "annotations": [
            {
                "polygon": [[0, 0], [30, 0], [30, 10], [0, 10]],
                "text": "sample",
            }
        ]
    }

    image, polygon_entries = _normalize_image_and_polygons(str(image_path), payload)

    assert len(polygon_entries) == 1

    coords = polygon_entries[0]["coords"]
    if orientation == 1:
        expected = np.array([[0, 0], [30, 0], [30, 10], [0, 10]], dtype=np.float32)
        np.testing.assert_allclose(coords, expected)
        assert image.size == (40, 20)
    else:
        expected = np.array([[19, 0], [19, 30], [9, 30], [9, 0]], dtype=np.float32)
        np.testing.assert_allclose(coords, expected)
        assert image.size == (20, 40)
