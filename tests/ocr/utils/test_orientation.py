from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ocr.utils.orientation import (
    EXIF_ORIENTATION_TAG,
    apply_affine_transform_to_polygons,
    get_exif_orientation,
    normalize_ndarray,
    normalize_pil_image,
    orientation_requires_rotation,
    remap_polygons,
)
from ui.utils.inference.engine import InferenceEngine


def _image_with_orientation(size: tuple[int, int], orientation: int | None) -> Image.Image:
    image = Image.new("RGB", size, color=(0, 0, 0))
    if orientation is not None:
        exif = image.getexif()
        exif[EXIF_ORIENTATION_TAG] = orientation
        image.info["exif"] = exif.tobytes()
    return image


def test_get_exif_orientation_defaults_to_one() -> None:
    image = _image_with_orientation((4, 4), orientation=None)
    assert get_exif_orientation(image) == 1


@pytest.mark.parametrize("orientation", [2, 3, 6, 8])
def test_get_exif_orientation_reads_tag(orientation: int) -> None:
    image = _image_with_orientation((4, 4), orientation=orientation)
    assert get_exif_orientation(image) == orientation


@pytest.mark.parametrize(
    "orientation,expected",
    [(1, False), (2, True), (3, True), (6, True), (9, False)],
)
def test_orientation_requires_rotation(orientation: int, expected: bool) -> None:
    assert orientation_requires_rotation(orientation) is expected


def test_normalize_pil_image_noop_orientation_one() -> None:
    image = _image_with_orientation((4, 4), orientation=1)
    normalized, applied = normalize_pil_image(image)
    assert applied == 1
    assert normalized is image


def test_normalize_pil_image_rotates_orientation_six() -> None:
    image = _image_with_orientation((4, 2), orientation=6)
    pixels = image.load()
    pixels[0, 0] = (255, 0, 0)

    normalized, applied = normalize_pil_image(image)
    assert applied == 6
    assert normalized.size == (2, 4)

    marker_coords = [
        (x, y) for y in range(normalized.height) for x in range(normalized.width) if normalized.getpixel((x, y)) == (255, 0, 0)
    ]
    assert marker_coords == [(1, 0)]


def test_normalize_ndarray_rotates_orientation_six() -> None:
    array = np.zeros((3, 4, 1), dtype=np.uint8)
    array[0, 0, 0] = 255

    rotated = normalize_ndarray(array, orientation=6)
    assert rotated.shape == (4, 3, 1)
    assert rotated[0, 2, 0] == 255
    assert rotated.sum() == 255


def test_remap_polygons_orientation_six_matches_image_rotation() -> None:
    width, height = 1280, 960
    polygon = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)

    remapped = remap_polygons([polygon], width, height, orientation=6)
    mapped = remapped[0]

    expected = np.array(
        [
            [height - 1, 0],
            [height - 1, 100],
            [height - 1 - 50, 100],
            [height - 1 - 50, 0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(mapped, expected, atol=1e-5)


def test_remap_polygons_noop_orientation_one() -> None:
    polygon = np.array([[10, 20], [30, 40]], dtype=np.float32)
    remapped = remap_polygons([polygon], width=100, height=200, orientation=1)
    np.testing.assert_array_equal(remapped[0], polygon)


def test_remap_polygons_flattens_to_viewer_format() -> None:
    width, height = 100, 200
    polygon = np.array([10, 20, 30, 20, 30, 40, 10, 40], dtype=np.float32).reshape(1, -1, 2)

    remapped = remap_polygons([polygon], width, height, orientation=6)
    flattened = remapped[0].reshape(-1)

    expected = np.array([179, 10, 179, 30, 159, 30, 159, 10], dtype=np.float32)
    np.testing.assert_allclose(flattened, expected, atol=1e-5)


def test_apply_affine_transform_handles_list_polygons() -> None:
    polygons = [[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]]
    matrix = np.array(
        [
            [1.0, 0.0, 5.0],
            [0.0, 1.0, -3.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    transformed = apply_affine_transform_to_polygons(polygons, matrix)

    expected = np.array([[5.0, -3.0], [15.0, -3.0], [15.0, 2.0], [5.0, 2.0]], dtype=np.float32)
    assert len(transformed) == 1
    np.testing.assert_allclose(transformed[0], expected)


@pytest.mark.parametrize("orientation", [5, 6, 7])
def test_inference_engine_remaps_polygons_back_to_raw_orientation(orientation: int) -> None:
    engine = InferenceEngine()

    raw_width, raw_height = 200, 100
    raw_polygon = np.array([[10, 20], [30, 20], [30, 40], [10, 40]], dtype=np.float32)

    canonical_polygons = remap_polygons([raw_polygon], raw_width, raw_height, orientation)
    canonical_polygon = canonical_polygons[0]

    if orientation in {5, 6, 7, 8}:
        canonical_width, canonical_height = raw_height, raw_width
    else:
        canonical_width, canonical_height = raw_width, raw_height

    canonical_str = ",".join(str(int(round(value))) for value in canonical_polygon.reshape(-1))
    result = {
        "polygons": canonical_str,
        "texts": ["dummy"],
        "confidences": [0.9],
    }

    remapped = engine._remap_predictions_if_needed(
        result,
        orientation=orientation,
        canonical_width=canonical_width,
        canonical_height=canonical_height,
    )

    expected_str = ",".join(str(int(round(value))) for value in raw_polygon.reshape(-1))
    assert remapped["polygons"] == expected_str
