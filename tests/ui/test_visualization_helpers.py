from __future__ import annotations

import numpy as np

from ui._visualization import helpers


def test_parse_polygon_string_accepts_comma_delimiters() -> None:
    polygon_str = "10,20,30,20,30,40,10,40|50,60,70,60,70,80,50,80"
    polygons = helpers.parse_polygon_string(polygon_str)

    assert len(polygons) == 2
    first = np.array(polygons[0], dtype=np.float32)
    np.testing.assert_allclose(first, np.array([10, 20, 30, 20, 30, 40, 10, 40], dtype=np.float32))


def test_parse_polygon_string_accepts_space_delimiters() -> None:
    polygon_str = "10 20 30 20 30 40 10 40"
    polygons = helpers.parse_polygon_string(polygon_str)

    assert len(polygons) == 1
    np.testing.assert_allclose(
        np.array(polygons[0], dtype=np.float32),
        np.array([10, 20, 30, 20, 30, 40, 10, 40], dtype=np.float32),
    )


def test_validate_polygons_counts_comma_delimited_entries() -> None:
    polygon_str = "10,20,30,20,30,40,10,40|invalid|"
    polygons, stats = helpers.validate_polygons(polygon_str)

    assert len(polygons) == 1
    assert stats["total"] == 3
    assert stats["valid"] == 1
    assert stats["parse_errors"] == 2
