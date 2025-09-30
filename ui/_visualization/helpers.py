from __future__ import annotations

"""Shared helpers for visualization components."""

from collections.abc import Iterable

from PIL import Image, ImageDraw

Polygon = list[float]


def parse_polygon_string(polygons_str: str) -> list[Polygon]:
    """Parse a serialized polygon string into numeric coordinates."""
    polygons: list[Polygon] = []
    if not polygons_str.strip():
        return polygons

    for raw_polygon in polygons_str.split("|"):
        coords: Polygon = []
        for token in raw_polygon.split():
            try:
                coords.append(float(token))
            except ValueError:
                coords = []
                break
        if len(coords) >= 8 and len(coords) % 2 == 0:
            polygons.append(coords)
    return polygons


def polygon_points(coords: Iterable[float]) -> list[tuple[float, float]]:
    """Convert a flat coordinate list into point tuples."""
    coords_list = list(coords)
    return [(coords_list[i], coords_list[i + 1]) for i in range(0, len(coords_list), 2)]


def draw_predictions_on_image(image: Image.Image, polygons_str: str, color: tuple[int, int, int]) -> Image.Image:
    """Draw polygon predictions on a copy of the image."""
    polygons = parse_polygon_string(polygons_str)
    if not polygons:
        return image

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    for index, coords in enumerate(polygons):
        points = polygon_points(coords)
        try:
            draw.polygon(points, outline=color + (255,), fill=color + (50,), width=2)
        except TypeError:
            draw.polygon(points, outline=color + (255,), fill=color + (50,))
        if points:
            draw.text((points[0][0], points[0][1] - 10), f"T{index + 1}", fill=color + (255,))

    return overlay
