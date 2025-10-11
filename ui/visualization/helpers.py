from __future__ import annotations

"""Shared helpers for visualization components."""

import re
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
        tokens = [token for token in re.split(r"[\s,]+", raw_polygon.strip()) if token]
        for token in tokens:
            try:
                coords.append(float(token))
            except ValueError:
                coords = []
                break
        if len(coords) >= 8 and len(coords) % 2 == 0:
            polygons.append(coords)
    return polygons


def validate_polygons(polygons_str: str) -> tuple[list[Polygon], dict[str, int]]:
    """Validate polygons and return details about filtering."""
    polygons: list[Polygon] = []
    stats = {"total": 0, "valid": 0, "too_small": 0, "odd_coords": 0, "parse_errors": 0}

    if not polygons_str.strip():
        return polygons, stats

    raw_polygons = polygons_str.split("|")
    stats["total"] = len(raw_polygons)

    for raw_polygon in raw_polygons:
        coords: Polygon = []
        tokens = [token for token in re.split(r"[\s,]+", raw_polygon.strip()) if token]
        if not tokens:
            stats["parse_errors"] += 1
            continue

        has_parse_error = False

        for token in tokens:
            try:
                coords.append(float(token))
            except ValueError:
                has_parse_error = True
                break

        if has_parse_error:
            stats["parse_errors"] += 1
        elif len(coords) < 8:
            stats["too_small"] += 1
        elif len(coords) % 2 != 0:
            stats["odd_coords"] += 1
        else:
            polygons.append(coords)
            stats["valid"] += 1

    return polygons, stats


def polygon_points(coords: Iterable[float]) -> list[tuple[float, float]]:
    """Convert a flat coordinate list into point tuples."""
    coords_list = list(coords)
    return [(coords_list[i], coords_list[i + 1]) for i in range(0, len(coords_list), 2)]


def transform_polygons_for_ccw90(polygons: list[Polygon], width: int, height: int) -> list[Polygon]:
    """Transform polygons for 90-degree counter-clockwise rotation."""
    transformed = []
    for coords in polygons:
        new_coords = []
        for i in range(0, len(coords), 2):
            x, y = coords[i], coords[i + 1]
            new_x = y
            new_y = width - x
            new_coords.extend([new_x, new_y])
        transformed.append(new_coords)
    return transformed


def draw_predictions_on_image(
    image: Image.Image, polygons_str: str, color: tuple[int, int, int], rotate_ccw90: bool = False
) -> Image.Image:
    """Draw polygon predictions on a copy of the image."""
    polygons = parse_polygon_string(polygons_str)
    if not polygons:
        return image

    if rotate_ccw90:
        original_size = image.size
        image = image.rotate(-90, expand=True)
        polygons = transform_polygons_for_ccw90(polygons, original_size[0], original_size[1])

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
