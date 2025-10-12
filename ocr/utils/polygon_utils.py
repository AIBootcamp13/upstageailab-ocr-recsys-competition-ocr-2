"""Polygon utilities for OCR dataset processing."""

import numpy as np


def ensure_polygon_array(polygon: np.ndarray) -> np.ndarray | None:
    """Normalize polygon arrays to have shape ``(N, 2)``."""
    if polygon is None:
        return None

    polygon_array = polygon if isinstance(polygon, np.ndarray) else np.asarray(polygon, dtype=np.float32)
    polygon_array = np.asarray(polygon_array, dtype=np.float32)

    if polygon_array.size == 0:
        return polygon_array.reshape(0, 2)

    if polygon_array.ndim == 1:
        if polygon_array.size % 2 != 0:
            raise ValueError("Polygon coordinate list must contain an even number of values")
        return polygon_array.reshape(-1, 2)

    if polygon_array.ndim == 2:
        if polygon_array.shape[1] == 2:
            return polygon_array
        return polygon_array.reshape(-1, 2)

    if polygon_array.ndim == 3 and polygon_array.shape[0] == 1:
        reshaped = polygon_array[0]
        return reshaped if reshaped.ndim == 2 else reshaped.reshape(-1, 2)

    return polygon_array.reshape(-1, 2)


def filter_degenerate_polygons(
    polygons: list[np.ndarray],
    min_side: float = 1.0,
) -> list[np.ndarray]:
    """
    Filter out degenerate polygons that don't meet minimum requirements.

    Args:
        polygons: List of polygon arrays
        min_side: Minimum side length for valid polygons

    Returns:
        List of valid polygons
    """
    from collections import Counter

    removed_counts = Counter(
        {
            "too_few_points": 0,
            "too_small": 0,
            "zero_span": 0,
            "empty": 0,
            "none": 0,
        }
    )
    filtered = []
    for polygon in polygons:
        if polygon is None:
            removed_counts["none"] += 1
            continue
        if polygon.size == 0:
            removed_counts["empty"] += 1
            continue

        reshaped = polygon.reshape(-1, 2)
        if reshaped.shape[0] < 3:
            removed_counts["too_few_points"] += 1
            continue

        width_span = float(reshaped[:, 0].max() - reshaped[:, 0].min())
        height_span = float(reshaped[:, 1].max() - reshaped[:, 1].min())

        rounded = np.rint(reshaped).astype(np.int32, copy=False)
        width_span_int = int(rounded[:, 0].ptp())
        height_span_int = int(rounded[:, 1].ptp())

        if width_span < min_side or height_span < min_side:
            removed_counts["too_small"] += 1
            continue

        if width_span_int == 0 or height_span_int == 0:
            removed_counts["zero_span"] += 1
            continue

        filtered.append(polygon)

    total_removed = sum(removed_counts.values())
    if total_removed > 0:
        import logging

        logger = logging.getLogger(__name__)
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "\nFiltered %d degenerate polygons (too_few_points=%d, too_small=%d, zero_span=%d, empty=%d, none=%d)",
                total_removed,
                removed_counts["too_few_points"],
                removed_counts["too_small"],
                removed_counts["zero_span"],
                removed_counts["empty"],
                removed_counts["none"],
            )

    return filtered


def validate_map_shapes(
    prob_map: np.ndarray,
    thresh_map: np.ndarray,
    image_height: int | None,
    image_width: int | None,
    filename: str,
) -> bool:
    """
    Validate that map shapes are correct and compatible.

    Args:
        prob_map: Probability map array
        thresh_map: Threshold map array
        image_height: Expected image height
        image_width: Expected image width
        filename: Filename for logging

    Returns:
        True if validation passes, False otherwise
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        # Check that both maps exist and are arrays
        if prob_map is None or thresh_map is None:
            logger.warning(f"Map validation failed for {filename}: prob_map or thresh_map is None")
            return False

        # Check shapes are compatible
        if prob_map.shape != thresh_map.shape:
            logger.warning(f"Map validation failed for {filename}: prob_map shape {prob_map.shape} != thresh_map shape {thresh_map.shape}")
            return False

        # Check expected shape format (should be CHW with C=1)
        if len(prob_map.shape) != 3 or prob_map.shape[0] != 1:
            logger.warning(f"Map validation failed for {filename}: prob_map shape {prob_map.shape} should be (1, H, W)")
            return False

        # If image dimensions provided, check they match
        if image_height is not None and image_width is not None:
            expected_shape = (1, image_height, image_width)
            if prob_map.shape != expected_shape:
                logger.warning(
                    f"Map validation failed for {filename}: prob_map shape {prob_map.shape} doesn't match image dimensions {expected_shape}"
                )
                return False

        return True

    except Exception as e:
        logger.warning(f"Map validation failed for {filename}: {e}")
        return False
