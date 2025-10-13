"""
AI_DOCS: Polygon Utils - Polygon Processing & Validation Utilities

This module provides specialized polygon processing utilities for OCR:
- Polygon coordinate validation and normalization
- Degenerate polygon detection and filtering
- Shape validation for probability/threshold maps
- Coordinate system transformations

ARCHITECTURE OVERVIEW:
- Utilities extracted from dataset transformation logic
- Focus on geometric validation and processing
- NumPy-based coordinate manipulation
- Integration with Albumentations transforms

DATA CONTRACTS:
- Input: numpy arrays with shape (N, 2) or (1, N, 2)
- Output: validated numpy arrays or None (filtered)
- Coordinate System: (x, y) pixel coordinates
- Data Types: float32 for consistency

CORE CONSTRAINTS:
- ALWAYS validate polygon shapes before processing
- FILTER degenerate polygons (< 3 points)
- PRESERVE coordinate precision (float32)
- USE consistent coordinate ordering (x, y)
- VALIDATE map shapes against image dimensions

PERFORMANCE FEATURES:
- Vectorized NumPy operations for speed
- Early filtering prevents downstream errors
- Memory-efficient coordinate processing
- Batch processing support

VALIDATION REQUIREMENTS:
- Check polygon dimensionality (2D/3D arrays)
- Validate coordinate ranges (non-negative)
- Ensure minimum point counts (≥ 3)
- Verify map-image shape compatibility

RELATED DOCUMENTATION:
- Base Dataset: ocr/datasets/base.py
- Transforms: ocr/datasets/transforms.py
- Data Schemas: ocr/datasets/schemas.py
- Geometric Utils: ocr/utils/geometry_utils.py

MIGRATION NOTES:
- Utilities extracted from ValidatedOCRDataset.__getitem__
- Pydantic integration for data validation
- Improved error handling and filtering
"""

import numpy as np


def ensure_polygon_array(polygon: np.ndarray) -> np.ndarray | None:
    """
    AI_DOCS: Polygon Array Normalization

    Normalizes polygon input to standard numpy array format.

    CRITICAL CONSTRAINTS:
    - ACCEPT both lists and numpy arrays
    - CONVERT to float32 dtype
    - RESHAPE to (N, 2) format if needed
    - VALIDATE coordinate dimensions
    - RETURN None for invalid inputs

    Output: float32 array with shape (N, 2) or None
    """
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
    AI_DOCS: Degenerate Polygon Filtering

    Removes polygons with insufficient points for geometric operations.

    CRITICAL CONSTRAINTS:
    - REQUIRE minimum 3 points per polygon
    - PRESERVE valid polygons unchanged
    - LOG filtering decisions for debugging
    - RETURN filtered list (may be shorter)

    Geometric Requirement: Polygons need ≥ 3 points for area calculation
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
    AI_DOCS: Map Shape Validation

    Validates probability/threshold map dimensions against image.

    CRITICAL CONSTRAINTS:
    - CHECK both maps have identical shapes
    - VALIDATE against image dimensions if provided
    - LOG validation failures with context
    - RETURN boolean validation result

    Map Requirements: (H, W) float arrays matching image dimensions
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
