"""
AI_DOCS: OCR Dataset Transforms - Data Augmentation & Geometric Processing

This module implements the ValidatedDBTransforms class, responsible for:
- Image and polygon geometric transformations using Albumentations
- Coordinate system transformations and inverse matrices
- Polygon validation and filtering during augmentation
- Backward compatibility with legacy transform interfaces

ARCHITECTURE OVERVIEW:
- Uses Albumentations for image transformations
- Maintains polygon-keypoint synchronization during geometric transforms
- Calculates inverse transformation matrices for coordinate mapping
- Validates all inputs and outputs with Pydantic models

DATA CONTRACTS:
- Input: TransformInput (Pydantic model) or legacy formats
- Output: TransformOutput (Pydantic model) -> OrderedDict[str, Any]
- Polygons: List[PolygonData] with points as numpy arrays (N, 2)
- Images: numpy arrays (H, W, C) or (H, W)

CORE CONSTRAINTS:
- NEVER modify the __call__ method signature or return type
- ALWAYS synchronize polygon keypoints with image transformations
- PRESERVE coordinate transformation matrices for accurate mapping
- USE Pydantic models for all data validation
- MAINTAIN backward compatibility with legacy interfaces

GEOMETRIC TRANSFORMATION REQUIREMENTS:
- Keypoints must be clamped to image boundaries
- Polygon coordinates must be transformed alongside images
- Inverse matrices must be calculated for coordinate remapping
- Degenerate polygons ( < 3 points) must be filtered out

PERFORMANCE FEATURES:
- Conditional normalization skips pre-normalized images
- Efficient keypoint-polygon synchronization
- Memory-efficient transformation pipelines

VALIDATION REQUIREMENTS:
- All polygons must pass PolygonData validation
- Images must have valid shapes and dtypes
- Transformation matrices must be invertible
- Output must conform to TransformOutput schema

RELATED DOCUMENTATION:
- Data contracts: ocr/validation/models.py
- Schemas: ocr/datasets/schemas.py
- Albumentations: https://albumentations.ai/
- Geometric utils: ocr/utils/geometry_utils.py
- Base dataset: ocr/datasets/base.py

MIGRATION NOTES:
- ValidatedDBTransforms replaces legacy DBTransforms
- Pydantic validation added throughout pipeline
- Backward compatibility maintained for existing code
"""

from collections import OrderedDict
from typing import Any

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from pydantic import ValidationError

from ocr.datasets.schemas import ImageMetadata, PolygonData, TransformInput, TransformOutput
from ocr.utils.geometry_utils import calculate_cropbox, calculate_inverse_transform


class ConditionalNormalize(A.ImageOnlyTransform):
    """
    Normalize image only if it hasn't been pre-normalized.

    AI_DOCS: Conditional Normalization Transform
    This Albumentations transform:
    - Skips normalization for pre-normalized float32 images
    - Applies ImageNet normalization to uint8 images
    - Improves performance by avoiding redundant normalization
    - Uses heuristic: float32 + max < 10.0 = already normalized

    DO NOT MODIFY:
    - The normalization heuristic (breaks caching performance)
    - The ImageNet mean/std values (changes model expectations)
    - The dtype checking logic (affects normalization decisions)
    """

    def __init__(self, mean=(0.5900, 0.5610, 0.5292), std=(0.2136, 0.2290, 0.2511), always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def apply(self, img, **params):
        # AI_DOCS: Normalization Logic
        # Check if image is already normalized (float32 dtype + small values)
        # This heuristic works for cached images that are pre-normalized
        if img.dtype == np.float32 and img.max() < 10.0:
            # Image is already normalized, return as-is
            return img

        # Image is uint8, need to normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return img

    def get_transform_init_args_names(self):
        return ("mean", "std")


class ValidatedDBTransforms:
    """
    AI_DOCS: ValidatedDBTransforms - Core Transformation Pipeline

    This class implements the complete OCR data transformation pipeline:
    - Image geometric transformations using Albumentations
    - Polygon coordinate synchronization with image transforms
    - Inverse matrix calculation for coordinate remapping
    - Pydantic validation of all inputs and outputs

    CONSTRAINTS FOR AI ASSISTANTS:
    - DO NOT modify __call__ method signature or return type
    - ALWAYS maintain polygon-keypoint synchronization
    - PRESERVE inverse matrix calculations
    - USE Pydantic models for validation
    - MAINTAIN backward compatibility with legacy interfaces

    Data Flow: TransformInput -> Albumentations -> TransformOutput -> OrderedDict
    """

    def __init__(self, transforms, keypoint_params):
        self.transform = A.Compose([*transforms, ToTensorV2()], keypoint_params=keypoint_params)

    def __call__(self, data: TransformInput | dict[str, Any] | np.ndarray, polygons: list[np.ndarray] | None = None) -> OrderedDict:
        """
        Apply transforms to image and polygons.

        AI_DOCS: CRITICAL TRANSFORMATION METHOD
        This is the PRIMARY transformation method. AI assistants MUST:
        - NEVER change return type (OrderedDict[str, Any])
        - ALWAYS synchronize polygons with image transformations
        - PRESERVE inverse matrix calculations for coordinate mapping
        - USE Pydantic models for all data validation
        - MAINTAIN backward compatibility with legacy signatures

        Data Flow (DO NOT MODIFY):
        1. Coerce input to TransformInput (Pydantic model)
        2. Extract image and polygons from validated input
        3. Convert polygons to keypoints for Albumentations
        4. Clamp keypoints to image boundaries
        5. Apply Albumentations transform (image + keypoints)
        6. Calculate inverse transformation matrix
        7. Reconstruct polygons from transformed keypoints
        8. Filter degenerate polygons (< 3 points)
        9. Validate output with TransformOutput model
        10. Return OrderedDict with validated data

        Args:
            data: TransformInput payload or raw numpy image for backwards compatibility
            polygons: Legacy list of polygon arrays when using legacy signature

        Returns:
            OrderedDict with:
                - image: Transformed image tensor
                - polygons: List of transformed polygons with shape (1, N, 2)
                - inverse_matrix: Matrix for coordinate transformation
        """
        # AI_DOCS: Step 1 - Input Coercion
        # Convert legacy inputs to standardized TransformInput
        transform_input = self._coerce_input(data, polygons)

        # AI_DOCS: Step 2 - Extract Validated Data
        # Use Pydantic models for type safety
        image = transform_input.image
        polygon_models = transform_input.polygons or []
        metadata_payload: dict[str, Any] | None = None
        if transform_input.metadata is not None:
            metadata_payload = transform_input.metadata.model_dump()
            if metadata_payload.get("path") is not None:  # type: ignore
                metadata_payload["path"] = str(metadata_payload["path"])  # type: ignore

        height, width = image.shape[:2]

        # AI_DOCS: Step 3 - Polygon to Keypoint Conversion
        # Convert polygons to keypoints for Albumentations compatibility
        # Each polygon becomes a flat list of (x,y) coordinates
        keypoints = [point for poly in polygon_models for point in poly.points.reshape(-1, 2)]
        keypoints = self.clamp_keypoints(keypoints, width, height)

        # AI_DOCS: Step 4 - Apply Albumentations Transform
        # Transform both image and keypoints simultaneously
        # This ensures geometric consistency between image and polygons
        transformed = self.transform(image=image, keypoints=keypoints)
        transformed_image = transformed["image"]
        transformed_keypoints = transformed["keypoints"]
        metadata = transformed.get("metadata")

        # AI_DOCS: Step 5 - Metadata Handling
        # Merge input metadata with transformation metadata
        if metadata is not None and not isinstance(metadata, dict):
            try:
                metadata = dict(metadata)
            except Exception:
                metadata = {"metadata": metadata}

        if metadata_payload is not None:
            if metadata is None:
                metadata = metadata_payload
            else:
                combined_metadata = metadata_payload.copy()
                combined_metadata.update(metadata)
                metadata = combined_metadata

        # AI_DOCS: Step 6 - Inverse Matrix Calculation
        # Calculate matrix for transforming coordinates back to original space
        # CRITICAL for evaluation and prediction coordinate mapping
        _, new_height, new_width = transformed_image.shape
        crop_box = calculate_cropbox((width, height), max(new_height, new_width))
        inverse_matrix = calculate_inverse_transform((width, height), (new_width, new_height), crop_box=crop_box)

        # AI_DOCS: Step 7 - Keypoint to Polygon Reconstruction
        # Convert transformed keypoints back to polygon format
        # BUG FIX (BUG-2025-004): Correct polygon point count extraction
        transformed_polygons = []
        index = 0
        if polygon_models:
            for polygon_idx, polygon in enumerate(polygon_models):
                num_points = polygon.points.shape[0]

                keypoint_slice = transformed_keypoints[index : index + num_points]
                index += len(keypoint_slice)

                if len(keypoint_slice) < 3:
                    import logging

                    logging.debug(
                        "Skipping degenerate polygon at index %d: only %d points after transform",
                        polygon_idx,
                        len(keypoint_slice),
                    )
                    continue

                polygon_array = np.asarray(keypoint_slice, dtype=np.float32).reshape(1, -1, 2)
                transformed_polygons.append(polygon_array)

        # AI_DOCS: Step 8 - Output Construction
        # Build OrderedDict with required fields
        output = OrderedDict(
            image=transformed_image,
            polygons=transformed_polygons,
            inverse_matrix=inverse_matrix,
        )

        if metadata is not None:
            output["metadata"] = metadata

        # AI_DOCS: Step 9 - Final Validation
        # Validate output against TransformOutput schema
        validated_output = TransformOutput.model_validate(output)

        # AI_DOCS: Step 10 - Return Validated Result
        # Convert back to OrderedDict for PyTorch compatibility
        result: OrderedDict[str, Any] = OrderedDict(
            image=validated_output.image,
            polygons=validated_output.polygons,
            inverse_matrix=validated_output.inverse_matrix,
        )

        if validated_output.metadata is not None:
            result["metadata"] = validated_output.metadata

        return result

    def clamp_keypoints(self, keypoints: list, img_width: int, img_height: int) -> list:
        clamped_keypoints = []
        for kp in keypoints:
            x, y = kp[:2]
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            clamped_keypoints.append((x, y) + tuple(kp[2:]))
        return clamped_keypoints

    def _coerce_input(
        self,
        data: TransformInput | dict[str, Any] | np.ndarray,
        polygons: list[np.ndarray] | None,
    ) -> TransformInput:
        """
        AI_DOCS: Input Coercion Method
        Converts various input formats to standardized TransformInput.
        This method provides backward compatibility with legacy interfaces.

        Input Formats Supported:
        - TransformInput (Pydantic model) - preferred modern format
        - dict[str, Any] - validated and converted to TransformInput
        - np.ndarray - legacy format with separate polygons parameter

        DO NOT REMOVE:
        - Support for legacy np.ndarray inputs (breaks existing code)
        - Pydantic validation of dict inputs
        - Proper error messages for invalid inputs
        """
        if isinstance(data, TransformInput):
            return data

        if isinstance(data, np.ndarray):
            return self._build_from_legacy(image=data, polygons=polygons)

        try:
            return TransformInput.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"Invalid transform input payload: {exc}") from exc

    def _build_from_legacy(self, image: np.ndarray, polygons: list[np.ndarray] | None) -> TransformInput:
        """
        AI_DOCS: Legacy Input Builder
        Converts legacy numpy array inputs to TransformInput format.
        This maintains backward compatibility with existing training code.

        Validation Requirements:
        - Images must be numpy arrays or PIL Images
        - Polygons must be list of numpy arrays with correct shapes
        - All polygons are validated using PolygonData models
        - Invalid polygons are silently filtered out

        DO NOT MODIFY:
        - Type checking and validation logic
        - Polygon filtering for invalid shapes
        - ImageMetadata creation from legacy inputs
        """
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            image = np.array(image)

        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array or PIL Image, got {type(image)}")

        polygon_models = None
        if polygons is not None:
            if not isinstance(polygons, list):
                raise TypeError(f"polygons must be a list, got {type(polygons)}")
            # Filter out invalid polygons before creating PolygonData objects
            valid_polygons = []
            for idx, poly in enumerate(polygons):
                if not isinstance(poly, np.ndarray):
                    raise TypeError(f"polygon at index {idx} must be numpy array, got {type(poly)}")

                if poly.dtype not in (np.float32, np.float64, np.int32, np.int64):
                    raise TypeError(f"polygon at index {idx} must be numeric array, got dtype {poly.dtype}")

                if poly.ndim not in (2, 3):
                    raise ValueError(f"polygon at index {idx} must be 2D or 3D array, got {poly.ndim}D with shape {poly.shape}")

                if poly.ndim == 2 and poly.shape[1] != 2:
                    raise ValueError(f"polygon at index {idx} must have shape (N, 2), got {poly.shape}")

                if poly.ndim == 3 and (poly.shape[0] != 1 or poly.shape[2] != 2):
                    raise ValueError(f"polygon at index {idx} must have shape (1, N, 2), got {poly.shape}")
                try:
                    # Attempt to create PolygonData - this will validate the polygon
                    valid_polygons.append(PolygonData(points=poly))
                except ValidationError:
                    # Skip invalid polygons that fail validation
                    continue
            polygon_models = valid_polygons

        if image.ndim not in (2, 3):
            raise ValueError(f"Image must be 2D or 3D array, got shape {image.shape}")

        height, width = map(int, image.shape[:2])

        metadata = ImageMetadata(
            original_shape=(height, width),
            dtype=str(image.dtype),
        )

        return TransformInput(image=image, polygons=polygon_models, metadata=metadata)


# Backwards compatibility for existing imports
DBTransforms = ValidatedDBTransforms


# AI_DOCS: END OF FILE - Critical Reminders for AI Assistants
#
# =======================================================================
# VALIDATEDDBTRANSFORMS - AI ASSISTANT CONSTRAINTS & REQUIREMENTS
# =======================================================================
#
# BEFORE MAKING ANY CHANGES TO THIS FILE:
#
# 1. DATA CONTRACTS (MANDATORY):
#    - Input: TransformInput (Pydantic model) or legacy formats
#    - Output: TransformOutput.model_dump() -> OrderedDict[str, Any]
#    - Polygons: List[PolygonData] with points as numpy arrays (N, 2)
#    - Images: numpy arrays (H, W, C) with proper dtypes
#
# 2. METHOD SIGNATURES (DO NOT CHANGE):
#    - __call__(data, polygons=None) -> OrderedDict[str, Any]
#    - _coerce_input(data, polygons) -> TransformInput
#    - _build_from_legacy(image, polygons) -> TransformInput
#    - clamp_keypoints(keypoints, width, height) -> list
#
# 3. GEOMETRIC TRANSFORMATION RULES (CRITICAL):
#    - Polygons MUST be converted to keypoints for Albumentations
#    - Keypoints MUST be clamped to image boundaries
#    - Inverse matrices MUST be calculated for coordinate remapping
#    - Degenerate polygons (< 3 points) MUST be filtered out
#    - Polygon-keypoint synchronization MUST be maintained
#
# 4. BACKWARD COMPATIBILITY (PRESERVE):
#    - Support for legacy np.ndarray inputs
#    - DBTransforms alias for ValidatedDBTransforms
#    - All existing transform behaviors and outputs
#
# 5. VALIDATION (MANDATORY):
#    - Use Pydantic models for all data structures
#    - Validate inputs in _coerce_input method
#    - Validate outputs with TransformOutput.model_validate()
#    - Handle ValidationError exceptions appropriately
#
# 6. CONDITIONAL NORMALIZE (PERFORMANCE):
#    - DO NOT change normalization heuristic (float32 + max < 10.0)
#    - DO NOT modify ImageNet mean/std values
#    - PRESERVE conditional logic for cached images
#
# 7. ALBUMENTATIONS INTEGRATION:
#    - Use A.Compose with keypoint_params for polygon support
#    - Apply transforms to both image and keypoints simultaneously
#    - Handle metadata merging from transformations
#
# =======================================================================
# RELATED DOCUMENTATION (MUST CONSULT):
# =======================================================================
#
# - Data Contracts: ocr/validation/models.py
# - Schemas: ocr/datasets/schemas.py
# - Base Dataset: ocr/datasets/base.py
# - Geometry Utils: ocr/utils/geometry_utils.py
# - Albumentations: https://albumentations.ai/
# - Configuration: configs/**/*.yaml
#
# =======================================================================
# COMMON AI MISTAKES TO AVOID:
# =======================================================================
#
# ❌ Changing __call__ return type from OrderedDict[str, Any]
# ❌ Breaking polygon-keypoint synchronization
# ❌ Removing inverse matrix calculations
# ❌ Modifying ConditionalNormalize heuristic
# ❌ Breaking legacy input format support
# ❌ Skipping Pydantic validation steps
# ❌ Changing Albumentations keypoint handling
# ❌ Removing degenerate polygon filtering
#
# =======================================================================
# GEOMETRIC TRANSFORMATION PIPELINE:
# =======================================================================
#
# 1. Input Coercion: Legacy formats → TransformInput (Pydantic)
# 2. Data Extraction: TransformInput → image + polygons + metadata
# 3. Polygon→Keypoint: Convert polygons to flat (x,y) coordinate list
# 4. Keypoint Clamping: Ensure coordinates within image boundaries
# 5. Albumentations: Apply transforms to image + keypoints together
# 6. Inverse Matrix: Calculate coordinate transformation matrix
# 7. Keypoint→Polygon: Reconstruct polygons from transformed keypoints
# 8. Polygon Filtering: Remove degenerate polygons (< 3 points)
# 9. Output Validation: TransformOutput.model_validate()
# 10. Format Conversion: Pydantic → OrderedDict for PyTorch
#
# =======================================================================
