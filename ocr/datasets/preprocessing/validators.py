"""Shared validation utilities for the preprocessing pipeline."""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

T = TypeVar("T")


class ImageValidator:
    """Reusable validator for numpy array images with common validation patterns."""

    @staticmethod
    def validate_image_array(
        array: Any, name: str = "image", min_dims: int = 2, max_dims: int = 3, allow_none: bool = False
    ) -> np.ndarray | None:
        """Validate a numpy array as an image.

        Args:
            array: Input array to validate
            name: Name of the field for error messages
            min_dims: Minimum number of dimensions required
            max_dims: Maximum number of dimensions allowed
            allow_none: Whether None is acceptable

        Returns:
            Validated numpy array or None

        Raises:
            ValueError: If validation fails
        """
        if allow_none and array is None:
            return None

        if array is None:
            raise ValueError(f"{name} cannot be None")

        if not isinstance(array, np.ndarray):
            try:
                array = np.array(array)
            except Exception as e:
                raise ValueError(f"{name} must be convertible to numpy array: {e}")

        if array.ndim < min_dims:
            raise ValueError(f"{name} must have at least {min_dims} dimensions, got {array.ndim}")

        if array.ndim > max_dims:
            raise ValueError(f"{name} cannot have more than {max_dims} dimensions, got {array.ndim}")

        if array.size == 0:
            raise ValueError(f"{name} cannot be empty")

        return array

    @staticmethod
    def validate_image_shape(array: np.ndarray, expected_shape: tuple[int, ...] | None = None, name: str = "image") -> np.ndarray:
        """Validate image shape constraints.

        Args:
            array: Input numpy array
            expected_shape: Expected shape (height, width, channels) or None for any shape
            name: Name of the field for error messages

        Returns:
            Validated numpy array

        Raises:
            ValueError: If shape validation fails
        """
        if expected_shape is not None:
            if len(array.shape) != len(expected_shape):
                raise ValueError(f"{name} shape must be {expected_shape}, got {array.shape}")
            for i, (actual, expected) in enumerate(zip(array.shape, expected_shape, strict=True)):
                if expected is not None and actual != expected:
                    raise ValueError(f"{name} dimension {i} must be {expected}, got {actual}")

        # Validate reasonable bounds
        for i, dim in enumerate(array.shape):
            if dim <= 0:
                raise ValueError(f"{name} dimension {i} must be positive, got {dim}")
            if dim > 10000:
                raise ValueError(f"{name} dimension {i} cannot exceed 10000, got {dim}")

        return array

    @staticmethod
    def validate_image_dtype(array: np.ndarray, allowed_dtypes: list[np.dtype] | None = None, name: str = "image") -> np.ndarray:
        """Validate image data type.

        Args:
            array: Input numpy array
            allowed_dtypes: List of allowed dtypes or None for any dtype
            name: Name of the field for error messages

        Returns:
            Validated numpy array

        Raises:
            ValueError: If dtype validation fails
        """
        if allowed_dtypes is None:
            allowed_dtypes = [np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32), np.dtype(np.float64)]

        if array.dtype not in allowed_dtypes:
            dtype_names = [str(dt) for dt in allowed_dtypes]
            raise ValueError(f"{name} dtype must be one of {dtype_names}, got {array.dtype}")

        return array

    @staticmethod
    def validate_corners_array(corners: Any, name: str = "corners", expected_points: int | None = None) -> np.ndarray | None:
        """Validate corner coordinates array.

        Args:
            corners: Corner coordinates to validate
            name: Name of the field for error messages
            expected_points: Expected number of corner points or None

        Returns:
            Validated corners array or None

        Raises:
            ValueError: If validation fails
        """
        if corners is None:
            return None

        corners = ImageValidator.validate_image_array(corners, name=name, min_dims=2, max_dims=2, allow_none=False)

        # Should be N x 2 (points x coordinates)
        if corners.shape[1] != 2:
            raise ValueError(f"{name} must have shape (N, 2), got {corners.shape}")

        if expected_points is not None and corners.shape[0] != expected_points:
            raise ValueError(f"{name} must have {expected_points} points, got {corners.shape[0]}")

        # Validate coordinate ranges (reasonable bounds)
        if corners.size > 0:
            min_coords = corners.min(axis=0)
            max_coords = corners.max(axis=0)

            # Coordinates should be non-negative and reasonable
            if min_coords[0] < -1000 or min_coords[1] < -1000:
                raise ValueError(f"{name} coordinates seem unreasonably small: {min_coords}")

            if max_coords[0] > 50000 or max_coords[1] > 50000:
                raise ValueError(f"{name} coordinates seem unreasonably large: {max_coords}")

        return corners

    @staticmethod
    def validate_transformation_matrix(
        matrix: Any, expected_shape: tuple[int, int] = (3, 3), name: str = "transformation_matrix"
    ) -> np.ndarray | None:
        """Validate transformation matrix (affine, perspective, etc.).

        Args:
            matrix: Matrix to validate
            expected_shape: Expected matrix shape
            name: Name of the field for error messages

        Returns:
            Validated matrix or None

        Raises:
            ValueError: If validation fails
        """
        if matrix is None:
            return None

        matrix = ImageValidator.validate_image_array(matrix, name=name, min_dims=2, max_dims=2, allow_none=False)

        if matrix.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {matrix.shape}")

        # Check if matrix is numeric
        if not np.issubdtype(matrix.dtype, np.number):
            raise ValueError(f"{name} must contain numeric values, got dtype {matrix.dtype}")

        return matrix


class ContractValidator:
    """Validator for data contract enforcement between components."""

    @staticmethod
    def validate_image_input_contract(image: Any, contract_name: str = "image_input") -> np.ndarray:
        """Validate image input against standard contract.

        Contract: Image must be numpy array with shape (H, W, C) where:
        - H, W > 0 and <= 10000
        - C in [1, 2, 3, 4]
        - dtype is uint8, uint16, float32, or float64

        Args:
            image: Image to validate
            contract_name: Name for error messages

        Returns:
            Validated image array

        Raises:
            ValueError: If contract is violated
        """
        image = ImageValidator.validate_image_array(image, contract_name, min_dims=2, max_dims=3)

        # Validate shape constraints
        if len(image.shape) == 2:
            height, width = image.shape
            channels = 1
        else:
            height, width, channels = image.shape[:3]

        if height <= 0 or width <= 0:
            raise ValueError(f"{contract_name} dimensions must be positive, got {image.shape}")

        if height > 10000 or width > 10000:
            raise ValueError(f"{contract_name} dimensions cannot exceed 10000, got {image.shape}")

        if channels not in [1, 2, 3, 4]:
            raise ValueError(f"{contract_name} must have 1-4 channels, got {channels}")

        # Validate dtype
        allowed_dtypes = [np.uint8, np.uint16, np.float32, np.float64]
        if image.dtype not in allowed_dtypes:
            dtype_names = [str(dt) for dt in allowed_dtypes]
            raise ValueError(f"{contract_name} dtype must be one of {dtype_names}, got {image.dtype}")

        return image

    @staticmethod
    def validate_preprocessing_result_contract(result: dict[str, Any], contract_name: str = "preprocessing_result") -> dict[str, Any]:
        """Validate preprocessing result against standard contract.

        Contract: Result must be dict with required keys:
        - "image": numpy array with valid image contract
        - "metadata": dict with processing information

        Args:
            result: Result to validate
            contract_name: Name for error messages

        Returns:
            Validated result dict

        Raises:
            ValueError: If contract is violated
        """
        if not isinstance(result, dict):
            raise ValueError(f"{contract_name} must be a dictionary")

        required_keys = ["image", "metadata"]
        for key in required_keys:
            if key not in result:
                raise ValueError(f"{contract_name} must contain '{key}' key")

        # Validate image
        ContractValidator.validate_image_input_contract(result["image"], f"{contract_name}.image")

        # Validate metadata (basic structure check)
        if not isinstance(result["metadata"], dict):
            raise ValueError(f"{contract_name}.metadata must be a dictionary")

        return result

    @staticmethod
    def validate_detection_result_contract(detection_result: Any, contract_name: str = "detection_result") -> dict[str, Any]:
        """Validate detection result against standard contract.

        Contract: Detection result must contain:
        - "corners": numpy array of shape (4, 2) or None
        - "confidence": float between 0.0 and 1.0
        - "method": string indicating detection method

        Args:
            detection_result: Result to validate
            contract_name: Name for error messages

        Returns:
            Validated detection result

        Raises:
            ValueError: If contract is violated
        """
        if not isinstance(detection_result, dict):
            raise ValueError(f"{contract_name} must be a dictionary")

        # Validate corners
        if "corners" in detection_result:
            corners = detection_result["corners"]
            if corners is not None:
                corners = ImageValidator.validate_corners_array(corners, f"{contract_name}.corners", 4)
                detection_result["corners"] = corners

        # Validate confidence
        if "confidence" in detection_result:
            confidence = detection_result["confidence"]
            if not isinstance(confidence, int | float):
                raise ValueError(f"{contract_name}.confidence must be numeric")
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"{contract_name}.confidence must be between 0.0 and 1.0, got {confidence}")

        # Validate method
        if "method" in detection_result:
            method = detection_result["method"]
            if not isinstance(method, str):
                raise ValueError(f"{contract_name}.method must be a string")

        return detection_result


# Custom Pydantic Types


class NumpyArray:
    """Pydantic-compatible numpy array type with validation."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """Generate Pydantic core schema for numpy arrays."""
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def _validate(cls, value: Any) -> np.ndarray:
        """Validate value as numpy array."""
        result = ImageValidator.validate_image_array(value, "numpy_array", allow_none=False)
        assert result is not None  # Should never be None when allow_none=False
        return result

    @classmethod
    def __pydantic_serializer__(cls, value: np.ndarray) -> Any:
        """Serialize numpy array for JSON/etc."""
        return value.tolist() if hasattr(value, "tolist") else value


class ImageArray(NumpyArray):
    """Pydantic-compatible image array type with image-specific validation."""

    @classmethod
    def _validate(cls, value: Any) -> np.ndarray:
        """Validate value as image array."""
        return ContractValidator.validate_image_input_contract(value, "image_array")


class CornerArray(NumpyArray):
    """Pydantic-compatible corner coordinate array type."""

    @classmethod
    def _validate(cls, value: Any) -> np.ndarray:
        """Validate value as corner coordinates."""
        result = ImageValidator.validate_corners_array(value, "corner_array")
        assert result is not None  # Should never be None for required validation
        return result


class TransformationMatrix(NumpyArray):
    """Pydantic-compatible transformation matrix type."""

    @classmethod
    def _validate(cls, value: Any) -> np.ndarray:
        """Validate value as transformation matrix."""
        result = ImageValidator.validate_transformation_matrix(value, (3, 3), "transformation_matrix")
        assert result is not None  # Should never be None for required validation
        return result


__all__ = [
    "ImageValidator",
    "ContractValidator",
    "NumpyArray",
    "ImageArray",
    "CornerArray",
    "TransformationMatrix",
]
