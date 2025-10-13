# Preprocessing Module Data Contracts

## Overview
This document describes the data contracts implemented during the Pydantic validation refactor of the preprocessing module.

## Core Contracts

### ImageInputContract
Validates input images for preprocessing components.
- **Fields**: `image: np.ndarray`
- **Validation**: Must be numpy array, 2-3 dimensions, 1-4 channels, non-empty

### PreprocessingResultContract
Validates preprocessing pipeline results.
- **Fields**: `image: np.ndarray`, `metadata: dict[str, Any]`
- **Validation**: Image must be numpy array, metadata must be dictionary

### DetectionResultContract
Validates document detection results.
- **Fields**: `corners: np.ndarray | None`, `confidence: float | None`, `method: str | None`
- **Validation**: Corners must be valid numpy array if provided, confidence 0.0-1.0 if provided

## Validation Decorators

### @validate_image_input_with_fallback
- Validates image inputs using ImageInputContract
- Provides fallback processing for invalid inputs
- Returns standardized error response on validation failure

### @validate_preprocessing_result_with_fallback
- Validates preprocessing results using PreprocessingResultContract
- Ensures contract compliance with graceful error handling

## Contract Enforcement

### ContractEnforcer Class
Utility methods for enforcing contracts across components:
- `validate_image_input_contract()`: Validates image inputs
- `validate_preprocessing_result_contract()`: Validates results
- `validate_detection_result_contract()`: Validates detection results

## Testing
Contract compliance is tested in `tests/unit/test_preprocessing_contracts.py` with:
- Valid input validation tests
- Invalid input rejection tests
- Decorator fallback behavior tests
- Contract enforcer utility tests

## Benefits
- **Type Safety**: Eliminates runtime type errors from invalid inputs
- **Clear Contracts**: Explicit data expectations between components
- **Graceful Degradation**: Fallback handling for invalid inputs
- **Better Error Messages**: Descriptive validation errors
- **Maintainability**: Self-documenting interfaces
