# Advanced Preprocessing Data Contracts & Pydantic Standards

## Overview

This reference document outlines the data contracts and Pydantic validation standards for the advanced document detection and preprocessing enhancement work. All new code must follow these established patterns for consistency and type safety.

## Reference Documentation

**Primary Reference**: `docs/pipeline/preprocessing-data-contracts.md`
- Contains complete contract specifications
- Includes validation decorator usage
- Documents contract enforcement utilities

## Pydantic Model Standards

### Required Patterns for New Models

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import numpy as np

class AdvancedPreprocessingConfig(BaseModel):
    """Configuration model for advanced preprocessing components.

    All configuration classes must inherit from BaseModel and include:
    - Field descriptions using Field()
    - Input validation using @validator
    - Proper type hints
    """

    # Required fields with validation
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for processing decisions"
    )

    # Optional fields
    max_iterations: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000,
        description="Maximum processing iterations (None for unlimited)"
    )

    # Complex types with validation
    processing_steps: List[str] = Field(
        default_factory=list,
        description="Ordered list of processing steps to apply"
    )

    @validator('processing_steps')
    def validate_processing_steps(cls, v):
        """Validate processing steps are known operations."""
        valid_steps = {
            'corner_detection', 'geometric_modeling', 'noise_elimination',
            'document_flattening', 'brightness_adjustment'
        }
        invalid_steps = set(v) - valid_steps
        if invalid_steps:
            raise ValueError(f"Unknown processing steps: {invalid_steps}")
        return v

    class Config:
        """Pydantic configuration for advanced types."""
        arbitrary_types_allowed = True  # Allow numpy arrays
        validate_assignment = True      # Validate on attribute assignment
```

### Data Contract Compliance

#### Input Contracts
```python
class ImageInputContract(BaseModel):
    """Standard contract for image inputs."""
    image: np.ndarray
    metadata: Optional[dict] = None

    @validator('image')
    def validate_image(cls, v):
        """Validate numpy array properties."""
        if not isinstance(v, np.ndarray):
            raise ValueError("Input must be numpy array")
        if v.ndim not in [2, 3]:
            raise ValueError("Image must be 2D or 3D array")
        if v.size == 0:
            raise ValueError("Image cannot be empty")
        return v
```

#### Output Contracts
```python
class ProcessingResultContract(BaseModel):
    """Standard contract for processing results."""
    image: np.ndarray
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)
    processing_time: Optional[float] = None
    method: str

    @validator('metadata')
    def validate_metadata(cls, v):
        """Ensure metadata is properly structured."""
        if not isinstance(v, dict):
            raise ValueError("Metadata must be dictionary")
        return v
```

## Validation Decorator Usage

### Public API Methods
```python
from pydantic import validate_call
from typing import Union

class AdvancedDocumentProcessor:
    """Example processor following contract standards."""

    @validate_call
    def detect_document_corners(
        self,
        image: np.ndarray,
        config: AdvancedPreprocessingConfig
    ) -> Union[DetectedCorners, ErrorResponse]:
        """Detect document corners with full validation.

        Args:
            image: Input image as numpy array
            config: Processing configuration

        Returns:
            DetectedCorners on success, ErrorResponse on failure
        """
        try:
            # Contract enforcement
            ContractEnforcer.validate_image_input_contract(image)

            # Processing logic
            corners = self._detect_corners(image, config)

            # Result validation
            result = DetectedCorners(
                corners=corners,
                confidence=0.95,
                method="advanced_harris"
            )

            ContractEnforcer.validate_detection_result_contract(result)
            return result

        except ValidationError as e:
            return ErrorResponse(
                error_code="VALIDATION_ERROR",
                message=f"Input validation failed: {e}",
                details=e.errors()
            )
        except Exception as e:
            return ErrorResponse(
                error_code="PROCESSING_ERROR",
                message=f"Corner detection failed: {str(e)}",
                details={"exception_type": type(e).__name__}
            )
```

## Contract Enforcement Utilities

### Using ContractEnforcer
```python
from ocr.datasets.preprocessing.contracts import ContractEnforcer

class ProcessingPipeline:
    """Pipeline that enforces contracts at each step."""

    def process(self, image: np.ndarray) -> ProcessingResult:
        """Process image with contract enforcement."""

        # Input validation
        ContractEnforcer.validate_image_input_contract(image)

        # Processing steps with validation
        corners = self.detect_corners(image)
        ContractEnforcer.validate_detection_result_contract(corners)

        geometry = self.fit_geometry(corners)
        ContractEnforcer.validate_geometry_contract(geometry)

        result = self.apply_enhancements(geometry)
        ContractEnforcer.validate_preprocessing_result_contract(result)

        return result
```

## Error Handling Standards

### Standardized Error Responses
```python
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    """Standard error response format."""
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(default=None, description="Additional error context")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True
```

### Error Code Standards
- `VALIDATION_ERROR`: Input/output validation failures
- `PROCESSING_ERROR`: Algorithm execution failures
- `CONFIGURATION_ERROR`: Invalid configuration parameters
- `RESOURCE_ERROR`: Memory/disk/network issues

## Testing Requirements

### Contract Compliance Tests
```python
import pytest
from pydantic import ValidationError

class TestDataContracts:
    """Test contract compliance for new components."""

    def test_image_input_contract_valid(self):
        """Test valid image input passes contract."""
        image = np.random.rand(100, 100, 3).astype(np.uint8)
        contract = ImageInputContract(image=image)
        assert contract.image.shape == (100, 100, 3)

    def test_image_input_contract_invalid(self):
        """Test invalid image input fails contract."""
        with pytest.raises(ValidationError):
            ImageInputContract(image="not_an_array")

    def test_processing_result_contract(self):
        """Test processing result validation."""
        result = ProcessingResultContract(
            image=np.zeros((50, 50), dtype=np.uint8),
            confidence=0.85,
            method="test_method"
        )
        assert result.confidence == 0.85
```

## Migration Guide

### Converting from Dataclasses to Pydantic

**Before (dataclass):**
```python
@dataclass
class OldConfig:
    threshold: float = 0.5
    method: str = "default"
```

**After (Pydantic):**
```python
class NewConfig(BaseModel):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    method: str = Field(default="default", regex=r"^(default|advanced|simple)$")

    @validator('threshold')
    def validate_threshold(cls, v):
        if v <= 0:
            raise ValueError("Threshold must be positive")
        return v
```

## Benefits

- **Runtime Safety**: Catch data validation errors before processing
- **API Clarity**: Self-documenting interfaces with clear expectations
- **Consistency**: Uniform patterns across all preprocessing components
- **Maintainability**: Easier refactoring with validated contracts
- **Testing**: Contract tests ensure component reliability
- **Error Handling**: Standardized error responses improve debugging

## Related Documents

- `docs/pipeline/preprocessing-data-contracts.md` - Complete contract reference
- `docs/ai_handbook/07_planning/plans/pydantic-data-validation/` - Original refactor documentation
- `tests/unit/test_preprocessing_contracts.py` - Contract compliance tests</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/ai_handbook/03_references/preprocessing/advanced-preprocessing-data-contracts.md
