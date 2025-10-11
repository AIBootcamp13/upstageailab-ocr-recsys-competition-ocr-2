# Refactor Plan: OCR Dataset Transforms (`ocr/datasets/transforms.py`)

## Table of Contents

1. [Objective and Why](#objective-and-why)
2. [Data Contracts with Pydantic v2](#data-contracts-with-pydantic-v2)
3. [Phase by Phase Plan](#phase-by-phase-plan)
4. [Delegated Development Work](#delegated-development-work)
5. [Tree Structure of Proposed Refactor](#tree-structure-of-proposed-refactor)

## Objective and Why

### Objective
Refactor `ocr/datasets/transforms.py` to establish robust data validation, eliminate manual validation code, standardize polygon handling, and improve maintainability. The refactored module will use Pydantic v2 for declarative data contracts, ensuring type safety and early error detection in the transform pipeline.

### Why
The current `transforms.py` suffers from:
- **Manual Validation Complexity**: Over 100 lines of brittle validation code that duplicates logic and is prone to errors.
- **Type Inconsistencies**: Defensive type checking (e.g., PIL to NumPy conversion) indicates unreliable input data.
- **Polygon Shape Confusion**: Multiple incompatible formats (`(N, 2)` vs `(1, N, 2)`) lead to transformation failures.
- **Tight Coupling**: Business logic is mixed with validation, making changes risky.
- **Debugging Challenges**: Scattered error handling masks root causes.

By adopting Pydantic v2 for data contracts, we create self-documenting, validated interfaces that prevent the "accidents" experienced during feature development. This aligns with the producer-consumer relationship where `transforms.py` is the consumer of data from `base.py`, requiring strict input guarantees.

## Data Contracts with Pydantic v2

Using Pydantic v2, we define clear schemas for input and output data. These replace manual validation and provide runtime type checking.

### Core Models

```python
from pydantic import BaseModel, field_validator, Field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
from pathlib import Path

class ImageMetadata(BaseModel):
    """Metadata for image processing."""
    filename: str
    path: Path
    original_shape: Tuple[int, int]  # (height, width) - NumPy convention
    orientation: int = Field(ge=1, le=8, default=1)
    is_normalized: bool = False
    dtype: str

class PolygonData(BaseModel):
    """Validated polygon with consistent shape."""
    points: np.ndarray
    confidence: Optional[float] = None
    label: Optional[str] = None

    @field_validator('points', mode='before')
    @classmethod
    def validate_points(cls, v):
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.float32)

        # Standardize to (N, 2) format
        if v.ndim == 1 and v.size % 2 == 0:
            v = v.reshape(-1, 2)
        elif v.ndim == 3 and v.shape[0] == 1:
            v = v.squeeze(0)

        if v.ndim != 2 or v.shape[1] != 2:
            raise ValueError(f"Polygon must be (N, 2) array, got shape {v.shape}")

        if v.shape[0] < 3:
            raise ValueError(f"Polygon must have at least 3 points, got {v.shape[0]}")

        return v.astype(np.float32)

class TransformInput(BaseModel):
    """Input data for transform pipeline."""
    image: np.ndarray
    polygons: Optional[List[PolygonData]] = None
    metadata: ImageMetadata

    @field_validator('image')
    @classmethod
    def validate_image(cls, v):
        if not isinstance(v, np.ndarray):
            raise TypeError(f"Image must be numpy array, got {type(v)}")

        if v.ndim not in (2, 3):
            raise ValueError(f"Image must be 2D or 3D, got {v.ndim}D")

        if v.ndim == 3 and v.shape[2] not in (1, 3):
            raise ValueError(f"Image must have 1 or 3 channels, got {v.shape[2]}")

        return v

class TransformOutput(BaseModel):
    """Validated output from transform pipeline."""
    image: torch.Tensor
    polygons: List[np.ndarray]  # Standardized to (1, N, 2) for compatibility
    inverse_matrix: np.ndarray
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('image')
    @classmethod
    def validate_output_image(cls, v):
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"Output image must be torch.Tensor, got {type(v)}")
        if v.ndim != 3:
            raise ValueError(f"Output image must be 3D (C, H, W), got shape {v.shape}")
        return v

    @field_validator('inverse_matrix')
    @classmethod
    def validate_matrix(cls, v):
        if v.shape != (3, 3):
            raise ValueError(f"Inverse matrix must be (3, 3), got {v.shape}")
        return v.astype(np.float32)

class TransformConfig(BaseModel):
    """Configuration for transforms."""
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    always_apply: bool = False
    p: float = 1.0

    @field_validator('mean', 'std')
    @classmethod
    def validate_mean_std(cls, v):
        if len(v) != 3:
            raise ValueError("Mean and std must have 3 values for RGB channels")
        return v
```

## Phase by Phase Plan

### Phase 1: Foundation (Week 1)
**Risk Level: Low** - Isolated schema creation with minimal dependencies.

**Success Criteria:**
- All Pydantic models defined and importable without errors
- Basic field validators pass unit tests
- Geometry utilities extracted and functional
- No breaking changes to existing code

### Phase 2: Core Refactor (Week 2)
**Risk Level: Medium** - Core logic changes with potential integration issues.

**Success Criteria:**
- ValidatedDBTransforms class replaces DBTransforms functionality
- All manual validation methods removed
- Polygon handling standardized across pipeline
- Existing transform tests pass with new implementation

### Phase 3: Integration and Testing (Week 3)
**Risk Level: High** - Full pipeline integration with potential performance impacts.

**Success Criteria:**
- End-to-end training pipeline works with refactored transforms
- Performance regression <5% compared to baseline
- All edge cases handled (empty polygons, various image formats)
- Comprehensive test coverage >90%

### Phase 4: Cleanup and Documentation (Week 4)
**Risk Level: Low** - Final polish with minimal risk.

**Success Criteria:**
- All deprecated code removed
- Documentation updated with new APIs
- Final integration tests pass
- Code review completed and approved

## Delegated Development Work

Using Qwen Coder CLI for parallel development. Each task includes a ready-to-use prompt for stdin execution.

### Task 1: Create Pydantic Schemas
**Prompt for Qwen Coder:**
```
echo "Create ocr/datasets/schemas.py with the following Pydantic v2 models: ImageMetadata, PolygonData, TransformInput, TransformOutput, TransformConfig. Include all field validators as specified. Ensure compatibility with numpy arrays and torch tensors." | qwen --prompt "Implement the Pydantic v2 data models for OCR dataset transforms. Use proper imports and handle arbitrary types for numpy/torch."
```

### Task 2: Extract Geometry Utilities
**Prompt for Qwen Coder:**
```
echo "Extract calculate_inverse_transform and calculate_cropbox methods from transforms.py into a new file ocr/utils/geometry_utils.py. Make them standalone functions with proper type hints." | qwen --prompt "Refactor geometry calculation methods from DBTransforms class into utility functions. Ensure they work independently and include docstrings."
```

### Task 3: Refactor DBTransforms Class
**Prompt for Qwen Coder:**
```
echo "Refactor the DBTransforms class in transforms.py to ValidatedDBTransforms. Replace manual validation with Pydantic models. Update __call__ method to accept TransformInput and return TransformOutput. Remove _validate_* methods." | qwen --prompt "Implement ValidatedDBTransforms class using Pydantic validation. Integrate with existing Albumentations pipeline while ensuring type safety."
```

### Task 4: Generate Unit Tests
**Prompt for Qwen Coder:**
```
echo "Generate comprehensive unit tests for the refactored transforms.py. Test Pydantic validation, polygon handling, and transform pipeline. Include edge cases and error conditions." | qwen --prompt "Create pytest unit tests for ValidatedDBTransforms and related schemas. Cover validation logic, polygon processing, and integration with Albumentations."
```

## Tree Structure of Proposed Refactor

```
ocr/
├── datasets/
│   ├── schemas.py              # NEW: Pydantic data models
│   └── transforms.py           # REFACTORED: ValidatedDBTransforms
└── utils/
    └── geometry_utils.py       # NEW: Extracted geometry functions
```</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/ai_handbook/07_planning/plans/refactor/refactor_preprocessing_transforms.md
