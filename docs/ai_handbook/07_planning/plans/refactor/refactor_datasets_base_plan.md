# Refactor Plan: OCR Dataset Base (`ocr/datasets/base.py`)

## Table of Contents

1. [Objective and Why](#objective-and-why)
2. [Data Contracts with Pydantic v2](#data-contracts-with-pydantic-v2)
3. [Phase by Phase Plan](#phase-by-phase-plan)
4. [Delegated Development Work](#delegated-development-work)
5. [Tree Structure of Proposed Refactor](#tree-structure-of-proposed-refactor)

## Objective and Why

### Objective
Refactor `ocr/datasets/base.py` to establish robust data validation, eliminate the "God Object" anti-pattern, and improve maintainability. The refactored module will use Pydantic v2 for declarative configuration and data contracts, ensuring type safety and early error detection in the data loading pipeline.

### Why
The current `base.py` suffers from:
- **God Object Complexity**: OCRDataset handles too many responsibilities (loading, caching, validation, EXIF processing).
- **Configuration Chaos**: Dozen of boolean flags in `__init__` make it error-prone.
- **Implicit Contracts**: OrderedDict output is loosely defined, leading to producer-consumer mismatches.
- **Mixed Concerns**: Caching, utilities, and business logic are tightly coupled.
- **Maintenance Burden**: Changes risk breaking the fragile handoff to transforms.

By adopting Pydantic v2 and separating concerns, we create a modular, testable, and predictable data loading system that prevents the "accidents" experienced during feature development.

## Data Contracts with Pydantic v2

Using Pydantic v2, we define clear schemas for configuration and output data.

### Core Models

```python
from pydantic import BaseModel, field_validator, Field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from pathlib import Path

class DatasetConfig(BaseModel):
    """Configuration for OCRDataset with validation."""
    image_path: str
    annotation_path: Optional[str] = None
    image_extensions: Optional[List[str]] = None
    preload_maps: bool = False
    load_maps: bool = False
    preload_images: bool = False
    prenormalize_images: bool = False
    cache_transformed_tensors: bool = False
    image_loading_config: Dict[str, Any] = {"use_turbojpeg": False, "turbojpeg_fallback": False}

    @field_validator('image_extensions', mode='before')
    @classmethod
    def validate_extensions(cls, v):
        if v is None:
            return [".jpg", ".jpeg", ".png"]
        return [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in v]

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

class DataItem(BaseModel):
    """Validated output from dataset before transforms."""
    image: np.ndarray
    image_filename: str
    image_path: str
    shape: Tuple[int, int]
    raw_size: Tuple[int, int]
    orientation: int
    polygon_frame: str
    polygons: Optional[List[np.ndarray]] = None
    prob_map: Optional[np.ndarray] = None
    thresh_map: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('image')
    @classmethod
    def validate_image(cls, v):
        if not isinstance(v, np.ndarray):
            raise TypeError(f"Image must be numpy array, got {type(v)}")
        if v.ndim not in (2, 3):
            raise ValueError(f"Image must be 2D or 3D, got {v.ndim}D")
        return v

class CacheManager(BaseModel):
    """Configuration for cache management."""
    enable_maps_cache: bool = False
    enable_images_cache: bool = False
    enable_tensors_cache: bool = False
    max_cache_size: int = 1000

    class Config:
        arbitrary_types_allowed = True
```

## Phase by Phase Plan

### Phase 1: Foundation (Week 1-2)
**Risk Level: Low** - Isolated schema and utility creation.

**Success Criteria:**
- All Pydantic models defined and functional
- Utility functions extracted to separate modules
- Basic configuration validation working
- No breaking changes to existing interfaces

### Phase 2: Core Refactor (Week 3-4)
**Risk Level: Medium** - Major class restructuring with dependency management.

**Success Criteria:**
- OCRDataset produces DataItem instances
- CacheManager extracted and functional
- Configuration decoupled from logic
- Existing dataset tests pass

### Phase 3: Integration and Optimization (Week 5-6)
**Risk Level: High** - Full pipeline integration with performance considerations.

**Success Criteria:**
- End-to-end data loading pipeline works
- Memory usage optimized
- Caching performance improved
- Comprehensive test coverage >90%

### Phase 4: Cleanup and Documentation (Week 7-8)
**Risk Level: Low** - Final refinements and documentation.

**Success Criteria:**
- Deprecated code removed
- API documentation updated
- Performance benchmarks established
- Code review completed

## Delegated Development Work

Using Qwen Coder CLI for parallel development. Each task includes a ready-to-use prompt for stdin execution.

### Task 1: Create Pydantic Schemas
**Prompt for Qwen Coder:**
```
echo "Create ocr/datasets/schemas.py with DatasetConfig, ImageMetadata, PolygonData, DataItem, and CacheManager Pydantic models. Include all field validators and proper imports." | qwen --prompt "Implement comprehensive Pydantic v2 schemas for OCR dataset components with validation logic for numpy arrays and file paths."
```

### Task 2: Extract Utility Functions
**Prompt for Qwen Coder:**
```
echo "Extract utility functions from base.py: _filter_degenerate_polygons, safe_get_image_size, rotate_image, _ensure_polygon_array, _clip_polygons_in_place. Move to ocr/utils/image_utils.py and ocr/utils/polygon_utils.py." | qwen --prompt "Refactor utility methods from OCRDataset into dedicated utility modules with proper type hints and documentation."
```

### Task 3: Implement CacheManager
**Prompt for Qwen Coder:**
```
echo "Create ocr/utils/cache_manager.py with CacheManager class. Extract caching logic from OCRDataset including maps, images, and tensors caching with statistics tracking." | qwen --prompt "Implement a dedicated CacheManager class for OCR dataset caching with configurable options and performance monitoring."
```

### Task 4: Refactor OCRDataset
**Prompt for Qwen Coder:**
```
echo "Refactor OCRDataset class to use DatasetConfig and produce DataItem instances. Integrate CacheManager and remove extracted utilities. Update __getitem__ to return validated DataItem." | qwen --prompt "Refactor OCRDataset to use Pydantic configuration and produce validated data items, separating concerns and improving maintainability."
```

### Task 5: Generate Unit Tests
**Prompt for Qwen Coder:**
```
echo "Generate comprehensive unit tests for refactored base.py components. Test Pydantic validation, caching logic, utility functions, and data loading pipeline." | qwen --prompt "Create pytest unit tests for all refactored components including schemas, utilities, cache manager, and dataset class with edge cases."
```

## Tree Structure of Proposed Refactor

```
ocr/
├── datasets/
│   ├── schemas.py              # NEW: Pydantic data models
│   └── base.py                 # REFACTORED: ValidatedOCRDataset
└── utils/
    ├── cache_manager.py        # NEW: CacheManager class
    ├── image_utils.py          # NEW: Image processing utilities
    └── polygon_utils.py        # NEW: Polygon processing utilities
```
