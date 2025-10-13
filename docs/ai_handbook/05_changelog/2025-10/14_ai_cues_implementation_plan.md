# AI Documentation Cues Implementation Plan

**Generated:** October 13, 2025
**Context:** OCR Dataset Modular Refactor - AI Assistant Guidance System
**Status:** Ready for Incremental Implementation

## Overview

This document provides comprehensive AI documentation cues for the remaining critical files in the OCR dataset system. These cues are designed to prevent AI assistants from making unintended modifications to core functionality.

**Completed Files:**
- ✅ `ocr/datasets/base.py` - Core dataset class with comprehensive AI cues
- ✅ `ocr/datasets/transforms.py` - Data transformation pipeline with detailed constraints

**Remaining Files to Process:**
1. `ocr/validation/models.py` - Pydantic data contracts
2. `ocr/utils/cache_manager.py` - Caching implementation
3. `ocr/utils/image_utils.py` - Image processing utilities
4. `ocr/utils/polygon_utils.py` - Polygon validation utilities
5. `ocr/lightning_modules/ocr_pl.py` - Training loop integration

## Implementation Strategy

### Phase 1: Utility Modules (High Priority)
- `ocr/utils/cache_manager.py`
- `ocr/utils/image_utils.py`
- `ocr/utils/polygon_utils.py`

### Phase 2: Data Contracts (Medium Priority)
- `ocr/validation/models.py`

### Phase 3: Training Integration (Lower Priority)
- `ocr/lightning_modules/ocr_pl.py`

---

# 1. ocr/utils/cache_manager.py

## File-Level Header

```python
"""
AI_DOCS: Cache Manager - Centralized Dataset Caching System

This module implements the CacheManager class, responsible for:
- Multi-level caching (images, tensors, maps) for dataset performance
- Cache statistics tracking and logging
- Memory-efficient storage with configurable limits
- Thread-safe cache operations for DataLoader compatibility

ARCHITECTURE OVERVIEW:
- Three cache types: image_cache, tensor_cache, maps_cache
- Configurable caching via CacheConfig (Pydantic model)
- Statistics tracking with periodic logging
- Lazy evaluation with conditional caching

DATA CONTRACTS:
- Input: CacheConfig (Pydantic model)
- Cache Keys: str (filenames) or int (dataset indices)
- Cache Values: ImageData, DataItem, MapData (Pydantic models)
- Statistics: hit/miss counts with configurable logging

CORE CONSTRAINTS:
- NEVER modify cache key formats (breaks cache invalidation)
- ALWAYS check cache config before operations
- PRESERVE statistics tracking for performance monitoring
- USE Pydantic models for all cached data
- MAINTAIN thread-safety for DataLoader compatibility

PERFORMANCE FEATURES:
- Lazy caching prevents memory bloat
- Configurable cache sizes and eviction policies
- Statistics logging for performance debugging
- Memory-efficient storage of large tensors

VALIDATION REQUIREMENTS:
- All cache values must be Pydantic models
- Cache keys must be hashable and deterministic
- Cache operations must handle missing keys gracefully
- Statistics must be accurate for performance analysis

RELATED DOCUMENTATION:
- Data Contracts: ocr/validation/models.py
- Configuration: ocr/datasets/schemas.py
- Base Dataset: ocr/datasets/base.py
- Performance Guide: docs/ai_handbook/04_performance/

MIGRATION NOTES:
- CacheManager replaces inline caching logic
- Pydantic models ensure data integrity
- Configurable caching improves memory management
"""
```

## Class-Level AI Cues

```python
class CacheManager:
    """
    AI_DOCS: CacheManager - Multi-Level Dataset Caching

    This class provides centralized caching for the OCR dataset system:
    - Image caching: Raw ImageData objects to avoid reloading
    - Tensor caching: Fully processed DataItem objects for speed
    - Maps caching: Probability/threshold maps for evaluation

    CONSTRAINTS FOR AI ASSISTANTS:
    - DO NOT modify cache data structures (dict types)
    - ALWAYS use config flags to enable/disable caching
    - PRESERVE statistics tracking methods
    - USE Pydantic models for cache values
    - MAINTAIN lazy evaluation pattern

    Cache Types:
    - image_cache: dict[str, ImageData] - keyed by filename
    - tensor_cache: dict[int, DataItem] - keyed by dataset index
    - maps_cache: dict[str, MapData] - keyed by filename
    """

    def __init__(self, config: CacheConfig) -> None:
        """
        AI_DOCS: Constructor Constraints
        - config: CacheConfig (Pydantic model) - NEVER pass raw dict
        - Initialize all cache dicts as empty
        - Setup statistics counters
        - DO NOT modify cache structure without updating all consumers
        """
```

## Critical Method Cues

```python
def get_cached_tensor(self, idx: int) -> DataItem | None:
    """
    AI_DOCS: Tensor Cache Retrieval
    Retrieves fully processed DataItem from cache by dataset index.

    CRITICAL CONSTRAINTS:
    - Return None if caching disabled (config.cache_transformed_tensors)
    - Record access statistics (hit/miss) for performance monitoring
    - Return DataItem Pydantic model (NOT dict)
    - Handle missing keys gracefully

    Performance Impact: Cache hits avoid expensive __getitem__ processing
    """

def set_cached_tensor(self, idx: int, data_item: DataItem) -> None:
    """
    AI_DOCS: Tensor Cache Storage
    Stores fully processed DataItem in cache by dataset index.

    CRITICAL CONSTRAINTS:
    - Only cache if config.cache_transformed_tensors is True
    - data_item MUST be DataItem Pydantic model
    - idx MUST be int (dataset index)
    - Overwrite existing entries without warning

    Memory Impact: Large tensors stored in memory for performance
    """
```

## End-of-File Summary

```python
# AI_DOCS: END OF FILE - CacheManager Constraints & Requirements
#
# =======================================================================
# CACHEMANAGER - AI ASSISTANT CONSTRAINTS & REQUIREMENTS
# =======================================================================
#
# 1. CACHE DATA STRUCTURES (DO NOT MODIFY):
#    - image_cache: dict[str, ImageData]
#    - tensor_cache: dict[int, DataItem]
#    - maps_cache: dict[str, MapData]
#
# 2. METHOD SIGNATURES (PRESERVE):
#    - get_cached_*() -> PydanticModel | None
#    - set_cached_*() -> None (takes Pydantic model)
#    - log_statistics() -> None
#    - reset_statistics() -> None
#
# 3. CONFIGURATION INTEGRATION:
#    - ALWAYS check config flags before caching
#    - RESPECT CacheConfig settings
#    - USE config.log_statistics_every_n for logging
#
# 4. STATISTICS TRACKING (MANDATORY):
#    - Record all cache accesses (hits/misses)
#    - Log statistics periodically
#    - Reset counters after logging
#    - Provide hit/miss count accessors
#
# 5. Pydantic MODEL REQUIREMENTS:
#    - All cache values MUST be Pydantic models
#    - ImageData for image cache
#    - DataItem for tensor cache
#    - MapData for maps cache
#
# =======================================================================
# COMMON AI MISTAKES TO AVOID:
# =======================================================================
#
# ❌ Changing cache dict structures or key types
# ❌ Skipping config flag checks
# ❌ Not recording cache statistics
# ❌ Using raw dicts instead of Pydantic models
# ❌ Modifying method signatures
# ❌ Breaking lazy evaluation pattern
#
# =======================================================================
```

---

# 2. ocr/utils/image_utils.py

## File-Level Header

```python
"""
AI_DOCS: Image Utils - Image Processing & Loading Utilities

This module provides specialized image processing utilities for the OCR dataset:
- EXIF-aware image loading with orientation correction
- RGB conversion and normalization
- Memory-efficient PIL image handling
- TurboJPEG integration for performance

ARCHITECTURE OVERVIEW:
- Utilities extracted from monolithic dataset code
- Focus on image loading and preprocessing
- Memory-safe PIL image lifecycle management
- Performance optimizations for large datasets

DATA CONTRACTS:
- Input: Path objects or strings (file paths)
- Output: PIL Images or numpy arrays
- Configuration: ImageLoadingConfig (Pydantic model)
- Metadata: ImageData with EXIF information

CORE CONSTRAINTS:
- ALWAYS close PIL images to prevent memory leaks
- PRESERVE EXIF orientation correction logic
- USE TurboJPEG when available for performance
- VALIDATE image loading before processing
- MAINTAIN backward compatibility with existing code

PERFORMANCE FEATURES:
- Lazy loading prevents memory bloat
- TurboJPEG acceleration for JPEG files
- EXIF orientation correction without full image rotation
- Memory-efficient image processing pipeline

VALIDATION REQUIREMENTS:
- Check file existence before loading
- Validate image formats and dimensions
- Handle corrupted image files gracefully
- Provide meaningful error messages

RELATED DOCUMENTATION:
- Base Dataset: ocr/datasets/base.py
- Configuration: ocr/datasets/schemas.py
- EXIF Handling: ocr/utils/orientation.py
- Performance Guide: docs/ai_handbook/04_performance/

MIGRATION NOTES:
- Utilities extracted from ValidatedOCRDataset._load_image_data
- Pydantic models ensure data integrity
- Memory management prevents leaks in long-running training
"""
```

## Critical Function Cues

```python
def load_pil_image(image_path: str | Path, config: ImageLoadingConfig | None = None) -> PILImage.Image:
    """
    AI_DOCS: PIL Image Loading with EXIF Support

    Loads PIL image with proper EXIF orientation handling.

    CRITICAL CONSTRAINTS:
    - ALWAYS apply EXIF orientation correction
    - USE TurboJPEG if available and enabled
    - VALIDATE file exists before loading
    - RETURN PIL Image (caller responsible for closing)
    - HANDLE OSError for corrupted files

    Memory Responsibility: Caller MUST close returned PIL image
    """

def ensure_rgb(image: PILImage.Image) -> PILImage.Image:
    """
    AI_DOCS: RGB Conversion Utility

    Converts any PIL image mode to RGB format.

    CRITICAL CONSTRAINTS:
    - ALWAYS convert to RGB (3 channels)
    - PRESERVE image dimensions
    - HANDLE all PIL image modes (L, P, RGBA, etc.)
    - RETURN new PIL image (original unchanged)

    Use Case: Ensures consistent 3-channel input for neural networks
    """

def pil_to_numpy(image: PILImage.Image) -> np.ndarray:
    """
    AI_DOCS: PIL to NumPy Conversion

    Converts PIL image to numpy array with proper dtype.

    CRITICAL CONSTRAINTS:
    - PRESERVE image dimensions and channels
    - USE uint8 dtype for original images
    - RETURN C-contiguous arrays for PyTorch compatibility
    - DO NOT modify original PIL image

    Output Format: (H, W, C) with dtype=uint8
    """

def prenormalize_imagenet(image_array: np.ndarray) -> np.ndarray:
    """
    AI_DOCS: ImageNet Normalization

    Applies ImageNet-style normalization to image array.

    CRITICAL CONSTRAINTS:
    - USE standard ImageNet mean/std values
    - CONVERT to float32 dtype
    - APPLY per-channel normalization
    - RETURN normalized array (0-1 range → normalized)

    Output: float32 array with ImageNet normalization applied
    """
```

---

# 3. ocr/utils/polygon_utils.py

## File-Level Header

```python
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
```

## Critical Function Cues

```python
def ensure_polygon_array(polygon: np.ndarray | list) -> np.ndarray | None:
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

def filter_degenerate_polygons(polygons: list[np.ndarray]) -> list[np.ndarray]:
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

def validate_map_shapes(
    prob_map: np.ndarray,
    thresh_map: np.ndarray,
    image_height: int | None = None,
    image_width: int | None = None,
    filename: str | None = None
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
```

---

# 4. ocr/validation/models.py

## File-Level Header

```python
"""
AI_DOCS: Validation Models - Pydantic Data Contracts for OCR Pipeline

This module defines all Pydantic v2 data validation models for the OCR system:
- Dataset input/output contracts with automatic validation
- Lightning module step contracts for training safety
- Evaluation metric configuration validation
- Comprehensive error messages and type checking

ARCHITECTURE OVERVIEW:
- Pydantic v2 BaseModel subclasses with validation
- Custom field validators for domain-specific constraints
- Automatic JSON serialization/deserialization
- Integration with PyTorch tensors and NumPy arrays

DATA CONTRACTS:
- Dataset: DatasetSample, TransformInput, TransformOutput, DataItem
- Lightning: ModelOutput, LightningStepPrediction, BatchSample, CollateOutput
- Evaluation: CLEvalMetric configuration models
- Utilities: PolygonArray, ImageMetadata, CacheConfig

CORE CONSTRAINTS:
- NEVER modify field names or types without migration plan
- ALWAYS use model_validate() for instantiation
- PRESERVE custom validators for data integrity
- USE model_dump() for serialization
- VALIDATE all inputs at system boundaries

VALIDATION FEATURES:
- Automatic type coercion and validation
- Custom field validators for complex constraints
- Detailed error messages with field context
- JSON schema generation for documentation
- Runtime performance with compiled validators

INTEGRATION REQUIREMENTS:
- Dataset classes use these models for data contracts
- Lightning modules validate step inputs/outputs
- Evaluation metrics receive validated configurations
- All data flow through validated contracts

RELATED DOCUMENTATION:
- Pydantic v2: https://docs.pydantic.dev/latest/
- Dataset Integration: ocr/datasets/base.py
- Lightning Integration: ocr/lightning_modules/ocr_pl.py
- Schema Definitions: ocr/datasets/schemas.py

MIGRATION NOTES:
- Pydantic v2 migration from dataclasses
- Enhanced validation prevents runtime errors
- Backward compatibility maintained through careful field design
"""
```

## Critical Model Cues

```python
class DataItem(BaseModel):
    """
    AI_DOCS: DataItem - Primary Dataset Output Contract

    The core data structure returned by dataset __getitem__ methods.

    CRITICAL CONSTRAINTS:
    - image: REQUIRED tensor field (transformed image)
    - polygons: REQUIRED list of polygon arrays
    - metadata: OPTIONAL dict with processing info
    - prob_map/thresh_map: OPTIONAL evaluation maps

    NEVER MODIFY:
    - Field names (breaks PyTorch DataLoader compatibility)
    - Field types (breaks downstream processing)
    - Required vs optional status

    Validation: Automatic shape and type checking
    """

class TransformInput(BaseModel):
    """
    AI_DOCS: TransformInput - Transformation Pipeline Input Contract

    Input contract for data transformation pipelines.

    CRITICAL CONSTRAINTS:
    - image: REQUIRED numpy array (H, W, C)
    - polygons: OPTIONAL list of PolygonData models
    - metadata: OPTIONAL ImageMetadata model

    Data Flow: Dataset → TransformInput → Transforms → TransformOutput
    """

class CollateOutput(BaseModel):
    """
    AI_DOCS: CollateOutput - DataLoader Collation Contract

    Output contract for DataLoader collation functions.

    CRITICAL CONSTRAINTS:
    - images: REQUIRED batched tensor
    - polygons: REQUIRED list of polygon batches
    - metadata: OPTIONAL batch metadata

    Lightning Integration: Used as input to training/validation steps
    """
```

---

# 5. ocr/lightning_modules/ocr_pl.py

## File-Level Header

```python
"""
AI_DOCS: OCR Lightning Module - PyTorch Lightning Training Integration

This module implements the OCRPLModule class, integrating OCR models with PyTorch Lightning:
- Training/validation/prediction step implementations
- Automatic optimization and logging
- Data contract validation at step boundaries
- Metrics calculation and tracking

ARCHITECTURE OVERVIEW:
- PyTorch Lightning Module subclass
- Composite model architecture with encoder/decoder/head/loss
- Data contract validation for training safety
- W&B integration for experiment tracking

DATA CONTRACTS:
- Input: CollateOutput (validated batch data)
- Model Output: ModelOutput (predictions with metadata)
- Metrics: CLEvalMetric results with configuration
- Logging: Structured logging with W&B integration

CORE CONSTRAINTS:
- NEVER modify step method signatures (breaks Lightning interface)
- ALWAYS validate inputs with Pydantic models
- PRESERVE logging structure for experiment tracking
- USE configured metrics for evaluation
- MAINTAIN backward compatibility with existing checkpoints

TRAINING FEATURES:
- Automatic mixed precision support
- Gradient accumulation and clipping
- Learning rate scheduling
- Early stopping and checkpointing

VALIDATION REQUIREMENTS:
- Validate CollateOutput inputs to training steps
- Ensure ModelOutput format consistency
- Check metric configuration validity
- Verify loss calculation correctness

RELATED DOCUMENTATION:
- PyTorch Lightning: https://lightning.ai/docs/
- Data Contracts: ocr/validation/models.py
- Model Architecture: ocr/models/
- Metrics: ocr/metrics/
- Configuration: configs/**/*.yaml

MIGRATION NOTES:
- Pydantic validation added for data integrity
- Enhanced error messages for debugging
- Backward compatibility maintained for existing experiments
"""
```

## Critical Method Cues

```python
def training_step(self, batch: CollateOutput, batch_idx: int) -> torch.Tensor:
    """
    AI_DOCS: Training Step - Core Training Logic

    Executes one training step with validated batch data.

    CRITICAL CONSTRAINTS:
    - batch: MUST be CollateOutput Pydantic model
    - RETURN: loss tensor for optimization
    - LOG: training metrics and losses
    - VALIDATE: input data contracts

    Data Flow: CollateOutput → Model → Loss → Optimization
    """

def validation_step(self, batch: CollateOutput, batch_idx: int) -> LightningStepPrediction:
    """
    AI_DOCS: Validation Step - Model Evaluation Logic

    Executes one validation step with validated batch data.

    CRITICAL CONSTRAINTS:
    - batch: MUST be CollateOutput Pydantic model
    - RETURN: LightningStepPrediction Pydantic model
    - LOG: validation metrics
    - COMPUTE: evaluation metrics using configured CLEvalMetric

    Data Flow: CollateOutput → Model → Predictions → Metrics
    """

def predict_step(self, batch: CollateOutput, batch_idx: int) -> ModelOutput:
    """
    AI_DOCS: Prediction Step - Inference Logic

    Executes one prediction step with validated batch data.

    CRITICAL CONSTRAINTS:
    - batch: MUST be CollateOutput Pydantic model
    - RETURN: ModelOutput Pydantic model
    - NO LOGGING: predictions are for external use
    - PRESERVE: polygon coordinate transformations

    Data Flow: CollateOutput → Model → Predictions → ModelOutput
    """
```

---

## Implementation Checklist

### Phase 1: Utility Modules
- [ ] `ocr/utils/cache_manager.py` - Add all AI cues
- [ ] `ocr/utils/image_utils.py` - Add all AI cues
- [ ] `ocr/utils/polygon_utils.py` - Add all AI cues

### Phase 2: Data Contracts
- [ ] `ocr/validation/models.py` - Add all AI cues

### Phase 3: Training Integration
- [ ] `ocr/lightning_modules/ocr_pl.py` - Add all AI cues

### Testing After Implementation
- [ ] Run existing tests to ensure no functionality changes
- [ ] Verify AI cues don't break imports or functionality
- [ ] Test that files can be imported and basic operations work

## AI Cue Patterns Used

### Documentation Levels
1. **File-Level**: Comprehensive overview and constraints
2. **Class-Level**: Architecture and interface constraints
3. **Method-Level**: Critical implementation requirements
4. **End-of-File**: Complete constraint reference

### Cue Types
- **AI_CRITICAL**: Must-follow constraints (red)
- **AI_CONTRACT**: Data contract requirements (orange)
- **AI_PERFORMANCE**: Performance considerations (yellow)
- **AI_REFERENCE**: Documentation pointers (blue)

### Common Patterns
- Step-by-step data flow documentation
- Before/after code examples
- Constraint rationales
- Migration and compatibility notes

---

## Next Steps

1. **Apply Phase 1** utility modules first (highest impact)
2. **Test thoroughly** after each file modification
3. **Apply Phase 2** data contracts
4. **Apply Phase 3** training integration
5. **Update this document** with any new patterns discovered

This systematic approach ensures comprehensive AI guidance while maintaining code quality and preventing unintended modifications.</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/ai_handbook/05_changelog/2025-10/14_ai_cues_implementation_plan.md
