# Procedural Refactor Blueprint: OCR Dataset Base

## Table of Contents
- [Overview](#overview)
- [Proposed Refactor Tree Structure](#proposed-refactor-tree-structure)
- [1. API Surface Definitions](#1-api-surface-definitions)
  - [`ocr.utils.cache_manager.CacheManager`](#ocrutilscache_managercachemanager)
  - [`ocr.datasets.base.ValidatedOCRDataset`](#ocrdatasetsbasevalidatedocrdataset)
- [2. Detailed Implementation Pseudocode](#2-detailed-implementation-pseudocode)
  - [`_load_annotations(self)` Helper Method](#_load_annotationsself-helper-method)
  - [`__init__(self, config: DatasetConfig, transform: Callable)` Method](#__init__self-config-datasetconfig-transform-callable-method)
  - [`__getitem__(self, idx: int)` Method](#__getitem__self-idx-int-method)
- [3. Data Flow Specification](#3-data-flow-specification)
- [4. Test Suite Generation Prompts](#4-test-suite-generation-prompts)
  - [CacheManager Tests](#cachemanager-tests)
  - [ValidatedOCRDataset Tests](#validatedocrdataset-tests)
  - [Integration Tests](#integration-tests)

## Overview
This blueprint provides a detailed, step-by-step implementation guide for refactoring the `ocr/datasets/base.py` file. The refactor introduces robust data validation using Pydantic v2, eliminates the "God Object" anti-pattern by separating concerns into dedicated modules, and improves maintainability. The refactored class will be named `ValidatedOCRDataset`.

## Proposed Refactor Tree Structure
```
ocr/
├── datasets/
│   ├── schemas.py          # NEW: Pydantic data models
│   └── base.py             # REFACTORED: ValidatedOCRDataset
└── utils/
    ├── cache_manager.py    # NEW: CacheManager class
    ├── image_utils.py      # NEW: Image processing utilities
    └── polygon_utils.py    # NEW: Polygon processing utilities
```

### Migration Outline (Rolling Plan)
1. [x] Introduce compatibility accessors on `ValidatedOCRDataset` (image/map flags, paths) so downstream code can transition without large rewrites.
2. [x] Update Hydra schemas (`configs/schemas/default_*.yaml` and related OmegaConf fixtures) to instantiate `DatasetConfig` + `ValidatedOCRDataset` instead of the legacy class.
3. [X] Migrate runtime scripts (preprocessing, benchmarking, agent tools) and callbacks to build `DatasetConfig`, import polygon helpers from `ocr.utils.polygon_utils`, and rely on the refactored dataset.
    - Completed: profiling utility (`scripts/analysis_validation/profile_data_loading.py`), contract validator (`scripts/analysis_validation/validate_pipeline_contracts.py`), and DB collate metadata ingestion.
    - Pending: preprocessing CLI, benchmarking/ablation scripts, Hydra runtime configs, Lightning callbacks.
4. [ ] Refactor unit/integration tests to use the new schemas and dataset constructor; retire asserts tied to the legacy init signature.
5. [ ] After consumers switch over, remove dead code paths (legacy helper statics, duplicate polygon/image utilities) and run the targeted pytest suite to confirm parity.

## 1. API Surface Definitions

### `ocr.utils.cache_manager.CacheManager`
```python
from typing import Optional, Dict, Any
from pathlib import Path
import logging

class CacheManager:
    """
    Manages caching of images, tensors, and maps for the OCR dataset.
    Provides centralized cache management with statistics tracking.
    """

    def __init__(self, config: 'CacheConfig') -> None:
        """
        Initialize the cache manager with configuration.

        Args:
            config: CacheConfig object containing cache settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.image_cache: Dict[str, 'ImageData'] = {}
        self.tensor_cache: Dict[int, 'DataItem'] = {}
        self.maps_cache: Dict[str, 'MapData'] = {}
        self._cache_hit_count: int = 0
        self._cache_miss_count: int = 0

    def get_cached_image(self, filename: str) -> Optional['ImageData']:
        """
        Retrieve cached image data by filename.

        Args:
            filename: Image filename

        Returns:
            ImageData if cached, None otherwise
        """
        if filename in self.image_cache:
            self._cache_hit_count += 1
            return self.image_cache[filename]
        self._cache_miss_count += 1
        return None

    def set_cached_image(self, filename: str, image_data: 'ImageData') -> None:
        """
        Cache image data by filename.

        Args:
            filename: Image filename
            image_data: ImageData object to cache
        """
        self.image_cache[filename] = image_data

    def get_cached_tensor(self, idx: int) -> Optional['DataItem']:
        """
        Retrieve cached tensor data by dataset index.

        Args:
            idx: Dataset index

        Returns:
            DataItem if cached, None otherwise
        """
        if idx in self.tensor_cache:
            self._cache_hit_count += 1
            return self.tensor_cache[idx]
        self._cache_miss_count += 1
        return None

    def set_cached_tensor(self, idx: int, data_item: 'DataItem') -> None:
        """
        Cache tensor data by dataset index.

        Args:
            idx: Dataset index
            data_item: DataItem object to cache
        """
        self.tensor_cache[idx] = data_item

    def get_cached_maps(self, filename: str) -> Optional['MapData']:
        """
        Retrieve cached map data by filename.

        Args:
            filename: Image filename

        Returns:
            MapData if cached, None otherwise
        """
        if filename in self.maps_cache:
            self._cache_hit_count += 1
            return self.maps_cache[filename]
        self._cache_miss_count += 1
        return None

    def set_cached_maps(self, filename: str, map_data: 'MapData') -> None:
        """
        Cache map data by filename.

        Args:
            filename: Image filename
            map_data: MapData object to cache
        """
        self.maps_cache[filename] = map_data

    def log_statistics(self) -> None:
        """
        Log cache statistics for monitoring and debugging.
        """
        total_accesses = self._cache_hit_count + self._cache_miss_count
        hit_rate = (self._cache_hit_count / total_accesses * 100) if total_accesses > 0 else 0

        self.logger.info(
            f"Cache Statistics - Hits: {self._cache_hit_count}, "
            f"Misses: {self._cache_miss_count}, "
            f"Hit Rate: {hit_rate:.1f}%, "
            f"Image Cache Size: {len(self.image_cache)}, "
            f"Tensor Cache Size: {len(self.tensor_cache)}, "
            f"Maps Cache Size: {len(self.maps_cache)}"
        )

        # Reset counters for next logging period
        self._cache_hit_count = 0
        self._cache_miss_count = 0

    def get_hit_count(self) -> int:
        """
        Get the current cache hit count.

        Returns:
            Current hit count
        """
        return self._cache_hit_count

    def get_miss_count(self) -> int:
        """
        Get the current cache miss count.

        Returns:
            Current miss count
        """
        return self._cache_miss_count
```

### `ocr.datasets.base.ValidatedOCRDataset`
```python
from typing import Callable, Dict, Any
from torch.utils.data import Dataset

class ValidatedOCRDataset(Dataset):
    """
    Refactored OCR dataset with Pydantic validation and separated concerns.
    """

    def __init__(self, config: 'DatasetConfig', transform: Callable[['TransformInput'], Dict[str, Any]]) -> None:
        """
        Initialize the validated OCR dataset.

        Args:
            config: DatasetConfig object containing all dataset configuration
            transform: Callable that takes TransformInput and returns transformed data dict
        """
        # Implementation details in pseudocode section

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            Number of samples
        """
        return len(self.anns)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing the processed sample data
        """
        # Implementation details in pseudocode section
```

## 2. Detailed Implementation Pseudocode

### `_load_annotations(self)` Helper Method

```python
def _load_annotations(self) -> None:
    """
    Load and parse annotations from the configured annotation file or image directory.
    Populates self.anns with filename-to-polygon mappings.
    """
    # Parse annotation_path to build self.anns dictionary
    if self.config.annotation_path is None:
        # No annotation file provided - load all valid images from image_path
        for file_path in self.config.image_path.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.config.image_extensions:
                self.anns[file_path.name] = None
    else:
        # Load and parse annotation file
        try:
            with open(self.config.annotation_path, 'r') as f:
                annotations = json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Annotation file not found: {self.config.annotation_path}")
            raise AnnotationFileError(f"Annotation file not found: {self.config.annotation_path}")
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in annotation file: {self.config.annotation_path}")
            raise AnnotationFileError(f"Invalid JSON in annotation file: {self.config.annotation_path}")
        except Exception as e:
            self.logger.error(f"Error loading annotation file {self.config.annotation_path}: {e}")
            raise AnnotationFileError(f"Error loading annotation file {self.config.annotation_path}: {e}")

        # Process each image in annotations
        for filename in annotations.get("images", {}).keys():
            image_file_path = self.config.image_path / filename
            if image_file_path.exists():
                # Check if image has word annotations
                image_annotations = annotations["images"][filename]
                if "words" in image_annotations:
                    # Extract polygon data from word annotations
                    gt_words = image_annotations["words"]
                    polygons = []
                    for word_data in gt_words.values():
                        if isinstance(word_data.get("points"), list) and len(word_data["points"]) > 0:
                            # Convert points to numpy array with rounded int32 values
                            polygon = np.array(np.round(word_data["points"]), dtype=np.int32)
                            polygons.append(polygon)
                    self.anns[filename] = polygons if polygons else None
                else:
                    self.anns[filename] = None
```

### `__init__(self, config: DatasetConfig, transform: Callable)` Method

```python
def __init__(self, config: DatasetConfig, transform: Callable) -> None:
    # Initialize basic attributes
    self.config = config
    self.transform = transform
    self.logger = logging.getLogger(__name__)
    self._canonical_frame_logged = set()

    # Initialize annotations dictionary
    self.anns = OrderedDict()

    # Load annotations using dedicated helper method
    self._load_annotations()

    # Instantiate CacheManager using configuration from config
    self.cache_manager = CacheManager(config.cache_config)

    # Dispatch to preloading methods based on config
    if config.preload_images:
        self._preload_images()

    if config.preload_maps:
        self._preload_maps()

    # Log initialization status
    if config.cache_config.cache_transformed_tensors:
        self.logger.info(f"Tensor caching enabled - will cache {len(self.anns)} transformed samples after first access")
```

### `__getitem__(self, idx: int)` Method

```python
def __getitem__(self, idx: int) -> Dict[str, Any]:
    # 1. Tensor Cache Check: Start by checking CacheManager for a fully processed DataItem
    cached_data_item = self.cache_manager.get_cached_tensor(idx)
    if cached_data_item is not None:
        # Log cache hits periodically for verification
        if idx % 50 == 0:
            self.logger.info(f"[CACHE HIT] Returning cached tensor for index {idx}")
        return cached_data_item.model_dump()

    # 2. Image Loading: If no tensor cached, get filename and check for cached image
    image_filename = list(self.anns.keys())[idx]
    image_path = self.config.image_path / image_filename

    cached_image_data = self.cache_manager.get_cached_image(image_filename)
    if cached_image_data is not None:
        # Use cached image data
        image_array = cached_image_data.image_array
        raw_width = cached_image_data.raw_width
        raw_height = cached_image_data.raw_height
        orientation = cached_image_data.orientation
        cache_source = "image_cache"
    else:
        # Load image from disk using image_utils
        try:
            pil_image = load_image_optimized(
                image_path,
                use_turbojpeg=self.config.image_loading_config.use_turbojpeg,
                turbojpeg_fallback=self.config.image_loading_config.turbojpeg_fallback
            )
        except OSError as e:
            raise RuntimeError(f"Failed to load image {image_filename}: {e}")

        # Handle EXIF orientation and convert to RGB numpy array
        raw_width, raw_height = safe_get_image_size(pil_image)
        try:
            normalized_image, orientation = normalize_pil_image(pil_image)
        except Exception as exc:
            pil_image.close()
            raise RuntimeError(f"Failed to normalize image {image_filename}: {exc}") from exc

        if normalized_image.mode != "RGB":
            rgb_image = normalized_image.convert("RGB")
        else:
            rgb_image = normalized_image.copy()

        image_array = np.array(rgb_image)

        # Clean up PIL objects
        rgb_image.close()
        if normalized_image is not pil_image:
            normalized_image.close()
        pil_image.close()

        # Cache the loaded image data
        image_data = ImageData(
            image_array=image_array,
            raw_width=raw_width,
            raw_height=raw_height,
            orientation=orientation,
            is_normalized=False  # Will be normalized during transformation if needed
        )
        self.cache_manager.set_cached_image(image_filename, image_data)
        cache_source = "disk"

    # 3. Annotation Processing: Load raw polygons and process them
    raw_polygons = self.anns[image_filename]
    processed_polygons = None
    polygon_frame = "raw"

    if raw_polygons is not None:
        # Convert raw polygons to list of numpy arrays
        polygons_list = [np.asarray(poly, dtype=np.float32) for poly in raw_polygons]

        # Use polygon_utils to handle orientation remapping
        if orientation_requires_rotation(orientation):
            if polygons_in_canonical_frame(polygons_list, raw_width, raw_height, orientation):
                processed_polygons = polygons_list
                polygon_frame = "canonical"
                if image_filename not in self._canonical_frame_logged:
                    self.logger.debug(
                        "Skipping EXIF remap for %s; polygons already align with canonical orientation (orientation=%d).",
                        image_filename,
                        orientation,
                    )
                    self._canonical_frame_logged.add(image_filename)
            else:
                processed_polygons = remap_polygons(polygons_list, raw_width, raw_height, orientation)
                polygon_frame = "canonical"
        else:
            processed_polygons = polygons_list

    # 4. Transformation: Assemble data into TransformInput Pydantic model and pass to transform
    height, width = image_array.shape[:2]

    metadata = ImageMetadata(
        filename=image_filename,
        path=image_path,
        original_shape=(height, width),
        orientation=orientation,
        is_normalized=image_array.dtype == np.float32,
        dtype=str(image_array.dtype),
        raw_size=(raw_width, raw_height),
        polygon_frame=polygon_frame,
        cache_source=cache_source,
        cache_hits=self.cache_manager.get_hit_count() if self.config.cache_config.cache_transformed_tensors else None,
        cache_misses=self.cache_manager.get_miss_count() if self.config.cache_config.cache_transformed_tensors else None,
    )

    polygon_models = None
    if processed_polygons is not None:
        polygon_models = []
        for poly in processed_polygons:
            try:
                polygon_models.append(PolygonData(points=poly))
            except ValidationError as exc:
                self.logger.warning("Dropping invalid polygon for %s: %s", image_filename, exc)

    transform_input = TransformInput(
        image=image_array,
        polygons=polygon_models,
        metadata=metadata
    )

    # Apply transformation
    transformed = self.transform(transform_input)

    # 5. Final Assembly & Validation: Construct DataItem from transformed outputs
    transformed_image = transformed["image"]
    transformed_polygons = transformed.get("polygons", []) or []

    # Filter degenerate polygons using polygon_utils
    if transformed_polygons:
        normalized_polygons = []
        for poly in transformed_polygons:
            poly_array = ensure_polygon_array(np.asarray(poly, dtype=np.float32))
            if poly_array is not None and poly_array.size > 0:
                normalized_polygons.append(poly_array)
        filtered_polygons = filter_degenerate_polygons(normalized_polygons)
    else:
        filtered_polygons = []

    # Load maps if enabled
    prob_map = None
    thresh_map = None
    if self.config.load_maps:
        cached_maps = self.cache_manager.get_cached_maps(image_filename)
        if cached_maps is not None:
            prob_map = cached_maps.prob_map
            thresh_map = cached_maps.thresh_map
        else:
            # Load from disk
            maps_dir = self.config.image_path.parent / f"{self.config.image_path.name}_maps"
            map_filename = maps_dir / f"{Path(image_filename).stem}.npz"

            if map_filename.exists():
                try:
                    maps_data = np.load(map_filename)
                    loaded_prob_map = maps_data["prob_map"]
                    loaded_thresh_map = maps_data["thresh_map"]

                    # Validate map shapes
                    if hasattr(transformed_image, "shape"):
                        image_height, image_width = transformed_image.shape[-2], transformed_image.shape[-1]
                    else:
                        image_height, image_width = None, None

                    if validate_map_shapes(loaded_prob_map, loaded_thresh_map, image_height, image_width, image_filename):
                        prob_map = loaded_prob_map
                        thresh_map = loaded_thresh_map

                        # Cache the maps
                        map_data = MapData(prob_map=prob_map, thresh_map=thresh_map)
                        self.cache_manager.set_cached_maps(image_filename, map_data)
                    else:
                        self.logger.warning(f"Skipping invalid maps for {image_filename}")
                except Exception as e:
                    self.logger.warning(f"Failed to load maps for {image_filename}: {e}")

    # Create final DataItem with validation
    data_item = DataItem(
        image=transformed_image,
        polygons=filtered_polygons,
        metadata=transformed.get("metadata"),
        prob_map=prob_map,
        thresh_map=thresh_map,
        inverse_matrix=transformed["inverse_matrix"]
    )

    # 6. Tensor Caching: Store validated DataItem in CacheManager
    if self.config.cache_config.cache_transformed_tensors:
        self.cache_manager.set_cached_tensor(idx, data_item)

    # 7. Return Value: Convert DataItem to dictionary
    return data_item.model_dump()
```

## 3. Data Flow Specification

### DatasetConfig Lifecycle
- **Creation**: Created by the user/client code with all dataset configuration parameters
- **Usage**: Passed as the first argument to `ValidatedOCRDataset.__init__()`
- **Storage**: Stored as `self.config` in the dataset instance
- **Access**: Used throughout the dataset methods to access configuration values (image paths, cache settings, etc.)

### TransformInput Lifecycle
- **Creation**: Created in `ValidatedOCRDataset.__getitem__()` after loading and processing image and polygon data
- **Contents**: Contains the image array, validated polygon models, and metadata
- **Usage**: Passed as the single argument to `self.transform` callable
- **Contract Enforcement**: Serves as the input data contract, ensuring the transform function receives properly validated and structured data

### DataItem Lifecycle
- **Creation**: Created in `ValidatedOCRDataset.__getitem__()` from the outputs of the transform function
- **Contents**: Contains the final processed image, filtered polygons, metadata, maps, and transformation matrices
- **Validation**: Acts as the final validation step, ensuring all output data conforms to the expected schema
- **Caching**: Stored in `CacheManager` for future retrieval, bypassing the entire processing pipeline
- **Return**: Converted to a dictionary via `.model_dump()` and returned to the data loader

### Data Flow Summary
1. User creates `DatasetConfig` → Dataset initialization
2. Dataset loads data → Creates `TransformInput` → Transform function
3. Transform outputs → Creates `DataItem` → Validation and caching
4. `DataItem` → Dictionary return

This flow ensures data contracts are enforced at every stage: input validation via `DatasetConfig`, transformation input via `TransformInput`, and output validation via `DataItem`.

## 4. Test Suite Generation Prompts

This section provides ready-to-use prompts for generating comprehensive test suites using Qwen Coder. Each prompt includes the necessary context and can be executed directly using the command structure from the Qwen Coder integration guide.

### CacheManager Tests

**Command:**
```bash
cat << 'EOF' | qwen --prompt "Generate a comprehensive pytest test suite for the CacheManager class. Include tests for all public methods, cache hit/miss scenarios, statistics logging, and edge cases. Use fixtures for common test data and ensure proper isolation between tests."
from typing import Optional, Dict, Any
from pathlib import Path
import logging

class CacheManager:
    """
    Manages caching of images, tensors, and maps for the OCR dataset.
    Provides centralized cache management with statistics tracking.
    """

    def __init__(self, config: 'CacheConfig') -> None:
        """
        Initialize the cache manager with configuration.

        Args:
            config: CacheConfig object containing cache settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.image_cache: Dict[str, 'ImageData'] = {}
        self.tensor_cache: Dict[int, 'DataItem'] = {}
        self.maps_cache: Dict[str, 'MapData'] = {}
        self._cache_hit_count: int = 0
        self._cache_miss_count: int = 0

    def get_cached_image(self, filename: str) -> Optional['ImageData']:
        """
        Retrieve cached image data by filename.

        Args:
            filename: Image filename

        Returns:
            ImageData if cached, None otherwise
        """
        if filename in self.image_cache:
            self._cache_hit_count += 1
            return self.image_cache[filename]
        self._cache_miss_count += 1
        return None

    def set_cached_image(self, filename: str, image_data: 'ImageData') -> None:
        """
        Cache image data by filename.

        Args:
            filename: Image filename
            image_data: ImageData object to cache
        """
        self.image_cache[filename] = image_data

    def get_cached_tensor(self, idx: int) -> Optional['DataItem']:
        """
        Retrieve cached tensor data by dataset index.

        Args:
            idx: Dataset index

        Returns:
            DataItem if cached, None otherwise
        """
        if idx in self.tensor_cache:
            self._cache_hit_count += 1
            return self.tensor_cache[idx]
        self._cache_miss_count += 1
        return None

    def set_cached_tensor(self, idx: int, data_item: 'DataItem') -> None:
        """
        Cache tensor data by dataset index.

        Args:
            idx: Dataset index
            data_item: DataItem object to cache
        """
        self.tensor_cache[idx] = data_item

    def get_cached_maps(self, filename: str) -> Optional['MapData']:
        """
        Retrieve cached map data by filename.

        Args:
            filename: Image filename

        Returns:
            MapData if cached, None otherwise
        """
        if filename in self.maps_cache:
            self._cache_hit_count += 1
            return self.maps_cache[filename]
        self._cache_miss_count += 1
        return None

    def set_cached_maps(self, filename: str, map_data: 'MapData') -> None:
        """
        Cache map data by filename.

        Args:
            filename: Image filename
            map_data: MapData object to cache
        """
        self.maps_cache[filename] = map_data

    def log_statistics(self) -> None:
        """
        Log cache statistics for monitoring and debugging.
        """
        total_accesses = self._cache_hit_count + self._cache_miss_count
        hit_rate = (self._cache_hit_count / total_accesses * 100) if total_accesses > 0 else 0

        self.logger.info(
            f"Cache Statistics - Hits: {self._cache_hit_count}, "
            f"Misses: {self._cache_miss_count}, "
            f"Hit Rate: {hit_rate:.1f}%, "
            f"Image Cache Size: {len(self.image_cache)}, "
            f"Tensor Cache Size: {len(self.tensor_cache)}, "
            f"Maps Cache Size: {len(self.maps_cache)}"
        )

        # Reset counters for next logging period
        self._cache_hit_count = 0
        self._cache_miss_count = 0

    def get_hit_count(self) -> int:
        """
        Get the current cache hit count.

        Returns:
            Current hit count
        """
        return self._cache_hit_count

    def get_miss_count(self) -> int:
        """
        Get the current cache miss count.

        Returns:
            Current miss count
        """
        return self._cache_miss_count
EOF
```

### ValidatedOCRDataset Tests

**Command:**
```bash
cat << 'EOF' | qwen --prompt "Generate a comprehensive pytest test suite for the ValidatedOCRDataset class. Include tests for initialization with different configurations, __len__ and __getitem__ methods, annotation loading, caching behavior, and error handling. Use mocks for external dependencies and fixtures for test data."
from typing import Callable, Dict, Any
from torch.utils.data import Dataset

class ValidatedOCRDataset(Dataset):
    """
    Refactored OCR dataset with Pydantic validation and separated concerns.
    """

    def __init__(self, config: 'DatasetConfig', transform: Callable[['TransformInput'], Dict[str, Any]]) -> None:
        """
        Initialize the validated OCR dataset.

        Args:
            config: DatasetConfig object containing all dataset configuration
            transform: Callable that takes TransformInput and returns transformed data dict
        """
        # Implementation details in pseudocode section

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            Number of samples
        """
        return len(self.anns)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing the processed sample data
        """
        # Implementation details in pseudocode section
EOF
```

### Integration Tests

**Command:**
```bash
cat << 'EOF' | qwen --prompt "Generate integration tests that verify the interaction between ValidatedOCRDataset, CacheManager, and the transform pipeline. Include tests for end-to-end data flow, caching effectiveness, performance benchmarks, and memory usage validation. Use realistic test data and measure actual performance improvements."
# Integration test context for the refactored OCR dataset components
# This should test the complete data pipeline from DatasetConfig to final DataItem
# Include performance benchmarks comparing cached vs non-cached access
# Test memory usage patterns and cache effectiveness
# Verify data validation at each stage of the pipeline
EOF
```
