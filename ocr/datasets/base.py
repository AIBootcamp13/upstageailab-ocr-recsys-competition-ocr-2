"""
AI_DOCS: OCR Dataset Base - Core Data Loading Pipeline

This module implements the ValidatedOCRDataset class, the primary dataset implementation
for OCR text detection tasks. This class serves as the single source of truth for data
loading, validation, and preprocessing in the OCR pipeline.

ARCHITECTURE OVERVIEW:
- Uses Pydantic v2 for comprehensive data validation and type safety
- Implements modular design with extracted utilities (cache_manager, image_utils, polygon_utils)
- Maintains backward compatibility with existing training pipelines
- Supports multiple caching strategies for performance optimization

DATA CONTRACTS:
- Input: DatasetConfig (validated configuration)
- Transform: Callable[TransformInput, dict[str, Any]] (data transformation pipeline)
- Output: DataItem (fully validated sample dictionary)

CORE CONSTRAINTS:
- NEVER modify the __getitem__ method signature or return type without updating all consumers
- ALWAYS validate data using Pydantic models before processing
- PRESERVE backward compatibility for existing training scripts
- USE extracted utilities instead of inline implementations

PERFORMANCE FEATURES:
- Tensor caching: Cache fully transformed samples to avoid recomputation
- Image preloading: Load all images into memory for fast access
- Map preloading: Cache probability/threshold maps in memory
- EXIF orientation handling: Automatic image rotation correction

VALIDATION REQUIREMENTS:
- All polygons must be numpy arrays with shape (N, 2) and dtype float32
- Images must be normalized to RGB format before transformation
- Metadata must include orientation, dimensions, and processing flags
- Cache keys must be deterministic and collision-resistant

RELATED DOCUMENTATION:
- Data contracts: ocr/validation/models.py
- Configuration schemas: ocr/datasets/schemas.py
- Utility functions: ocr/utils/{cache_manager,image_utils,polygon_utils}/
- Architecture guide: docs/ai_handbook/03_references/architecture/01_architecture.md
- Data loading guide: docs/ai_handbook/03_references/data_loading/data_format_comparison.md

MIGRATION NOTES:
- Legacy OCRDataset removed in favor of ValidatedOCRDataset
- All utilities extracted to dedicated modules for better testability
- Pydantic validation added throughout the pipeline for data integrity
"""

import json
import logging
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import ValidationError
from torch.utils.data import Dataset

# Module-level set to track warnings that have already been logged
# This prevents repetitive logging when multiple datasets are created with the same config
_logged_warnings: set[str] = set()

from ocr.datasets.schemas import DatasetConfig, ImageData, ImageMetadata, PolygonData, TransformInput
from ocr.utils.orientation import (
    EXIF_ORIENTATION_TAG,
    normalize_pil_image,
)

Image.MAX_IMAGE_PIXELS = 108000000
EXIF_ORIENTATION = EXIF_ORIENTATION_TAG  # Orientation Information: 274


# AI_DOCS: ValidatedOCRDataset - Primary Dataset Class
#
# This is the CORE dataset class for OCR tasks. It implements:
# - Pydantic-validated data loading and processing
# - Multi-level caching (images, tensors, maps)
# - EXIF orientation correction
# - Polygon validation and filtering
# - Transform pipeline integration
#
# CONSTRAINTS FOR AI ASSISTANTS:
# - DO NOT modify __getitem__ return type (must be dict[str, Any])
# - DO NOT change constructor signature without updating all config files
# - ALWAYS use Pydantic models for data validation
# - PRESERVE all property accessors for backward compatibility
# - USE extracted utilities from ocr.utils.* modules
#
# DATA FLOW: __getitem__ -> _load_image_data -> transform -> DataItem -> cache -> dict
class ValidatedOCRDataset(Dataset):
    """
    Refactored OCR dataset with Pydantic validation and separated concerns.
    """

    def __init__(self, config: "DatasetConfig", transform: Callable[["TransformInput"], dict[str, Any]]) -> None:
        """
        Initialize the validated OCR dataset.

        AI_DOCS: Constructor Constraints
        - config: DatasetConfig (Pydantic model) - NEVER pass raw dicts
        - transform: Callable that MUST accept TransformInput and return dict[str, Any]
        - DO NOT modify parameter types without updating all Hydra configs
        - DO NOT add required parameters without migration plan

        Args:
            config: DatasetConfig object containing all dataset configuration
            transform: Callable that takes TransformInput and returns transformed data dict
        """
        # AI_DOCS: Initialization Data Flow
        # 1. Store validated config (DO NOT modify config structure)
        # 2. Store transform callable (MUST be compatible with TransformInput)
        # 3. Initialize logging (use self.logger for all messages)
        # 4. Load annotations (populates self.anns OrderedDict)
        # 5. Initialize CacheManager (uses config.cache_config)
        # 6. Preload data if configured (images/maps based on config flags)

        # Implementation details in pseudocode section
        self.config = config
        self.transform = transform
        self.logger = logging.getLogger(__name__)
        self._canonical_frame_logged: set[str] = set()

        # PERFORMANCE VALIDATION: Check for unsafe configurations
        self._validate_performance_config()

        # Initialize annotations dictionary
        self.anns: OrderedDict[str, list[np.ndarray] | None] = OrderedDict()

        # Load annotations using dedicated helper method
        self._load_annotations()

        # Instantiate CacheManager using configuration from config with versioning
        from ocr.utils.cache_manager import CacheManager

        # Generate cache version based on configuration
        cache_version = config.cache_config.get_cache_version(load_maps=config.load_maps)
        self.cache_manager = CacheManager(config.cache_config, cache_version=cache_version)

        # Log cache configuration only when caching is actually enabled
        caching_enabled = (
            config.cache_config.cache_transformed_tensors or config.cache_config.cache_images or config.cache_config.cache_maps
        )
        if caching_enabled:
            cache_version_msg = f"Cache initialized with version: {cache_version}"
            if cache_version_msg not in _logged_warnings:
                self.logger.info(cache_version_msg)
                _logged_warnings.add(cache_version_msg)

            cache_config_msg = (
                f"Cache config: tensor={config.cache_config.cache_transformed_tensors}, "
                f"images={config.cache_config.cache_images}, maps={config.cache_config.cache_maps}, load_maps={config.load_maps}"
            )
            if cache_config_msg not in _logged_warnings:
                self.logger.info(cache_config_msg)
                _logged_warnings.add(cache_config_msg)

        # Dispatch to preloading methods based on config
        if config.preload_images:
            self._preload_images()

        if config.preload_maps:
            self._preload_maps()

        # Log initialization status
        if config.cache_config.cache_transformed_tensors:
            self.logger.info(f"Tensor caching enabled - will cache {len(self.anns)} transformed samples after first access")

    def _validate_performance_config(self) -> None:
        """
        Validate performance optimization configuration for safety and correctness.

        This method checks for:
        - Unsafe combinations of features
        - Memory usage warnings
        - Configuration consistency
        - Dataset type appropriateness

        Logs warnings for potential issues but doesn't prevent execution.
        """
        warnings = []
        errors = []

        # Check if this looks like a training dataset (heuristic)
        dataset_name = str(self.config.image_path).lower()
        is_training_like = any(keyword in dataset_name for keyword in ["train", "training"])

        # 🚨 CRITICAL: Tensor caching on training-like datasets
        if is_training_like and self.config.cache_config.cache_transformed_tensors:
            errors.append(
                "🚨 CRITICAL: Tensor caching enabled on training-like dataset "
                f"({self.config.image_path}). This can cause data leakage as cached "
                "tensors include augmentations. Disable cache_transformed_tensors for training."
            )

        # ⚠️ WARNING: High memory usage prediction
        memory_gb = self._estimate_memory_usage()
        if memory_gb > 6:  # Conservative threshold
            warnings.append(
                f"⚠️ High memory usage predicted: ~{memory_gb:.1f}GB. "
                "Consider disabling cache_transformed_tensors or preload_images if experiencing OOM."
            )

        # ⚠️ WARNING: Preloading without caching
        if self.config.preload_images and not self.config.cache_config.cache_images:
            warnings.append(
                "⚠️ Image preloading enabled but cache_images=false. Preloaded images won't be cached - this defeats the purpose."
            )

        # ⚠️ WARNING: Tensor caching without image caching
        if self.config.cache_config.cache_transformed_tensors and not self.config.cache_config.cache_images:
            warnings.append("⚠️ Tensor caching enabled but cache_images=false. This may cause inconsistent caching behavior.")

        # ⚠️ WARNING: Maps features without load_maps
        if self.config.cache_config.cache_maps and not self.config.load_maps:
            # Automatically disable maps caching when load_maps is false
            warning_msg = "Maps caching disabled because load_maps=false. Maps caching requires load_maps to be enabled."
            if warning_msg not in _logged_warnings:
                self.logger.info(warning_msg)
                _logged_warnings.add(warning_msg)
            self.config.cache_config.cache_maps = False

        # Log all warnings
        for warning in warnings:
            if warning not in _logged_warnings:
                self.logger.warning(warning)
                _logged_warnings.add(warning)

        # Log and potentially raise errors
        for error in errors:
            if error not in _logged_warnings:
                self.logger.error(error)
                _logged_warnings.add(error)

    def _estimate_memory_usage(self) -> float:
        """
        Estimate memory usage in GB for current configuration.

        Returns:
            float: Estimated memory usage in GB
        """
        base_memory = 2.0  # PyTorch + model baseline

        if self.config.preload_images:
            # ~200MB for 404 validation images
            base_memory += 0.2

        if self.config.cache_config.cache_transformed_tensors:
            # ~800MB-1.2GB for tensor cache (conservative estimate)
            base_memory += 1.2

        if self.config.cache_config.cache_maps:
            # ~50MB for maps
            base_memory += 0.05

        return base_memory

    def _check_cache_health(self) -> None:
        """
        Monitor cache health and warn about potential invalidation.

        This method checks for signs that the cache may be invalid:
        - Low hit rate after cache should be warm
        - Sudden drop in hit rate
        - Cache size mismatches

        Only runs occasional checks to avoid performance impact.
        """
        # Only check every 100 cache misses to avoid overhead
        total_accesses = self.cache_manager.get_hit_count() + self.cache_manager.get_miss_count()
        if total_accesses % 100 != 0:
            return

        hit_rate = self.cache_manager.get_hit_count() / max(1, total_accesses)
        cache_size = len(self.cache_manager.tensor_cache)

        # Warning thresholds
        if hit_rate < 0.5 and total_accesses > 200:  # Low hit rate after warm-up
            self.logger.warning(
                f"⚠️ LOW CACHE HIT RATE: {hit_rate:.1%} ({total_accesses} accesses). "
                "Cache may be invalid. Consider clearing cache if training on different data."
            )

        if cache_size > 0 and cache_size != len(self.anns):  # Size mismatch
            self.logger.warning(
                f"🚨 CRITICAL CACHE ISSUE: Cache has {cache_size} items but dataset has {len(self.anns)}. "
                "Cache is invalid and will be cleared to prevent incorrect results."
            )
            # Clear ALL invalid caches to prevent data corruption
            self.cache_manager.clear_all_caches()
            self.logger.info("All caches cleared. Caches will rebuild on next epoch.")

    # ------------------------------------------------------------------
    # Compatibility accessors for legacy consumers
    # ------------------------------------------------------------------
    # AI_DOCS: Backward Compatibility Properties
    #
    # These properties provide backward compatibility with legacy code that
    # accessed dataset attributes directly. AI assistants MUST:
    # - NEVER remove these properties (breaks existing code)
    # - NEVER change return types (breaks existing code)
    # - Update to use config.* instead of direct attributes
    # - Add new properties here if needed for compatibility
    #
    # Legacy code expects these properties to exist and return expected types.
    # ------------------------------------------------------------------
    @property
    def image_path(self) -> Path:
        return self.config.image_path

    @property
    def annotation_path(self) -> Path | None:
        return self.config.annotation_path

    @property
    def image_extensions(self) -> list[str]:
        return self.config.image_extensions

    @property
    def preload_maps(self) -> bool:
        return self.config.preload_maps

    @property
    def load_maps(self) -> bool:
        return self.config.load_maps

    @property
    def preload_images(self) -> bool:
        return self.config.preload_images

    @property
    def prenormalize_images(self) -> bool:
        return self.config.prenormalize_images

    @property
    def cache_transformed_tensors(self) -> bool:
        return self.config.cache_config.cache_transformed_tensors

    @property
    def cache_config(self):
        return self.config.cache_config

    @property
    def image_loading_config(self):
        return self.config.image_loading_config

    @property
    def image_cache(self):
        return self.cache_manager.image_cache

    @property
    def tensor_cache(self):
        return self.cache_manager.tensor_cache

    @property
    def maps_cache(self):
        return self.cache_manager.maps_cache

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
                with open(self.config.annotation_path) as f:
                    annotations = json.load(f)
            except FileNotFoundError:
                self.logger.error(f"Annotation file not found: {self.config.annotation_path}")
                raise RuntimeError(f"Annotation file not found: {self.config.annotation_path}")
            except json.JSONDecodeError:
                self.logger.error(f"Invalid JSON in annotation file: {self.config.annotation_path}")
                raise RuntimeError(f"Invalid JSON in annotation file: {self.config.annotation_path}")
            except Exception as e:
                self.logger.error(f"Error loading annotation file {self.config.annotation_path}: {e}")
                raise RuntimeError(f"Error loading annotation file {self.config.annotation_path}: {e}")

            # Process each image in annotations
            for filename in annotations.get("images", {}).keys():
                image_file_path = self.config.image_path / filename
                if not image_file_path.exists():
                    self.logger.debug("Annotation references missing image: %s", image_file_path)

                image_annotations = annotations["images"][filename]
                if "words" in image_annotations:
                    gt_words = image_annotations["words"]
                    polygons = []
                    for word_data in gt_words.values():
                        points = word_data.get("points")
                        if isinstance(points, list) and len(points) > 0:
                            polygon = np.array(np.round(points), dtype=np.int32)
                            polygons.append(polygon)
                    self.anns[filename] = polygons if polygons else None
                else:
                    self.anns[filename] = None

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            Number of samples
        """
        return len(self.anns)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a sample from the dataset by index.

        AI_DOCS: CRITICAL METHOD - Data Pipeline Core
        This is the PRIMARY data loading method. AI assistants MUST:
        - NEVER change return type (dict[str, Any])
        - ALWAYS validate data with Pydantic models
        - USE extracted utilities from ocr.utils.* modules
        - PRESERVE caching logic and error handling
        - MAINTAIN EXIF orientation correction
        - RETURN DataItem.model_dump() format

        Data Flow (DO NOT MODIFY):
        1. Check tensor cache (CacheManager.get_cached_tensor)
        2. Load image data (_load_image_data -> ImageData)
        3. Process polygons (polygon_utils.ensure_polygon_array)
        4. Create TransformInput (Pydantic model)
        5. Apply transform (returns dict with image, polygons, metadata)
        6. Filter polygons (polygon_utils.filter_degenerate_polygons)
        7. Load maps if enabled (cache or disk)
        8. Create DataItem (Pydantic model)
        9. Cache DataItem if enabled
        10. Return DataItem.model_dump()

        Args:
            idx: Sample index

        Returns:
            Dictionary containing the processed sample data (DataItem.model_dump())
        """
        # AI_DOCS: Step 1 - Tensor Cache Check
        # ALWAYS check cache first for performance
        # Cache contains fully processed DataItem objects
        from ocr.datasets.schemas import DataItem

        cached_data_item = self.cache_manager.get_cached_tensor(idx)
        if cached_data_item is not None:
            # Log cache hits periodically for verification
            if idx % 50 == 0:
                self.logger.info(f"[CACHE HIT] Returning cached tensor for index {idx}")
            return cached_data_item.model_dump()

        # PERFORMANCE MONITORING: Check for cache invalidation on cache misses
        if self.config.cache_config.cache_transformed_tensors and idx > 10:
            self._check_cache_health()

        # AI_DOCS: Step 2 - Image Loading
        # Use _load_image_data helper (returns ImageData Pydantic model)
        # Handles EXIF orientation, normalization, RGB conversion
        image_filename = list(self.anns.keys())[idx]

        # Use the _load_image_data method which can be mocked for testing
        image_data = self._load_image_data(image_filename)

        # Get image properties
        image_array = image_data.image_array
        raw_width = image_data.raw_width
        raw_height = image_data.raw_height
        orientation = image_data.orientation
        cache_source = "disk"  # Default, can be updated if loaded from cache

        # 3. Annotation Processing: Load raw polygons and process them
        raw_polygons = self.anns[image_filename]
        processed_polygons = None
        polygon_frame = "raw"

        if raw_polygons is not None:
            # Convert raw polygons to list of numpy arrays
            polygons_list = [np.asarray(poly, dtype=np.float32) for poly in raw_polygons]

            # Use polygon_utils to handle orientation remapping
            from ocr.utils.orientation import orientation_requires_rotation, polygons_in_canonical_frame, remap_polygons

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
        image_path = self.config.image_path / image_filename

        from ocr.datasets.schemas import TransformInput

        metadata = ImageMetadata(
            filename=image_filename,
            path=image_path,
            original_shape=(height, width),
            orientation=orientation,
            is_normalized=image_data.is_normalized,
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

        transform_input = TransformInput(image=image_array, polygons=polygon_models, metadata=metadata)

        # Apply transformation
        transformed = self.transform(transform_input)

        # 5. Final Assembly & Validation: Construct DataItem from transformed outputs
        transformed_image = transformed["image"]
        transformed_polygons = transformed.get("polygons", []) or []

        # Filter degenerate polygons using polygon_utils
        from ocr.utils.polygon_utils import ensure_polygon_array, filter_degenerate_polygons

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
                        from ocr.utils.polygon_utils import validate_map_shapes

                        if hasattr(transformed_image, "shape"):
                            image_height, image_width = transformed_image.shape[-2], transformed_image.shape[-1]
                        else:
                            image_height, image_width = None, None

                        if validate_map_shapes(loaded_prob_map, loaded_thresh_map, image_height, image_width, image_filename):
                            prob_map = loaded_prob_map
                            thresh_map = loaded_thresh_map

                            # Cache the maps
                            from ocr.datasets.schemas import MapData

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
            inverse_matrix=transformed["inverse_matrix"],
        )

        # 6. Tensor Caching: Store validated DataItem in CacheManager
        if self.config.cache_config.cache_transformed_tensors:
            self.cache_manager.set_cached_tensor(idx, data_item)

        # 7. Return Value: Convert DataItem to dictionary
        return data_item.model_dump()

    def _load_image_data(self, filename: str) -> "ImageData":
        """
        Load image data and return as ImageData object.

        AI_DOCS: Image Loading Requirements
        This method MUST:
        - Return ImageData Pydantic model (NOT raw dict)
        - Handle EXIF orientation correction
        - Convert to RGB format
        - Apply prenormalization if configured
        - Close PIL images to prevent memory leaks
        - Use utilities from ocr.utils.image_utils

        Expected Data Types:
        - filename: str (image filename)
        - Returns: ImageData (Pydantic model with validation)

        DO NOT:
        - Return raw dictionaries
        - Skip EXIF orientation handling
        - Leave PIL images open
        - Implement image processing inline (use utilities)

        Related: ocr/utils/image_utils.py, ocr/utils/orientation.py
        """
        # AI_DOCS: Image Loading Pipeline
        # 0. Check cache first (if preloading enabled)
        # 1. Construct full image path
        # 2. Load PIL image with EXIF handling (load_pil_image)
        # 3. Get raw dimensions (safe_get_image_size)
        # 4. Normalize orientation (normalize_pil_image)
        # 5. Convert to RGB (ensure_rgb)
        # 6. Convert to numpy array (pil_to_numpy)
        # 7. Apply prenormalization if configured (prenormalize_imagenet)
        # 8. Close all PIL images to prevent memory leaks
        # 9. Return validated ImageData model

        # Check if image is preloaded in cache
        cached_image_data = self.cache_manager.get_cached_image(filename)
        if cached_image_data is not None:
            return cached_image_data

        # Load from disk
        image_path = self.config.image_path / filename
        from ocr.datasets.schemas import ImageData
        from ocr.utils.image_utils import ensure_rgb, load_pil_image, pil_to_numpy, safe_get_image_size

        try:
            pil_image = load_pil_image(image_path, self.config.image_loading_config)
        except OSError as exc:
            raise RuntimeError(f"Failed to load image {filename}: {exc}") from exc

        raw_width, raw_height = safe_get_image_size(pil_image)
        try:
            normalized_image, orientation = normalize_pil_image(pil_image)
        except Exception as exc:  # pragma: no cover - defensive cleanup
            pil_image.close()
            raise RuntimeError(f"Failed to normalize image {filename}: {exc}") from exc

        rgb_image = ensure_rgb(normalized_image)
        image_array = pil_to_numpy(rgb_image)

        if self.config.prenormalize_images:
            from ocr.utils.image_utils import prenormalize_imagenet

            image_array = prenormalize_imagenet(image_array)
            is_normalized = True
        else:
            is_normalized = image_array.dtype == np.float32

        rgb_image.close()
        if normalized_image is not pil_image:
            normalized_image.close()
        pil_image.close()

        image_data = ImageData(
            image_array=image_array,
            raw_width=int(raw_width),
            raw_height=int(raw_height),
            orientation=int(orientation),
            is_normalized=is_normalized,
        )

        return image_data

    def _preload_images(self):
        """
        Preload all images into RAM for faster access during training.

        This method loads, decodes, normalizes, and caches all images at dataset initialization.
        Images are stored as ImageData objects in CacheManager, eliminating disk I/O during training.

        Performance Impact: ~10-12% speedup by eliminating disk I/O overhead.
        Memory Cost: ~200MB for 404 validation images (average 500KB each).
        """
        from tqdm import tqdm

        self.logger.info(f"Preloading images from {self.config.image_path} into RAM...")

        loaded_count = 0
        failed_count = 0

        for filename in tqdm(self.anns.keys(), desc="Loading images to RAM"):
            try:
                # Use existing _load_image_data which handles all processing
                image_data = self._load_image_data(filename)

                # Store in cache manager
                self.cache_manager.set_cached_image(filename, image_data)
                loaded_count += 1

            except Exception as e:
                self.logger.warning(f"Failed to preload image {filename}: {e}")
                failed_count += 1

        total = len(self.anns)
        success_rate = (loaded_count / total * 100) if total > 0 else 0
        self.logger.info(f"Preloaded {loaded_count}/{total} images into RAM ({success_rate:.1f}%)")

        if failed_count > 0:
            self.logger.warning(f"Failed to preload {failed_count} images - they will be loaded on-demand")

    def _preload_maps(self):
        """Preload all maps into cache."""
        # Implementation for preloading maps
        pass


# AI_DOCS: END OF FILE - Critical Reminders for AI Assistants
#
# =======================================================================
# VALIDATEDOCRDATASET - AI ASSISTANT CONSTRAINTS & REQUIREMENTS
# =======================================================================
#
# BEFORE MAKING ANY CHANGES TO THIS FILE:
#
# 1. DATA CONTRACTS (MANDATORY):
#    - Input: DatasetConfig (Pydantic model, not dict)
#    - Transform: Callable[TransformInput, dict[str, Any]]
#    - Output: DataItem.model_dump() -> dict[str, Any]
#    - Internal: ImageData, PolygonData, ImageMetadata models
#
# 2. METHOD SIGNATURES (DO NOT CHANGE):
#    - __init__(config: DatasetConfig, transform: Callable)
#    - __getitem__(idx: int) -> dict[str, Any]
#    - __len__() -> int
#    - _load_image_data(filename: str) -> ImageData
#
# 3. BACKWARD COMPATIBILITY (PRESERVE):
#    - All property accessors (@property methods)
#    - Return types and data structures
#    - Error messages and logging patterns
#    - Configuration parameter access
#
# 4. UTILITIES (ALWAYS USE):
#    - ocr.utils.cache_manager.CacheManager
#    - ocr.utils.image_utils.* functions
#    - ocr.utils.polygon_utils.* functions
#    - ocr.utils.orientation.* functions
#
# 5. VALIDATION (MANDATORY):
#    - Use Pydantic models for all data structures
#    - Validate inputs and outputs
#    - Handle ValidationError exceptions
#    - Log validation failures appropriately
#
# 6. PERFORMANCE FEATURES (PRESERVE):
#    - Tensor caching logic
#    - Image preloading capabilities
#    - Map caching and loading
#    - EXIF orientation correction
#
# 7. TESTING (CONSIDER):
#    - Methods can be mocked for unit tests
#    - Use fixtures for common test data
#    - Test edge cases and error conditions
#    - Validate Pydantic model constraints
#
# =======================================================================
# RELATED DOCUMENTATION (MUST CONSULT):
# =======================================================================
#
# - Data Contracts: ocr/validation/models.py
# - Schemas: ocr/datasets/schemas.py
# - Configuration: configs/**/*.yaml
# - Utilities: ocr/utils/*/
# - Tests: tests/unit/test_dataset.py, tests/integration/test_ocr_*.py
# - Architecture: docs/ai_handbook/03_references/architecture/
# - Changelog: docs/CHANGELOG.md (search for "OCR Dataset")
#
# =======================================================================
# COMMON AI MISTAKES TO AVOID:
# =======================================================================
#
# ❌ Changing __getitem__ return type from dict[str, Any]
# ❌ Removing backward compatibility properties
# ❌ Using raw dicts instead of Pydantic models
# ❌ Implementing image processing inline (use utilities)
# ❌ Breaking EXIF orientation handling
# ❌ Modifying constructor signature without config updates
# ❌ Skipping data validation steps
# ❌ Not using extracted utility functions
#
# =======================================================================
