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

from ocr.datasets.schemas import DatasetConfig, ImageData, ImageMetadata, PolygonData, TransformInput
from ocr.utils.orientation import (
    EXIF_ORIENTATION_TAG,
    normalize_pil_image,
)

Image.MAX_IMAGE_PIXELS = 108000000
EXIF_ORIENTATION = EXIF_ORIENTATION_TAG  # Orientation Information: 274


# Refactored OCR dataset with Pydantic validation and separated concerns
class ValidatedOCRDataset(Dataset):
    """
    Refactored OCR dataset with Pydantic validation and separated concerns.
    """

    def __init__(self, config: "DatasetConfig", transform: Callable[["TransformInput"], dict[str, Any]]) -> None:
        """
        Initialize the validated OCR dataset.

        Args:
            config: DatasetConfig object containing all dataset configuration
            transform: Callable that takes TransformInput and returns transformed data dict
        """
        # Implementation details in pseudocode section
        self.config = config
        self.transform = transform
        self.logger = logging.getLogger(__name__)
        self._canonical_frame_logged: set[str] = set()

        # Initialize annotations dictionary
        self.anns: OrderedDict[str, list[np.ndarray] | None] = OrderedDict()

        # Load annotations using dedicated helper method
        self._load_annotations()

        # Instantiate CacheManager using configuration from config
        from ocr.utils.cache_manager import CacheManager

        self.cache_manager = CacheManager(config.cache_config)

        # Dispatch to preloading methods based on config
        if config.preload_images:
            self._preload_images()

        if config.preload_maps:
            self._preload_maps()

        # Log initialization status
        if config.cache_config.cache_transformed_tensors:
            self.logger.info(f"Tensor caching enabled - will cache {len(self.anns)} transformed samples after first access")

    # ------------------------------------------------------------------
    # Compatibility accessors for legacy consumers
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

        Args:
            idx: Sample index

        Returns:
            Dictionary containing the processed sample data
        """
        # 1. Tensor Cache Check: Start by checking CacheManager for a fully processed DataItem
        from ocr.datasets.schemas import DataItem

        cached_data_item = self.cache_manager.get_cached_tensor(idx)
        if cached_data_item is not None:
            # Log cache hits periodically for verification
            if idx % 50 == 0:
                self.logger.info(f"[CACHE HIT] Returning cached tensor for index {idx}")
            return cached_data_item.model_dump()

        # 2. Image Loading: If no tensor cached, get filename and load image data
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
        """Load image data and return as ImageData object."""
        # This is a helper method for testing
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
        """Preload all images into cache."""
        # Implementation for preloading images
        pass

    def _preload_maps(self):
        """Preload all maps into cache."""
        # Implementation for preloading maps
        pass
