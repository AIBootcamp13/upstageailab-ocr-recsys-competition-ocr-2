import json
import logging
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ocr.utils.orientation import (
    EXIF_ORIENTATION_TAG,
    FLIP_LEFT_RIGHT,
    normalize_pil_image,
    orientation_requires_rotation,
    polygons_in_canonical_frame,
    remap_polygons,
)

Image.MAX_IMAGE_PIXELS = 108000000
EXIF_ORIENTATION = EXIF_ORIENTATION_TAG  # Orientation Information: 274


class OCRDataset(Dataset):
    DEFAULT_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

    def __init__(
        self,
        image_path,
        annotation_path,
        transform,
        image_extensions=None,
        preload_maps=False,
        load_maps=False,
        preload_images=False,
        prenormalize_images=False,
        image_loading_config=None,
        cache_transformed_tensors=False,
    ):
        self.image_path = Path(image_path)
        self.transform = transform
        self.logger = logging.getLogger(__name__)
        self._canonical_frame_logged = set()
        self.preload_maps = preload_maps
        self.load_maps = load_maps
        self.preload_images = preload_images
        self.prenormalize_images = prenormalize_images
        self.cache_transformed_tensors = cache_transformed_tensors
        self.maps_cache = {}
        self.image_cache = {}
        self.tensor_cache = {}  # Cache for final transformed tensors

        # Cache statistics tracking (for verification)
        self._cache_hit_count = 0
        self._cache_miss_count = 0

        # Image loading configuration
        self.image_loading_config = image_loading_config or {"use_turbojpeg": False, "turbojpeg_fallback": False}

        # Allow configurable image extensions, fallback to default if not provided
        if image_extensions is None:
            self.image_extensions = self.DEFAULT_IMAGE_EXTENSIONS
        else:
            self.image_extensions = [ext.lower() for ext in image_extensions]

        self.anns = OrderedDict()

        # annotation_path가 없다면, image_path에서 이미지만 불러오기
        if annotation_path is None:
            for file in self.image_path.glob("*"):
                # Ensure file is a file (not a directory) and has a valid image extension
                if file.is_file() and file.suffix.lower() in self.image_extensions:
                    self.anns[file.name] = None
            return

        class AnnotationFileError(Exception):
            pass

        try:
            with open(annotation_path) as f:
                annotations = json.load(f)
        except FileNotFoundError:
            logging.error(f"Annotation file not found: {annotation_path}")
            raise AnnotationFileError(f"Annotation file not found: {annotation_path}")
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in annotation file: {annotation_path}")
            raise AnnotationFileError(f"Invalid JSON in annotation file: {annotation_path}")
        except Exception as e:
            logging.error(f"Error loading annotation file {annotation_path}: {e}")
            raise AnnotationFileError(f"Error loading annotation file {annotation_path}: {e}")

        for filename in annotations.get("images", {}).keys():
            # Image file이 경로에 존재하는지 확인
            if (self.image_path / filename).exists():
                # words 정보를 가지고 있는지 확인
                if "words" in annotations.get("images", {}).get(filename, {}):
                    # Words의 Points 변환
                    gt_words = annotations.get("images", {}).get(filename, {}).get("words", {})
                    polygons = [
                        np.array(np.round(word_data["points"]), dtype=np.int32)
                        for word_data in gt_words.values()
                        if isinstance(word_data.get("points"), list) and len(word_data["points"]) > 0
                    ]
                    self.anns[filename] = polygons or None
                else:
                    self.anns[filename] = None

        # Preload maps into RAM if requested
        if self.preload_maps:
            self._preload_maps_to_ram()

        # Preload images into RAM if requested
        if self.preload_images:
            self._preload_images_to_ram()

        # Log tensor caching status (Phase 6E)
        if self.cache_transformed_tensors:
            self.logger.info(f"Tensor caching enabled - will cache {len(self.anns)} transformed samples after first access")

    def _preload_maps_to_ram(self):
        """Preload all .npz maps into RAM for faster access."""
        maps_dir = self.image_path.parent / f"{self.image_path.name}_maps"

        if not maps_dir.exists():
            self.logger.info(f"Maps directory not found: {maps_dir}. Skipping RAM preloading.")
            return

        from tqdm import tqdm

        self.logger.info(f"Preloading maps from {maps_dir} into RAM...")

        loaded_count = 0
        for filename in tqdm(self.anns.keys(), desc="Loading maps to RAM"):
            map_filename = maps_dir / f"{Path(filename).stem}.npz"
            if map_filename.exists():
                try:
                    maps_data = np.load(map_filename)
                    # Store as dict to avoid keeping file handle open
                    self.maps_cache[filename] = {"prob_map": maps_data["prob_map"].copy(), "thresh_map": maps_data["thresh_map"].copy()}
                    loaded_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to load map for {filename}: {e}")

        self.logger.info(f"Preloaded {loaded_count}/{len(self.anns)} maps into RAM ({loaded_count / len(self.anns) * 100:.1f}%)")

    def _preload_images_to_ram(self):
        """Preload decoded PIL images to RAM for faster access."""
        from tqdm import tqdm

        if self.prenormalize_images:
            self.logger.info(f"Preloading and pre-normalizing images from {self.image_path} into RAM...")
        else:
            self.logger.info(f"Preloading images from {self.image_path} into RAM...")

        loaded_count = 0
        for filename in tqdm(self.anns.keys(), desc="Loading images to RAM"):
            image_path = self.image_path / filename
            if image_path.exists():
                try:
                    from ocr.utils.image_loading import load_image_optimized

                    pil_image = load_image_optimized(image_path)
                    # Normalize and convert to RGB immediately to avoid duplicate work later
                    normalized_image, orientation = normalize_pil_image(pil_image)
                    if normalized_image.mode != "RGB":
                        rgb_image = normalized_image.convert("RGB")
                        normalized_image.close()
                    else:
                        rgb_image = normalized_image

                    # Store as numpy array to save memory and avoid keeping PIL objects
                    # Also store metadata needed for __getitem__
                    raw_width, raw_height = self.safe_get_image_size(pil_image)
                    image_array = np.array(rgb_image)

                    # Pre-normalize if requested (ImageNet normalization)
                    if self.prenormalize_images:
                        # Convert to float32 and normalize in-place
                        image_array = image_array.astype(np.float32) / 255.0
                        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                        image_array = (image_array - mean) / std

                    self.image_cache[filename] = {
                        "image_array": image_array,
                        "raw_width": raw_width,
                        "raw_height": raw_height,
                        "orientation": orientation,
                        "is_normalized": self.prenormalize_images,
                    }

                    # Clean up PIL objects
                    rgb_image.close()
                    if normalized_image is not pil_image:
                        normalized_image.close()
                    pil_image.close()

                    loaded_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to preload image {filename}: {e}")

        if self.prenormalize_images:
            self.logger.info(
                f"Preloaded and pre-normalized {loaded_count}/{len(self.anns)} images into RAM ({loaded_count / len(self.anns) * 100:.1f}%)"
            )
        else:
            self.logger.info(f"Preloaded {loaded_count}/{len(self.anns)} images into RAM ({loaded_count / len(self.anns) * 100:.1f}%)")

    def log_cache_statistics(self):
        """
        Log tensor cache statistics. Call this at the end of each epoch to verify caching is working.
        Useful for debugging and performance monitoring.
        """
        if not self.cache_transformed_tensors:
            return

        total_accesses = self._cache_hit_count + self._cache_miss_count
        hit_rate = (self._cache_hit_count / total_accesses * 100) if total_accesses > 0 else 0

        self.logger.info(
            f"Tensor Cache Statistics - Hits: {self._cache_hit_count}, "
            f"Misses: {self._cache_miss_count}, "
            f"Hit Rate: {hit_rate:.1f}%, "
            f"Cache Size: {len(self.tensor_cache)}"
        )

        # Reset counters for next epoch
        self._cache_hit_count = 0
        self._cache_miss_count = 0

    def __len__(self):
        return len(self.anns.keys())

    @staticmethod
    def safe_get_image_size(image):
        """
        Safely extract (width, height) from PIL Image or NumPy array.

        Handles the type confusion between PIL.Image.size (returns (w, h))
        and numpy.ndarray.size (returns total element count).

        Args:
            image: PIL Image or NumPy array

        Returns:
            tuple: (width, height)

        Raises:
            TypeError: If image type is not supported
        """
        if isinstance(image, np.ndarray):
            # NumPy array: shape is (height, width, channels)
            height, width = image.shape[:2]
            return (width, height)  # Return (width, height) for consistency
        elif hasattr(image, "size"):
            # PIL Image or similar: .size returns (width, height)
            return image.size
        else:
            raise TypeError(f"Unsupported image type: {type(image)}. Expected PIL Image or NumPy array.")

    def __getitem__(self, idx):
        image_filename = list(self.anns.keys())[idx]

        # Check if final transformed tensor is cached (Phase 6E)
        if self.cache_transformed_tensors and idx in self.tensor_cache:
            self._cache_hit_count += 1
            # Log cache hits periodically to verify caching works
            if idx % 50 == 0:
                self.logger.info(f"[CACHE HIT] Returning cached tensor for index {idx} (file: {image_filename})")
            return self.tensor_cache[idx]

        # Track cache misses
        if self.cache_transformed_tensors:
            self._cache_miss_count += 1

        image_path = self.image_path / image_filename

        # Check if image is in cache
        if image_filename in self.image_cache:
            # Use preloaded image from RAM
            cached_data = self.image_cache[image_filename]
            image_array = cached_data["image_array"]
            raw_width = cached_data["raw_width"]
            raw_height = cached_data["raw_height"]
            orientation = cached_data["orientation"]

            # Always use numpy arrays for transforms (Albumentations/DBTransforms require numpy)
            # BUG FIX (BUG-2025-002): Previously converted to PIL Image when is_normalized=False,
            # causing AttributeError in transforms.py:42 (PIL Image has no .shape attribute)
            image = image_array  # Keep as numpy array (uint8 or float32)
        else:
            # Load from disk (original behavior)
            try:
                from ocr.utils.image_loading import load_image_optimized

                pil_image = load_image_optimized(
                    image_path,
                    use_turbojpeg=self.image_loading_config["use_turbojpeg"],
                    turbojpeg_fallback=self.image_loading_config["turbojpeg_fallback"],
                )
            except OSError as e:
                raise RuntimeError(f"Failed to load image {image_filename}: {e}")

            raw_width, raw_height = self.safe_get_image_size(pil_image)
            try:
                normalized_image, orientation = normalize_pil_image(pil_image)
            except Exception as exc:
                pil_image.close()
                raise RuntimeError(f"Failed to normalize image {image_filename}: {exc}") from exc

            if normalized_image.mode != "RGB":
                rgb_image = normalized_image.convert("RGB")
            else:
                rgb_image = normalized_image.copy()

            # Convert to numpy array for consistency with cached path
            # BUG FIX (BUG-2025-002): Always pass numpy arrays to transforms
            image = np.array(rgb_image)
            rgb_image.close()

            if normalized_image is not pil_image:
                normalized_image.close()
            pil_image.close()

        raw_polygons = self.anns[image_filename] or None
        polygons = None
        polygon_frame = "raw"
        if raw_polygons:
            polygons_list = [np.asarray(poly, dtype=np.float32) for poly in raw_polygons]
            if orientation_requires_rotation(orientation):
                if polygons_in_canonical_frame(polygons_list, raw_width, raw_height, orientation):
                    polygons = polygons_list
                    polygon_frame = "canonical"
                    if image_filename not in self._canonical_frame_logged:
                        self.logger.debug(
                            "Skipping EXIF remap for %s; polygons already align with canonical orientation (orientation=%d).",
                            image_filename,
                            orientation,
                        )
                        self._canonical_frame_logged.add(image_filename)
                else:
                    polygons = remap_polygons(polygons_list, raw_width, raw_height, orientation)
                    polygon_frame = "canonical"
            else:
                polygons = polygons_list

        # Ensure shape is always a tuple (width, height)
        org_shape = self.safe_get_image_size(image)

        item: OrderedDict[str, Any] = OrderedDict(
            image=image,
            image_filename=image_filename,
            image_path=str(image_path),
            shape=org_shape,
            raw_size=(raw_width, raw_height),
            orientation=orientation,
            polygon_frame=polygon_frame,
        )

        if self.transform is None:
            raise ValueError("Transform function is a required value.")

        # Image transform
        # The transform pipeline handles PIL images and numpy arrays correctly.
        # The explicit np.array() call was causing issues with pre-normalized float32 arrays.
        transformed = self.transform(image=image, polygons=polygons)

        transformed_image = transformed["image"]
        transformed_polygons = transformed.get("polygons", []) or []

        # Normalize and filter polygons that became degenerate after transforms
        if transformed_polygons:
            normalized_polygons: list[np.ndarray] = []
            for poly in transformed_polygons:
                poly_array = self._ensure_polygon_array(np.asarray(poly, dtype=np.float32))
                if poly_array is None or poly_array.size == 0:
                    continue
                normalized_polygons.append(poly_array)

            filtered_polygons = self._filter_degenerate_polygons(normalized_polygons)
        else:
            filtered_polygons = []

        item["image"] = transformed_image
        item["polygons"] = filtered_polygons
        item["inverse_matrix"] = transformed["inverse_matrix"]

        if "metadata" in transformed:
            item["metadata"] = transformed["metadata"]

        # Load pre-processed probability and threshold maps (only if enabled)
        if self.load_maps:
            # First check RAM cache
            if image_filename in self.maps_cache:
                item["prob_map"] = self.maps_cache[image_filename]["prob_map"]
                item["thresh_map"] = self.maps_cache[image_filename]["thresh_map"]
            else:
                # Fallback to loading from disk
                maps_dir = self.image_path.parent / f"{self.image_path.name}_maps"
                map_filename = maps_dir / f"{Path(image_filename).stem}.npz"

                if map_filename.exists():
                    try:
                        maps_data = np.load(map_filename)
                        item["prob_map"] = maps_data["prob_map"]
                        item["thresh_map"] = maps_data["thresh_map"]
                    except Exception as e:
                        self.logger.warning(f"Failed to load maps for {image_filename}: {e}")
                        # If maps fail to load, we'll let the collate function handle it
                        pass

        # Cache the final transformed item if enabled (Phase 6E)
        if self.cache_transformed_tensors:
            self.tensor_cache[idx] = item

        return item

    @staticmethod
    def rotate_image(image, orientation):
        """
        Rotate image based on EXIF orientation.
        Handles orientations 1-8 according to EXIF standard.
        Orientation 1 (normal) requires no rotation.
        """

        try:
            bicubic = Image.Resampling.BICUBIC  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - Pillow<9 fallback
            bicubic = Image.BICUBIC  # type: ignore[attr-defined]

        fillcolor: tuple[int, ...]
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            fillcolor = (0, 0, 0, 0)
        else:
            fillcolor = (255, 255, 255)

        rotate_kwargs: dict[str, Any] = {
            "resample": bicubic,
            "fillcolor": fillcolor,
        }

        if orientation == 2:
            image = image.transpose(FLIP_LEFT_RIGHT)
        elif orientation == 3:
            image = image.rotate(180, **rotate_kwargs)
        elif orientation == 4:
            image = image.rotate(180, **rotate_kwargs).transpose(FLIP_LEFT_RIGHT)
        elif orientation == 5:
            image = image.rotate(-90, expand=True, **rotate_kwargs).transpose(FLIP_LEFT_RIGHT)
        elif orientation == 6:
            image = image.rotate(-90, expand=True, **rotate_kwargs)
        elif orientation == 7:
            image = image.rotate(90, expand=True, **rotate_kwargs).transpose(FLIP_LEFT_RIGHT)
        elif orientation == 8:
            image = image.rotate(90, expand=True, **rotate_kwargs)
        # Orientation 1 (normal) and any other values: no rotation needed
        return image

    @staticmethod
    def _ensure_polygon_array(polygon: np.ndarray | Sequence | None) -> np.ndarray | None:
        if polygon is None:
            return None
        polygon_array = polygon if isinstance(polygon, np.ndarray) else np.asarray(polygon, dtype=np.float32)

        if polygon_array.size == 0:
            return polygon_array

        if polygon_array.ndim == 1:
            if polygon_array.size % 2 != 0:
                return polygon_array.reshape(1, -1)
            return polygon_array.reshape(1, -1, 2)
        if polygon_array.ndim == 2:
            if polygon_array.shape[0] == 1:
                return polygon_array
            return polygon_array[np.newaxis, ...]
        if polygon_array.ndim == 3 and polygon_array.shape[0] == 1:
            return polygon_array
        return polygon_array.reshape(1, -1, 2)

    @staticmethod
    def _clip_polygons_in_place(polygons: Iterable[np.ndarray | None], width: int, height: int) -> None:
        for polygon in polygons:
            if polygon is None:
                continue
            polygon[..., 0] = np.clip(polygon[..., 0], 0, max(width - 1, 0))
            polygon[..., 1] = np.clip(polygon[..., 1], 0, max(height - 1, 0))

    @staticmethod
    def _filter_degenerate_polygons(
        polygons: Iterable[np.ndarray | None],
        min_side: float = 1.0,
    ) -> list[np.ndarray]:
        removed_counts = {
            "too_few_points": 0,
            "too_small": 0,
            "zero_span": 0,
            "empty": 0,
            "none": 0,
        }
        filtered: list[np.ndarray] = []
        for polygon in polygons:
            if polygon is None:
                removed_counts["none"] += 1
                continue
            if polygon.size == 0:
                removed_counts["empty"] += 1
                continue

            reshaped = polygon.reshape(-1, 2)
            if reshaped.shape[0] < 3:
                removed_counts["too_few_points"] += 1
                continue

            width_span = float(reshaped[:, 0].max() - reshaped[:, 0].min())
            height_span = float(reshaped[:, 1].max() - reshaped[:, 1].min())

            rounded = np.rint(reshaped).astype(np.int32, copy=False)
            width_span_int = int(rounded[:, 0].ptp())
            height_span_int = int(rounded[:, 1].ptp())

            if width_span < min_side or height_span < min_side:
                removed_counts["too_small"] += 1
                continue

            if width_span_int == 0 or height_span_int == 0:
                removed_counts["zero_span"] += 1
                continue

            filtered.append(polygon)

        total_removed = sum(removed_counts.values())
        if total_removed > 0:
            logger = logging.getLogger(__name__)
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "\nFiltered %d degenerate polygons (too_few_points=%d, too_small=%d, zero_span=%d, empty=%d, none=%d)",
                    total_removed,
                    removed_counts["too_few_points"],
                    removed_counts["too_small"],
                    removed_counts["zero_span"],
                    removed_counts["empty"],
                    removed_counts["none"],
                )

        return filtered
