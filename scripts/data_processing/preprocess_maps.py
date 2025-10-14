# scripts/preprocess_maps.py
"""
Preprocess probability and threshold maps for DB text detection.

This script generates and saves probability/threshold maps for faster training.
Maps are saved as compressed .npz files with the following format:

Map File Format (.npz):
- prob_map: np.ndarray with shape (1, H, W), dtype=float32
  - Probability map where values indicate text region likelihood (0.0-1.0)
  - Channel dimension (1) for compatibility with PyTorch conv layers
- thresh_map: np.ndarray with shape (1, H, W), dtype=float32
  - Threshold map for adaptive binarization (typically 0.3-0.7 range)
  - Same shape as prob_map for element-wise operations

Validation:
- Both maps must have identical shapes
- Values must be finite (no NaN/Inf)
- prob_map values should be in [0, 1] range
- thresh_map values should be in [0, 1] range
- Files are validated after generation to ensure correctness

Usage:
    python scripts/preprocess_maps.py
"""

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from ocr.datasets import ValidatedOCRDataset

logging.basicConfig(level=logging.INFO)


def preprocess(cfg: DictConfig, dataset_key: str):
    """
    Generates and saves probability and threshold maps for a dataset.
    """
    logging.info(f"Initializing dataset and collate function for {dataset_key}...")
    try:
        # Validate config structure
        if "datasets" not in cfg or dataset_key not in cfg.datasets:
            raise ValueError(f"Dataset '{dataset_key}' not found in config.datasets")
        if "collate_fn" not in cfg:
            raise ValueError("collate_fn not found in config")

        # Use Hydra to instantiate the dataset and collate_fn from configs
        # This ensures all transforms are consistent with training.
        dataset: ValidatedOCRDataset = hydra.utils.instantiate(cfg.datasets[dataset_key])
        collate_fn = hydra.utils.instantiate(cfg.collate_fn)

        # Determine the number of samples to process
        num_samples_key = dataset_key.replace("_dataset", "_num_samples")
        limit = getattr(cfg.data, num_samples_key, None) if hasattr(cfg.data, num_samples_key) else None
        num_samples = min(len(dataset), limit) if limit and limit > 0 else len(dataset)
        logging.info(f"Processing {num_samples} samples for {dataset_key}")
    except Exception as e:
        logging.error(f"Failed to initialize for {dataset_key}: {e}")
        raise

    # Define the output directory for the pre-processed maps
    try:
        image_path = Path(dataset.config.image_path)
    except (AttributeError, TypeError):
        image_path = Path(dataset.image_path)  # OCRDataset has image_path as str
    output_dir = image_path.parent / f"{image_path.name}_maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory for maps: {output_dir}")

    # Main processing loop
    generated_count = 0

    for i in tqdm(range(num_samples), desc=f"Processing {image_path.name}"):
        try:
            sample = dataset[i]
            metadata_obj = sample.get("metadata")
            if metadata_obj is None:
                metadata = {}
            elif hasattr(metadata_obj, "model_dump"):
                metadata = metadata_obj.model_dump()
            elif isinstance(metadata_obj, dict):
                metadata = metadata_obj
            else:
                metadata = {}
            image_filename = metadata.get("filename") or f"sample_{i}"
            image_filename_str = str(image_filename)

            image_data = sample["image"]
            if isinstance(image_data, torch.Tensor):
                image_tensor = image_data.detach().clone()
            elif isinstance(image_data, np.ndarray):
                if image_data.ndim == 3:
                    image_array = np.ascontiguousarray(np.transpose(image_data, (2, 0, 1)))
                    image_tensor = torch.from_numpy(image_array)
                elif image_data.ndim == 2:
                    image_array = np.ascontiguousarray(np.expand_dims(image_data, axis=0))
                    image_tensor = torch.from_numpy(image_array)
                else:
                    raise TypeError(f"Unsupported numpy image shape {image_data.shape} from sample {image_filename_str}")
            else:
                raise TypeError(
                    f"Expected torch.Tensor or numpy.ndarray for image, got {type(image_data)} from sample {image_filename_str}"
                )

            polygons = sample.get("polygons", [])
            if not isinstance(polygons, list | tuple):
                logging.warning(
                    "Sample %s returned non-iterable polygons (%s); skipping",
                    image_filename,
                    type(polygons),
                )
                continue
            valid_polygons = [poly for poly in polygons if isinstance(poly, np.ndarray) and poly.ndim == 3 and poly.shape[1] >= 3]

            # Transform polygons if perspective correction was applied during preprocessing
            preprocessing_applied = metadata.get("processing_steps") and any(
                step in metadata["processing_steps"] for step in ["document_detection", "perspective_correction"]
            )

            if valid_polygons and preprocessing_applied:
                if metadata.get("perspective_matrix") is not None:
                    perspective_matrix = np.array(metadata["perspective_matrix"])
                    if perspective_matrix.shape == (3, 3):
                        transformed_polygons = []
                        for poly in valid_polygons:
                            try:
                                # Apply perspective transformation to polygon points
                                poly_points = poly[0]  # Remove batch dimension (1, N, 2) -> (N, 2)
                                # Convert to homogeneous coordinates
                                ones = np.ones((poly_points.shape[0], 1))
                                homogeneous_points = np.hstack([poly_points, ones])
                                # Apply transformation
                                transformed_points = perspective_matrix @ homogeneous_points.T
                                # Convert back to cartesian coordinates
                                transformed_points = transformed_points[:2] / transformed_points[2]
                                transformed_points = transformed_points.T
                                # Reshape back to (1, N, 2)
                                transformed_polygons.append(transformed_points.reshape(1, -1, 2))
                            except Exception as e:
                                logging.warning(f"Failed to transform polygon for {image_filename_str}: {e}")
                                continue
                        if transformed_polygons:
                            valid_polygons = transformed_polygons
                            logging.debug(f"Applied perspective transformation to {len(valid_polygons)} polygons for {image_filename_str}")
                        else:
                            logging.warning(f"All polygons failed transformation for {image_filename_str}, skipping sample")
                            continue
                    else:
                        logging.warning(
                            f"Invalid perspective matrix shape {perspective_matrix.shape} for {image_filename_str}, skipping sample"
                        )
                        continue
                else:
                    # Preprocessing was applied but no transformation matrix available
                    logging.warning(
                        f"Preprocessing applied to {image_filename_str} but no transformation matrix available, skipping sample"
                    )
                    continue

            if len(valid_polygons) == 0:
                continue

            # Generate the maps using the existing, proven logic
            maps = collate_fn.make_prob_thresh_map(image_tensor, valid_polygons, image_filename_str)

            prob_map = np.expand_dims(maps["prob_map"], axis=0)
            thresh_map = np.expand_dims(maps["thresh_map"], axis=0)

            # Save the maps to a compressed .npz file
            output_filename = output_dir / f"{Path(image_filename_str).stem}.npz"
            np.savez_compressed(output_filename, prob_map=prob_map, thresh_map=thresh_map)
            generated_count += 1
        except Exception as e:
            logging.error(f"Error processing sample {i}: {e}")
            continue

    if generated_count == 0:
        logging.warning("No probability maps generated for %s; check polygon availability or dataset configuration.", dataset_key)
        return

    # Comprehensive validation of generated maps
    validate_generated_maps(output_dir, generated_count)


def validate_generated_maps(output_dir: Path, expected_count: int):
    """
    Validate all generated map files for correctness.

    Args:
        output_dir: Directory containing the .npz map files
        expected_count: Expected number of map files

    Raises:
        ValueError: If validation fails
    """
    logging.info("Validating generated map files...")

    map_files = list(output_dir.glob("*.npz"))
    if len(map_files) != expected_count:
        raise ValueError(f"Expected {expected_count} map files, found {len(map_files)}")

    if not map_files:
        raise ValueError("No map files found to validate")

    validation_errors = []

    for map_file in map_files:
        try:
            data = np.load(map_file)

            # Check required keys exist
            if "prob_map" not in data:
                validation_errors.append(f"{map_file.name}: missing 'prob_map' key")
                continue
            if "thresh_map" not in data:
                validation_errors.append(f"{map_file.name}: missing 'thresh_map' key")
                continue

            prob_map = data["prob_map"]
            thresh_map = data["thresh_map"]

            # Validate shapes
            if prob_map.ndim != 3:
                validation_errors.append(f"{map_file.name}: prob_map should be 3D (C,H,W), got {prob_map.ndim}D")
                continue
            if thresh_map.ndim != 3:
                validation_errors.append(f"{map_file.name}: thresh_map should be 3D (C,H,W), got {thresh_map.ndim}D")
                continue

            if prob_map.shape[0] != 1:
                validation_errors.append(f"{map_file.name}: prob_map should have 1 channel, got {prob_map.shape[0]}")
                continue
            if thresh_map.shape[0] != 1:
                validation_errors.append(f"{map_file.name}: thresh_map should have 1 channel, got {thresh_map.shape[0]}")
                continue

            if prob_map.shape != thresh_map.shape:
                validation_errors.append(f"{map_file.name}: prob_map {prob_map.shape} != thresh_map {thresh_map.shape}")
                continue

            # Validate data types
            if prob_map.dtype != np.float32:
                validation_errors.append(f"{map_file.name}: prob_map should be float32, got {prob_map.dtype}")
                continue
            if thresh_map.dtype != np.float32:
                validation_errors.append(f"{map_file.name}: thresh_map should be float32, got {thresh_map.dtype}")
                continue

            # Validate value ranges
            if np.any(np.isnan(prob_map)) or np.any(np.isinf(prob_map)):
                validation_errors.append(f"{map_file.name}: prob_map contains NaN or Inf values")
                continue
            if np.any(np.isnan(thresh_map)) or np.any(np.isinf(thresh_map)):
                validation_errors.append(f"{map_file.name}: thresh_map contains NaN or Inf values")
                continue

            # Validate prob_map range (should be 0-1 for probability)
            if prob_map.min() < 0 or prob_map.max() > 1:
                validation_errors.append(
                    f"{map_file.name}: prob_map values out of range [0,1]: min={prob_map.min():.3f}, max={prob_map.max():.3f}"
                )
                continue

            # Validate thresh_map range (typically 0.3-0.7 based on DB paper)
            if thresh_map.min() < 0 or thresh_map.max() > 1:
                validation_errors.append(
                    f"{map_file.name}: thresh_map values out of range [0,1]: min={thresh_map.min():.3f}, max={thresh_map.max():.3f}"
                )
                continue

        except Exception as e:
            validation_errors.append(f"{map_file.name}: failed to load or validate - {e}")
            continue

    if validation_errors:
        error_msg = f"Map validation failed with {len(validation_errors)} errors:\n" + "\n".join(validation_errors)
        raise ValueError(error_msg)

    # Log successful validation
    sample_file = map_files[0]
    data = np.load(sample_file)
    logging.info(
        "✅ Map validation passed for %d files. Sample shapes: prob_map %s, thresh_map %s",
        len(map_files),
        data["prob_map"].shape,
        data["thresh_map"].shape,
    )


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    # Validate overall config
    if "datasets" not in cfg:
        raise ValueError("datasets not found in config")
    if "train_dataset" not in cfg.datasets or "val_dataset" not in cfg.datasets:
        raise ValueError("train_dataset or val_dataset not found in config.datasets")

    # This allows us to run preprocessing for both train and val sets
    logging.info("--- Pre-processing Training Data ---")
    preprocess(cfg, "train_dataset")

    logging.info("--- Pre-processing Validation Data ---")
    preprocess(cfg, "val_dataset")

    logging.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
