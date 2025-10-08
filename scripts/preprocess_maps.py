# scripts/preprocess_maps.py
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from ocr.datasets.base import OCRDataset

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
        dataset: OCRDataset = hydra.utils.instantiate(cfg.datasets[dataset_key])
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
    output_dir = Path(dataset.image_path).parent / f"{Path(dataset.image_path).name}_maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory for maps: {output_dir}")

    # Main processing loop
    generated_count = 0

    for i in tqdm(range(num_samples), desc=f"Processing {Path(dataset.image_path).name}"):
        try:
            sample = dataset[i]
            image_filename = sample.get("image_filename", f"sample_{i}")
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

    # Sanity check on one of the generated files
    sample_file = next(output_dir.glob("*.npz"))
    data = np.load(sample_file)
    assert data["prob_map"].shape[0] == 1, f"Prob map should have channel dim, got shape {data['prob_map'].shape}"
    assert data["thresh_map"].shape[0] == 1, f"Thresh map should have channel dim, got shape {data['thresh_map'].shape}"
    assert (
        data["prob_map"].shape == data["thresh_map"].shape
    ), f"Maps should have same shape, prob: {data['prob_map'].shape}, thresh: {data['thresh_map'].shape}"
    assert data["prob_map"].ndim == 3, f"Prob map should be 3D, got {data['prob_map'].ndim}D"
    logging.info(
        "Sanity check passed. Map shapes: prob_map %s, thresh_map %s",
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
