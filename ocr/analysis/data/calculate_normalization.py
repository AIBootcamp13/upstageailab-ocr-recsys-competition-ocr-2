# ocr/analysis/data/calculate_normalization.py

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from tqdm import tqdm

from ...datasets import ValidatedOCRDataset
from ...datasets.schemas import DatasetConfig
from ...datasets.transforms import DBTransforms


def calculate_normalization_stats(cfg: DictConfig):
    print("🚀 Calculating normalization stats for the training dataset...")

    # Create a minimal transform for loading images
    temp_transform = DBTransforms(
        transforms=[{"_target_": "albumentations.pytorch.ToTensorV2", "p": 1.0}],
        keypoint_params={"format": "xy", "remove_invisible": True},
    )

    # Get dataset paths from config
    dataset_base_path = cfg.get("dataset_base_path", "/data/datasets/")
    train_image_path = f"{dataset_base_path}images/train"
    train_annotation_path = f"{dataset_base_path}jsons/train.json"

    # Create dataset config
    dataset_config = DatasetConfig(
        image_path=Path(train_image_path),
        annotation_path=Path(train_annotation_path),
        preload_images=False,  # Don't preload for stats calculation
        load_maps=False,
    )

    dataset = ValidatedOCRDataset(
        config=dataset_config,
        transform=temp_transform,
    )
    print(f"Found {len(dataset)} images in the training set.")

    channel_sum = np.zeros(3)
    channel_sum_sq = np.zeros(3)
    pixel_count = 0

    for i in tqdm(range(len(dataset)), desc="Processing images"):
        sample = dataset[i]
        image_tensor = sample["image"]  # Already a tensor from the transform

        # Convert to numpy and denormalize if needed
        if image_tensor.dtype == torch.float32:
            # Assuming the tensor is in [0,1] range, convert to [0,255]
            image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            image_np = image_tensor.permute(1, 2, 0).numpy()

        channel_sum += image_np.sum(axis=(0, 1))
        channel_sum_sq += (image_np**2).sum(axis=(0, 1))
        pixel_count += image_np.shape[0] * image_np.shape[1]

    mean = channel_sum / pixel_count
    std = np.sqrt((channel_sum_sq / pixel_count) - (mean**2))

    mean_norm, std_norm = mean / 255.0, std / 255.0

    print("\n🎉 Calculation Complete!")
    print("-----------------------------------------")
    print("Optimal Normalization Values:")
    print(f"mean: [{mean_norm[0]:.4f}, {mean_norm[1]:.4f}, {mean_norm[2]:.4f}]")
    print(f"std:  [{std_norm[0]:.4f}, {std_norm[1]:.4f}, {std_norm[2]:.4f}]")
    print("-----------------------------------------")
    print("\nUpdate `configs/transforms/default.yaml` with these values.")


if __name__ == "__main__":
    # Load config using the project's structure
    config_dir = os.path.join(os.path.dirname(__file__), "../../../configs")
    with initialize(config_path=config_dir, version_base="1.2"):
        cfg = compose(config_name="train")
        calculate_normalization_stats(cfg)
