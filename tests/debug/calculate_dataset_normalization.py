#!/usr/bin/env python3
"""
Calculate dataset normalization statistics for comparison with ImageNet values.

This script computes the mean and standard deviation of RGB channels
across a sample of training images to determine optimal normalization parameters.
"""

import glob
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def calculate_dataset_stats(image_paths, max_images=1000):
    """
    Calculate mean and std for RGB channels across dataset images.

    Args:
        image_paths: List of paths to images
        max_images: Maximum number of images to process

    Returns:
        dict: Contains mean and std for each channel
    """
    # Initialize accumulators
    pixel_sum = np.zeros(3)
    pixel_sum_sq = np.zeros(3)
    total_pixels = 0

    # Sample images
    if len(image_paths) > max_images:
        np.random.seed(42)  # For reproducibility
        image_paths = np.random.choice(image_paths, max_images, replace=False)

    print(f"Processing {len(image_paths)} images...")

    for img_path in tqdm(image_paths):
        try:
            # Load image
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]

            # Accumulate statistics
            pixel_sum += img_array.sum(axis=(0, 1))
            pixel_sum_sq += (img_array**2).sum(axis=(0, 1))
            total_pixels += img_array.shape[0] * img_array.shape[1]

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Calculate mean and std
    mean = pixel_sum / total_pixels
    std = np.sqrt(pixel_sum_sq / total_pixels - mean**2)

    return {"mean": mean, "std": std, "total_pixels": total_pixels, "num_images": len(image_paths)}


def main():
    # Find all training images
    image_dir = Path("/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/images")
    image_paths = []

    # Look for images in train subdirectories
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(str(image_dir / "**" / ext), recursive=True))

    # Filter out test images if they exist
    train_paths = [p for p in image_paths if "test" not in p.lower()]

    if not train_paths:
        print("No training images found, using all available images")
        train_paths = image_paths

    print(f"Found {len(train_paths)} images")

    # Calculate statistics
    stats = calculate_dataset_stats(train_paths, max_images=500)  # Sample 500 images

    # ImageNet values for comparison
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    print("\n" + "=" * 50)
    print("NORMALIZATION STATISTICS COMPARISON")
    print("=" * 50)

    print("\nDataset Statistics (calculated):")
    print(f"  Mean: [{stats['mean'][0]:.4f}, {stats['mean'][1]:.4f}, {stats['mean'][2]:.4f}]")
    print(f"  Std:  [{stats['std'][0]:.4f}, {stats['std'][1]:.4f}, {stats['std'][2]:.4f}]")

    print("\nImageNet Statistics (current):")
    print(f"  Mean: [{imagenet_mean[0]:.4f}, {imagenet_mean[1]:.4f}, {imagenet_mean[2]:.4f}]")
    print(f"  Std:  [{imagenet_std[0]:.4f}, {imagenet_std[1]:.4f}, {imagenet_std[2]:.4f}]")

    print("\nDifference (Dataset - ImageNet):")
    print(
        f"  Mean: [{stats['mean'][0] - imagenet_mean[0]:.4f}, {stats['mean'][1] - imagenet_mean[1]:.4f}, {stats['mean'][2] - imagenet_mean[2]:.4f}]"
    )
    print(
        f"  Std:  [{stats['std'][0] - imagenet_std[0]:.4f}, {stats['std'][1] - imagenet_std[1]:.4f}, {stats['std'][2] - imagenet_std[2]:.4f}]"
    )

    print("\nRelative Difference (%):")
    mean_rel_diff = ((stats["mean"] - imagenet_mean) / imagenet_mean) * 100
    std_rel_diff = ((stats["std"] - imagenet_std) / imagenet_std) * 100
    print(f"  Mean: [{mean_rel_diff[0]:.1f}%, {mean_rel_diff[1]:.1f}%, {mean_rel_diff[2]:.1f}%]")
    print(f"  Std:  [{std_rel_diff[0]:.1f}%, {std_rel_diff[1]:.1f}%, {std_rel_diff[2]:.1f}%]")

    print(f"\nProcessed {stats['num_images']} images with {stats['total_pixels']:,} total pixels")


if __name__ == "__main__":
    main()
