#!/usr/bin/env python3
"""
Calculate dataset normalization statistics AFTER preprocessing.

This script computes the mean and standard deviation of RGB channels
across a sample of PREPROCESSED training images to determine optimal
normalization parameters that account for enhancement effects.
"""

import glob
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


class ImagePreprocessor:
    """Apply the same preprocessing as used in training."""

    def __init__(self):
        self.enhancer = ImageEnhancer()

    def preprocess_image(self, image_array: np.ndarray) -> np.ndarray:
        """Apply the same preprocessing pipeline as the dataset."""
        # Convert to RGB if needed
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)

        # Apply enhancement (conservative method as per config)
        enhanced, _ = self.enhancer.enhance(image_array, "conservative")

        return enhanced


class ImageEnhancer:
    """Simplified version of the image enhancer used in preprocessing."""

    def enhance(self, image: np.ndarray, method: str) -> tuple[np.ndarray, list[str]]:
        if method == "office_lens":
            return self._enhance_image_office_lens(image)
        return self._enhance_image(image)

    def _enhance_image(self, image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Conservative enhancement method."""
        enhanced = image.copy()
        applied_enhancements = ["clahe_mild", "bilateral_filter_mild"]

        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        enhanced = cv2.bilateralFilter(enhanced, 5, 25, 25)

        return enhanced, applied_enhancements

    def _enhance_image_office_lens(self, image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Office Lens enhancement method."""
        enhanced = image.copy()
        applied_enhancements = [
            "gamma_correction",
            "clahe_lab",
            "saturation_boost",
            "sharpening",
            "noise_reduction",
        ]

        gamma = 1.1
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)

        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
        l_channel = clahe.apply(l)
        lab = cv2.merge([l_channel, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Saturation boost
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.1)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        # Noise reduction
        enhanced = cv2.bilateralFilter(enhanced, 5, 15, 15)

        return enhanced, applied_enhancements


def calculate_preprocessed_dataset_stats(image_paths, max_images=100):
    """
    Calculate mean and std for RGB channels across PREPROCESSED dataset images.

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

    # Initialize preprocessor
    preprocessor = ImagePreprocessor()

    # Sample images
    if len(image_paths) > max_images:
        np.random.seed(42)  # For reproducibility
        image_paths = np.random.choice(image_paths, max_images, replace=False)

    print(f"Processing {len(image_paths)} images with preprocessing...")

    for img_path in tqdm(image_paths):
        try:
            # Load image
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img, dtype=np.uint8)

            # Apply preprocessing (same as training)
            processed_array = preprocessor.preprocess_image(img_array)

            # Convert to float and normalize to [0,1] for statistics
            processed_float = processed_array.astype(np.float32) / 255.0

            # Accumulate statistics
            pixel_sum += processed_float.sum(axis=(0, 1))
            pixel_sum_sq += (processed_float**2).sum(axis=(0, 1))
            total_pixels += processed_float.shape[0] * processed_float.shape[1]

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

    # Calculate statistics from PREPROCESSED images
    stats = calculate_preprocessed_dataset_stats(train_paths, max_images=100)  # Sample 100 images

    # ImageNet values for comparison
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    # Previously calculated raw image stats
    raw_mean = np.array([0.5900, 0.5610, 0.5292])
    raw_std = np.array([0.2136, 0.2290, 0.2511])

    print("\n" + "=" * 60)
    print("NORMALIZATION STATISTICS COMPARISON (AFTER PREPROCESSING)")
    print("=" * 60)

    print("\nPreprocessed Dataset Statistics (AFTER enhancement):")
    print(".4f")
    print(".4f")

    print("\nRaw Dataset Statistics (BEFORE enhancement - what we calculated earlier):")
    print(".4f")
    print(".4f")

    print("\nImageNet Statistics (current config):")
    print(".4f")
    print(".4f")

    print("\nImpact of Preprocessing:")
    print("Raw → Preprocessed Mean Change:")
    mean_change = stats["mean"] - raw_mean
    print(f"  Δ: [{mean_change[0]:.4f}, {mean_change[1]:.4f}, {mean_change[2]:.4f}]")
    print("Raw → Preprocessed Std Change:")
    std_change = stats["std"] - raw_std
    print(f"  Δ: [{std_change[0]:.4f}, {std_change[1]:.4f}, {std_change[2]:.4f}]")

    print("\nPreprocessed vs ImageNet Mean Difference:")
    imagenet_diff = stats["mean"] - imagenet_mean
    print(f"  Δ: [{imagenet_diff[0]:.4f}, {imagenet_diff[1]:.4f}, {imagenet_diff[2]:.4f}]")
    print("\nPreprocessed vs ImageNet Std Difference:")
    imagenet_std_diff = stats["std"] - imagenet_std
    print(f"  Δ: [{imagenet_std_diff[0]:.4f}, {imagenet_std_diff[1]:.4f}, {imagenet_std_diff[2]:.4f}]")

    print(f"\nProcessed {stats['num_images']} images with {stats['total_pixels']:,} total pixels")
    print("\n⚠️  CRITICAL: These are the CORRECT normalization values to use with preprocessing!")
    print("   The previous calculation from raw images was INCORRECT for preprocessed data.")


if __name__ == "__main__":
    main()
