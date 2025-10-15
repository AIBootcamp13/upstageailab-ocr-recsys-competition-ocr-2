#!/usr/bin/env python3
"""
Test script to compare conservative vs office-lens enhancement methods on multiple images
"""

import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Import our preprocessing module
sys.path.append("ocr")
from ocr.datasets.preprocessing import DocumentPreprocessor


def get_sample_images(num_samples=10):
    """Get a list of sample images from the dataset"""
    image_pattern = "data/datasets/images/test/*.jpg"
    image_paths = glob.glob(image_pattern)

    # Shuffle and select the requested number of images
    np.random.seed(42)  # For reproducible results
    selected_paths = np.random.choice(image_paths, min(num_samples, len(image_paths)), replace=False)

    return sorted(selected_paths)


def test_enhancement_methods_on_image(image_path):
    """Test both enhancement methods on a single image"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return None, None

    # Test conservative enhancement
    conservative_preprocessor = DocumentPreprocessor(
        enable_document_detection=False,
        enable_perspective_correction=False,
        enable_enhancement=True,
        enhancement_method="conservative",
        target_size=(640, 640),
    )

    conservative_result = conservative_preprocessor(image)
    conservative_image = conservative_result["image"]

    # Test office-lens enhancement
    office_lens_preprocessor = DocumentPreprocessor(
        enable_document_detection=False,
        enable_perspective_correction=False,
        enable_enhancement=True,
        enhancement_method="office_lens",
        target_size=(640, 640),
    )

    office_lens_result = office_lens_preprocessor(image)
    office_lens_image = office_lens_result["image"]

    return conservative_image, office_lens_image


def create_multi_image_comparison(image_paths, output_dir="enhancement_comparison_samples"):
    """Create comparison visualizations for multiple images"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {len(image_paths)} images...")

    # Create a figure with subplots for all images
    fig, axes = plt.subplots(len(image_paths), 3, figsize=(15, 5 * len(image_paths)))
    if len(image_paths) == 1:
        axes = axes.reshape(1, -1)  # Handle single image case

    successful_count = 0

    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i + 1}/{len(image_paths)}: {os.path.basename(image_path)}")

        # Load original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"  Skipping {image_path} - could not load")
            continue

        # Test both enhancement methods
        conservative_img, office_lens_img = test_enhancement_methods_on_image(image_path)
        if conservative_img is None or office_lens_img is None:
            print(f"  Skipping {image_path} - enhancement failed")
            continue

        # Plot original
        axes[i, 0].imshow(original_image[:, :, ::-1])
        axes[i, 0].set_title(f"Original\n{os.path.basename(image_path)}")
        axes[i, 0].axis("off")

        # Plot conservative enhancement
        axes[i, 1].imshow(conservative_img[:, :, ::-1])
        axes[i, 1].set_title("Conservative\nEnhancement")
        axes[i, 1].axis("off")

        # Plot office-lens enhancement
        axes[i, 2].imshow(office_lens_img[:, :, ::-1])
        axes[i, 2].set_title("Office Lens\nEnhancement")
        axes[i, 2].axis("off")

        successful_count += 1

    plt.tight_layout()
    comparison_file = f"{output_dir}/enhancement_comparison_{len(image_paths)}_images.png"
    plt.savefig(comparison_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSuccessfully processed {successful_count}/{len(image_paths)} images")
    print(f"Saved comparison to: {comparison_file}")

    return comparison_file


def create_individual_comparisons(image_paths, output_dir="enhancement_comparison_samples"):
    """Create individual comparison images for each sample"""
    os.makedirs(output_dir, exist_ok=True)

    print("Creating individual comparison images...")

    for i, image_path in enumerate(image_paths):
        print(f"Processing {i + 1}/{len(image_paths)}: {os.path.basename(image_path)}")

        # Load original
        original_image = cv2.imread(image_path)
        if original_image is None:
            continue

        # Get enhanced versions
        conservative_img, office_lens_img = test_enhancement_methods_on_image(image_path)
        if conservative_img is None or office_lens_img is None:
            continue

        # Create individual comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_image[:, :, ::-1])
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(conservative_img[:, :, ::-1])
        axes[1].set_title("Conservative Enhancement")
        axes[1].axis("off")

        axes[2].imshow(office_lens_img[:, :, ::-1])
        axes[2].set_title("Office Lens Enhancement")
        axes[2].axis("off")

        plt.tight_layout()
        filename = f"{output_dir}/comparison_{i + 1:02d}_{os.path.basename(image_path).replace('.jpg', '.png')}"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  Saved: {filename}")


def main():
    num_samples = 10
    output_dir = "enhancement_comparison_samples"

    print(f"=== Generating Enhancement Comparison for {num_samples} Images ===")

    # Get sample images
    image_paths = get_sample_images(num_samples)
    print(f"Selected {len(image_paths)} images for comparison")

    # Create multi-image comparison
    multi_comparison_file = create_multi_image_comparison(image_paths, output_dir)

    # Create individual comparisons
    create_individual_comparisons(image_paths, output_dir)

    print("\n=== Summary ===")
    print(f"Processed {len(image_paths)} images")
    print(f"Output directory: {output_dir}")
    print(f"Multi-image comparison: {multi_comparison_file}")
    print(f"Individual comparisons: {output_dir}/comparison_*.png")

    print("\nEnhancement Methods:")
    print("- Conservative: CLAHE + mild bilateral filter")
    print("- Office Lens: Gamma correction + CLAHE + saturation boost + sharpening + noise reduction")


if __name__ == "__main__":
    main()
