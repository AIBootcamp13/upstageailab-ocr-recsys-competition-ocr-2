"""
Demonstration script for document flattening functionality.

This script demonstrates the document flattening capabilities on real images
from the LOW_PERFORMANCE_IMGS_canonical dataset.
"""

from pathlib import Path

import cv2
import numpy as np

from ocr.datasets.preprocessing.document_flattening import DocumentFlattener, FlatteningConfig, FlatteningMethod


def demo_flattening_on_image(image_path: str, output_dir: str = "/tmp/flattening_demo"):
    """
    Demonstrate document flattening on a single image.

    Args:
        image_path: Path to input image
        output_dir: Directory to save results
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load image
    print(f"\n{'=' * 80}")
    print(f"Processing: {Path(image_path).name}")
    print(f"{'=' * 80}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image from {image_path}")
        return

    print(f"Image shape: {image.shape}")
    print(f"Image size: {image.shape[0]}x{image.shape[1]}")

    # Test all flattening methods
    methods = [FlatteningMethod.THIN_PLATE_SPLINE, FlatteningMethod.CYLINDRICAL, FlatteningMethod.SPHERICAL, FlatteningMethod.ADAPTIVE]

    results = {}

    for method in methods:
        print(f"\n--- Testing {method.value} method ---")

        # Create configuration
        config = FlatteningConfig(method=method, grid_size=20, smoothing_factor=0.1, enable_quality_assessment=True)

        # Create flattener and process image
        flattener = DocumentFlattener(config)
        result = flattener.flatten_document(image)

        results[method.value] = result

        # Print results
        print(f"Processing time: {result.processing_time_ms:.2f}ms")
        print(f"Method used: {result.method_used.value}")
        print(f"Warping confidence: {result.warping_transform.confidence:.3f}")

        if result.surface_normals:
            print(f"Mean curvature: {result.surface_normals.mean_curvature:.4f}")
            print(f"Max curvature: {result.surface_normals.max_curvature:.4f}")

        if result.quality_metrics:
            print("\nQuality Metrics:")
            print(f"  Distortion score: {result.quality_metrics.distortion_score:.3f}")
            print(f"  Edge preservation: {result.quality_metrics.edge_preservation_score:.3f}")
            print(f"  Smoothness: {result.quality_metrics.smoothness_score:.3f}")
            print(f"  Overall quality: {result.quality_metrics.overall_quality:.3f}")
            print(f"  Processing successful: {result.quality_metrics.processing_successful}")

        # Save flattened image
        output_filename = f"{Path(image_path).stem}_{method.value}.jpg"
        output_path = Path(output_dir) / output_filename
        cv2.imwrite(str(output_path), result.flattened_image)
        print(f"Saved to: {output_path}")

    # Create comparison visualization
    print("\n--- Creating comparison visualization ---")
    create_comparison_visualization(image, results, Path(output_dir) / f"{Path(image_path).stem}_comparison.jpg")


def create_comparison_visualization(original: np.ndarray, results: dict, output_path: Path):
    """
    Create a comparison visualization of all flattening methods.

    Args:
        original: Original image
        results: Dictionary of flattening results
        output_path: Path to save comparison image
    """
    # Resize images to consistent size for comparison
    target_width = 400
    target_height = int(original.shape[0] * target_width / original.shape[1])

    resized_original = cv2.resize(original, (target_width, target_height))

    # Create grid: 1 original + 4 methods = 5 images
    # Layout: 2 rows x 3 columns
    images_to_display = [resized_original]
    labels = ["Original"]

    for method_name, result in results.items():
        resized_result = cv2.resize(result.flattened_image, (target_width, target_height))
        images_to_display.append(resized_result)

        # Add quality score to label
        if result.quality_metrics:
            quality = result.quality_metrics.overall_quality
            labels.append(f"{method_name}\nQ: {quality:.2f}")
        else:
            labels.append(method_name)

    # Add padding for grid
    while len(images_to_display) < 6:
        images_to_display.append(np.zeros_like(resized_original))
        labels.append("")

    # Create grid
    rows = []
    text_height = 80  # Fixed text height
    for i in range(0, 6, 3):
        # Add text labels above images
        labeled_images = []
        for j in range(3):
            if i + j < len(images_to_display):
                img_with_label = images_to_display[i + j].copy()

                # Add label text
                label_text = labels[i + j]
                # Add white background for text (fixed height)
                text_area = np.ones((text_height, target_width, 3), dtype=np.uint8) * 255
                y_offset = 20
                for line_idx, line in enumerate(label_text.split("\n")):
                    cv2.putText(text_area, line, (10, y_offset + line_idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                img_with_label = np.vstack([text_area, img_with_label])
                labeled_images.append(img_with_label)

        if labeled_images:
            row = np.hstack(labeled_images)
            rows.append(row)

    comparison = np.vstack(rows)

    # Save comparison
    cv2.imwrite(str(output_path), comparison)
    print(f"Comparison saved to: {output_path}")


def main():
    """Main demonstration function."""
    print("\n" + "=" * 80)
    print("Document Flattening Demonstration")
    print("=" * 80)

    # Test on the specified image
    test_image_path = "data/datasets/LOW_PERFORMANCE_IMGS_canonical/drp.en_ko.in_house.selectstar_003795.jpg"

    if not Path(test_image_path).exists():
        print(f"\nERROR: Test image not found: {test_image_path}")
        print("Please ensure the image exists or adjust the path.")
        return

    # Run demonstration
    demo_flattening_on_image(test_image_path)

    # Also test on a few more images if available
    data_dir = Path("data/datasets/LOW_PERFORMANCE_IMGS_canonical")
    if data_dir.exists():
        other_images = list(data_dir.glob("*.jpg"))[:3]  # Test on first 3 images

        for img_path in other_images:
            if img_path.name != Path(test_image_path).name:
                demo_flattening_on_image(str(img_path))

    print("\n" + "=" * 80)
    print("Demonstration complete!")
    print("Results saved to: /tmp/flattening_demo/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
