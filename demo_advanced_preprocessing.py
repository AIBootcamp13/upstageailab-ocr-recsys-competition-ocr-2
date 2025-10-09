#!/usr/bin/env python3
"""Demonstration script for advanced document preprocessing with Office Lens quality."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ocr.datasets.preprocessing import (
    AdvancedDocumentDetector,
    AdvancedDocumentPreprocessor,
    create_high_accuracy_preprocessor,
    create_office_lens_preprocessor,
)


def create_test_image() -> np.ndarray:
    """Create a simple test image with a document-like rectangle."""
    # Create a white background
    image = np.full((600, 800, 3), 255, dtype=np.uint8)

    # Add a gray document rectangle
    cv2.rectangle(image, (100, 100), (600, 500), (200, 200, 200), -1)

    # Add some texture/noise
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)

    # Add a border to make it look more like a document
    cv2.rectangle(image, (100, 100), (600, 500), (150, 150, 150), 3)

    return image


def demonstrate_advanced_detection(output_dir: str) -> None:
    """Demonstrate the advanced document detection capabilities."""
    print("🔍 Demonstrating Advanced Document Detection")
    print("=" * 50)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create test image
    test_image = create_test_image()
    cv2.imwrite(str(output_path / "test_input.png"), test_image)

    # Test different detection configurations
    configs = {
        "default": {},
        "strict": {"min_overall_confidence": 0.9, "min_geometric_confidence": 0.85},
        "relaxed": {"min_overall_confidence": 0.7, "min_geometric_confidence": 0.6},
    }

    for config_name, config_kwargs in configs.items():
        print(f"\nTesting {config_name} configuration...")

        detector = AdvancedDocumentDetector(logger=logging.getLogger(__name__), **config_kwargs)

        corners, method, metadata = detector.detect_document(test_image)

        if corners is not None:
            print(f"  ✅ Detection successful using method: {method}")
            print(".2f")
            print(f"  📊 Metadata: {len(metadata)} keys")

            # Draw detected corners on image
            debug_image = test_image.copy()
            cv2.polylines(debug_image, [corners.astype(np.int32)], True, (0, 255, 0), 3)
            cv2.imwrite(str(output_path / f"detection_{config_name}.png"), debug_image)
        else:
            print(f"  ❌ Detection failed for {config_name} configuration")


def demonstrate_office_lens_preprocessing(output_dir: str) -> None:
    """Demonstrate the Office Lens quality preprocessing pipeline."""
    print("\n📷 Demonstrating Office Lens Quality Preprocessing")
    print("=" * 50)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create test image
    test_image = create_test_image()

    # Test different preprocessor configurations
    preprocessors = {
        "office_lens": create_office_lens_preprocessor(logger=logging.getLogger(__name__)),
        "high_accuracy": create_high_accuracy_preprocessor(logger=logging.getLogger(__name__)),
        "custom": AdvancedDocumentPreprocessor(logger=logging.getLogger(__name__)),
    }

    for name, preprocessor in preprocessors.items():
        print(f"\nTesting {name} preprocessor...")

        try:
            result = preprocessor(test_image)
            processed_image = result["image"]
            metadata = result["metadata"]

            print("  ✅ Preprocessing successful")
            print(f"  📊 Processing steps: {metadata.get('processing_steps', [])}")
            print(f"  📏 Final shape: {processed_image.shape}")

            # Check for Office Lens quality metrics
            if "orientation" in metadata and metadata["orientation"]:
                orientation_data = metadata["orientation"]
                if "office_lens_quality_score" in orientation_data:
                    orientation_data["office_lens_quality_score"]
                    quality_achieved = orientation_data.get("office_lens_quality_achieved", False)
                    print(".2f")
                    print(f"  🎯 Office Lens quality achieved: {quality_achieved}")

            # Save processed image
            cv2.imwrite(str(output_path / f"preprocessed_{name}.png"), processed_image)

        except Exception as e:
            print(f"  ❌ Preprocessing failed: {e}")


def run_basic_validation(output_dir: str) -> None:
    """Run basic validation tests."""
    print("\n✅ Running Basic Validation Tests")
    print("=" * 50)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Test with synthetic dataset
    from ocr.datasets.preprocessing.advanced_detector_test import create_test_dataset

    print("Creating synthetic test dataset...")
    test_cases = create_test_dataset(output_path / "validation_data", num_samples=5)

    print(f"Created {len(test_cases)} test cases")

    # Test detection on synthetic data
    detector = AdvancedDocumentDetector(logger=logging.getLogger(__name__))
    successful_detections = 0

    for test_case in test_cases:
        image = cv2.imread(test_case["image_path"])
        if image is None:
            continue

        corners, method, metadata = detector.detect_document(image)

        if corners is not None:
            successful_detections += 1

    print(".1%")


def main() -> None:
    """Main demonstration function."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Use the specified output directory
    output_dir = "logs/DEBUG_PREPROCESSING_DOCTR_SCANNER"

    print("🚀 Advanced Document Preprocessing Demonstration")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()

    try:
        # Run demonstrations
        demonstrate_advanced_detection(output_dir)
        demonstrate_office_lens_preprocessing(output_dir)
        run_basic_validation(output_dir)

        print("\n🎉 Demonstration completed successfully!")
        print(f"Check the output directory for results: {output_dir}")

    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
