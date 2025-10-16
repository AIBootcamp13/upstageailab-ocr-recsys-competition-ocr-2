"""
Demonstration script for Intelligent Brightness Adjustment.

Tests the brightness adjustment module on real images and synthetic test cases.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ocr.datasets.preprocessing.intelligent_brightness import BrightnessConfig, BrightnessMethod, IntelligentBrightnessAdjuster


def create_test_images():
    """Create synthetic test images with different brightness issues."""
    images = {}

    # 1. Dark image with text
    dark = np.ones((400, 600), dtype=np.uint8) * 50
    cv2.putText(dark, "Dark Document", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, 150, 3)
    images["dark"] = dark

    # 2. Bright/overexposed image
    bright = np.ones((400, 600), dtype=np.uint8) * 220
    cv2.putText(bright, "Bright Document", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, 180, 3)
    images["bright"] = bright

    # 3. Low contrast image
    low_contrast = np.ones((400, 600), dtype=np.uint8) * 128
    cv2.rectangle(low_contrast, (100, 100), (500, 300), 135, -1)
    cv2.putText(low_contrast, "Low Contrast", (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 140, 2)
    images["low_contrast"] = low_contrast

    # 4. Uneven lighting (gradient)
    uneven = np.zeros((400, 600), dtype=np.uint8)
    for i in range(400):
        brightness = int(50 + (i / 400) * 150)
        uneven[i, :] = brightness
    cv2.putText(uneven, "Uneven Lighting", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 2)
    images["uneven"] = uneven

    return images


def test_auto_method_selection():
    """Test automatic method selection on various images."""
    print("\n" + "=" * 80)
    print("TEST 1: Auto Method Selection")
    print("=" * 80)

    adjuster = IntelligentBrightnessAdjuster()
    images = create_test_images()

    for name, image in images.items():
        result = adjuster.adjust_brightness(image)

        print(f"\n{name.upper()}:")
        print(f"  Original brightness: {np.mean(image):.1f}")
        print(f"  Adjusted brightness: {np.mean(result.adjusted_image):.1f}")
        print(f"  Method selected: {result.method_used.value}")
        print(f"  Quality score: {result.quality_metrics.overall_quality:.3f}")
        print(f"  - Contrast: {result.quality_metrics.contrast_score:.3f}")
        print(f"  - Uniformity: {result.quality_metrics.brightness_uniformity:.3f}")
        print(f"  - Histogram spread: {result.quality_metrics.histogram_spread:.3f}")
        print(f"  - Text preservation: {result.quality_metrics.text_preservation_score:.3f}")
        print(f"  Processing time: {result.processing_time_ms:.2f}ms")


def test_individual_methods():
    """Test each brightness adjustment method individually."""
    print("\n" + "=" * 80)
    print("TEST 2: Individual Methods")
    print("=" * 80)

    # Test on dark image
    dark = np.ones((400, 600), dtype=np.uint8) * 50
    cv2.putText(dark, "Test Document", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, 150, 3)

    methods = [
        BrightnessMethod.CLAHE,
        BrightnessMethod.GAMMA_CORRECTION,
        BrightnessMethod.ADAPTIVE_HISTOGRAM,
        BrightnessMethod.CONTENT_AWARE,
    ]

    print(f"\nOriginal image brightness: {np.mean(dark):.1f}")

    for method in methods:
        config = BrightnessConfig(method=method)
        adjuster = IntelligentBrightnessAdjuster(config)
        result = adjuster.adjust_brightness(dark)

        print(f"\n{method.value.upper()}:")
        print(f"  Adjusted brightness: {np.mean(result.adjusted_image):.1f}")
        print(f"  Contrast (std): {np.std(result.adjusted_image):.1f}")
        print(f"  Quality score: {result.quality_metrics.overall_quality:.3f}")
        print(f"  Processing time: {result.processing_time_ms:.2f}ms")


def test_gamma_values():
    """Test different gamma values."""
    print("\n" + "=" * 80)
    print("TEST 3: Gamma Correction Values")
    print("=" * 80)

    dark = np.ones((400, 600), dtype=np.uint8) * 50

    gamma_values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    print(f"\nOriginal image brightness: {np.mean(dark):.1f}")
    print("\nGamma values (gamma > 1.0 brightens, gamma < 1.0 darkens):")

    for gamma in gamma_values:
        config = BrightnessConfig(method=BrightnessMethod.GAMMA_CORRECTION, auto_gamma=False, gamma_value=gamma)
        adjuster = IntelligentBrightnessAdjuster(config)
        result = adjuster.adjust_brightness(dark)

        print(f"  Gamma {gamma:.2f}: brightness = {np.mean(result.adjusted_image):.1f}")


def test_real_images():
    """Test on real images from the dataset if available."""
    print("\n" + "=" * 80)
    print("TEST 4: Real Images (if available)")
    print("=" * 80)

    # Check if LOW_PERFORMANCE_IMGS_canonical exists
    data_dir = Path("/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/LOW_PERFORMANCE_IMGS_canonical")

    if not data_dir.exists():
        print("\nDataset not found. Skipping real image tests.")
        return

    # Get first 3 images
    image_files = sorted(data_dir.glob("*.jpg"))[:3]

    if not image_files:
        print("\nNo images found in dataset. Skipping real image tests.")
        return

    adjuster = IntelligentBrightnessAdjuster()

    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")

        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print("  Failed to load image")
            continue

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply brightness adjustment
        result = adjuster.adjust_brightness(img)

        print(f"  Original brightness: {np.mean(gray):.1f}")
        result_gray = cv2.cvtColor(result.adjusted_image, cv2.COLOR_BGR2GRAY)
        print(f"  Adjusted brightness: {np.mean(result_gray):.1f}")
        print(f"  Method used: {result.method_used.value}")
        print(f"  Quality score: {result.quality_metrics.overall_quality:.3f}")
        print(f"  Processing time: {result.processing_time_ms:.2f}ms")


def test_quality_metrics():
    """Test quality metric calculations."""
    print("\n" + "=" * 80)
    print("TEST 5: Quality Metrics")
    print("=" * 80)

    images = create_test_images()
    adjuster = IntelligentBrightnessAdjuster()

    print("\nQuality Metrics Breakdown:")
    print("-" * 80)
    print(f"{'Image':<15} {'Overall':<10} {'Contrast':<10} {'Uniformity':<12} {'Histogram':<10} {'Text Pres.':<10}")
    print("-" * 80)

    for name, image in images.items():
        result = adjuster.adjust_brightness(image)
        m = result.quality_metrics

        print(
            f"{name:<15} {m.overall_quality:<10.3f} {m.contrast_score:<10.3f} "
            f"{m.brightness_uniformity:<12.3f} {m.histogram_spread:<10.3f} "
            f"{m.text_preservation_score:<10.3f}"
        )


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("INTELLIGENT BRIGHTNESS ADJUSTMENT - DEMONSTRATION")
    print("=" * 80)

    try:
        test_auto_method_selection()
        test_individual_methods()
        test_gamma_values()
        test_real_images()
        test_quality_metrics()

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
