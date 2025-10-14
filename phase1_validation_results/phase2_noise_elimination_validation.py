"""
Phase 2 Validation: Advanced Noise Elimination Effectiveness

This script validates that the noise elimination implementation achieves
the >90% effectiveness target for Phase 2.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ocr.datasets.preprocessing.advanced_noise_elimination import AdvancedNoiseEliminator, NoiseEliminationConfig, NoiseReductionMethod


def create_test_images():
    """Create test images with various types of noise."""
    test_images = []

    # Test 1: Clean document with Gaussian noise
    img1 = np.ones((400, 600), dtype=np.uint8) * 255
    cv2.rectangle(img1, (100, 100), (500, 300), 200, -1)
    cv2.putText(img1, "Test Document", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)

    noise = np.random.normal(0, 20, img1.shape).astype(np.int16)
    noisy1 = np.clip(img1.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    test_images.append(("Gaussian Noise", img1, noisy1))

    # Test 2: Document with shadows
    img2 = np.ones((400, 600), dtype=np.uint8) * 240
    cv2.rectangle(img2, (100, 100), (500, 300), 220, -1)

    # Add shadow gradient
    shadow = np.linspace(0.5, 1.0, 600).reshape(1, -1)
    shadow = np.repeat(shadow, 400, axis=0)
    shadowed = (img2 * shadow).astype(np.uint8)
    test_images.append(("Shadow Gradient", img2, shadowed))

    # Test 3: Document with salt-and-pepper noise
    img3 = np.ones((400, 600), dtype=np.uint8) * 255
    cv2.rectangle(img3, (100, 100), (500, 300), 200, -1)

    salt_pepper = np.random.random(img3.shape)
    noisy3 = img3.copy()
    noisy3[salt_pepper < 0.02] = 255
    noisy3[salt_pepper > 0.98] = 0
    test_images.append(("Salt-and-Pepper", img3, noisy3))

    # Test 4: Document with combined noise (realistic case)
    img4 = np.ones((400, 600), dtype=np.uint8) * 245
    cv2.rectangle(img4, (100, 100), (500, 300), 210, -1)
    cv2.putText(img4, "Important", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    cv2.putText(img4, "Information", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)

    # Add Gaussian noise
    noise = np.random.normal(0, 15, img4.shape).astype(np.int16)
    noisy4 = np.clip(img4.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add shadow
    shadow = np.ones_like(noisy4, dtype=np.float32)
    shadow[:, :200] *= 0.7
    noisy4 = (noisy4 * shadow).astype(np.uint8)

    # Add salt-and-pepper
    sp = np.random.random(noisy4.shape)
    noisy4[sp < 0.01] = 255
    noisy4[sp > 0.99] = 0

    test_images.append(("Combined Noise", img4, noisy4))

    return test_images


def calculate_psnr(original, cleaned):
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = np.mean((original.astype(np.float32) - cleaned.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index (simplified version)."""
    # Convert to float
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)

    # Calculate means
    mean1 = np.mean(img1_f)
    mean2 = np.mean(img2_f)

    # Calculate variances and covariance
    var1 = np.var(img1_f)
    var2 = np.var(img2_f)
    cov = np.mean((img1_f - mean1) * (img2_f - mean2))

    # SSIM constants
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    # SSIM formula
    numerator = (2 * mean1 * mean2 + c1) * (2 * cov + c2)
    denominator = (mean1**2 + mean2**2 + c1) * (var1 + var2 + c2)

    ssim = numerator / denominator
    return ssim


def validate_noise_elimination():
    """Run validation tests for noise elimination."""
    print("=" * 80)
    print("Phase 2 Validation: Advanced Noise Elimination")
    print("=" * 80)
    print()

    # Create eliminator with combined method
    config = NoiseEliminationConfig(method=NoiseReductionMethod.COMBINED, preserve_text_regions=True, content_aware=True)
    eliminator = AdvancedNoiseEliminator(config)

    # Get test images
    test_images = create_test_images()

    results = []

    for test_name, clean, noisy in test_images:
        print(f"Testing: {test_name}")
        print("-" * 80)

        # Process noisy image
        result = eliminator.eliminate_noise(noisy)

        # Calculate metrics
        effectiveness = result.effectiveness_score
        psnr_noisy = calculate_psnr(clean, noisy)
        psnr_cleaned = calculate_psnr(clean, result.cleaned_image)
        ssim_noisy = calculate_ssim(clean, noisy)
        ssim_cleaned = calculate_ssim(clean, result.cleaned_image)

        # Calculate improvement
        psnr_improvement = psnr_cleaned - psnr_noisy
        ssim_improvement = ssim_cleaned - ssim_noisy

        print(f"  Effectiveness Score: {effectiveness:.2%}")
        print(f"  PSNR (Noisy):        {psnr_noisy:.2f} dB")
        print(f"  PSNR (Cleaned):      {psnr_cleaned:.2f} dB")
        print(f"  PSNR Improvement:    {psnr_improvement:+.2f} dB")
        print(f"  SSIM (Noisy):        {ssim_noisy:.4f}")
        print(f"  SSIM (Cleaned):      {ssim_cleaned:.4f}")
        print(f"  SSIM Improvement:    {ssim_improvement:+.4f}")
        print()

        results.append(
            {"test": test_name, "effectiveness": effectiveness, "psnr_improvement": psnr_improvement, "ssim_improvement": ssim_improvement}
        )

    # Calculate overall statistics
    print("=" * 80)
    print("Overall Results")
    print("=" * 80)

    avg_effectiveness = np.mean([r["effectiveness"] for r in results])
    avg_psnr_improvement = np.mean([r["psnr_improvement"] for r in results])
    avg_ssim_improvement = np.mean([r["ssim_improvement"] for r in results])

    print(f"Average Effectiveness:     {avg_effectiveness:.2%}")
    print(f"Average PSNR Improvement:  {avg_psnr_improvement:+.2f} dB")
    print(f"Average SSIM Improvement:  {avg_ssim_improvement:+.4f}")
    print()

    # Check if target is met
    target_effectiveness = 0.90
    target_met = avg_effectiveness >= target_effectiveness

    print("=" * 80)
    print("Target Validation")
    print("=" * 80)
    print(f"Target Effectiveness:  >={target_effectiveness:.0%}")
    print(f"Achieved Effectiveness: {avg_effectiveness:.2%}")
    print(f"Status: {'✅ PASS' if target_met else '❌ FAIL'}")
    print()

    if target_met:
        print("🎉 Phase 2 Advanced Noise Elimination target achieved!")
    else:
        print("⚠️  Target not yet achieved. Additional tuning may be needed.")
        print(f"   Gap: {(target_effectiveness - avg_effectiveness):.2%}")

    print()

    return target_met, avg_effectiveness, results


if __name__ == "__main__":
    target_met, effectiveness, results = validate_noise_elimination()

    # Exit with appropriate code
    sys.exit(0 if target_met else 1)
