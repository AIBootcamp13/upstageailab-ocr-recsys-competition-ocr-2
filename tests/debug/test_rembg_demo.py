#!/usr/bin/env python3
"""
Quick rembg demo - Test background removal on your images.

Usage:
    python test_rembg_demo.py path/to/image.jpg
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove


def test_rembg_basic():
    """Test rembg with a simple synthetic image."""
    print("🧪 Testing rembg with synthetic image...")

    # Create test image: white rectangle on black background
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (500, 300), (255, 255, 255), -1)
    cv2.putText(img, "Document", (200, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # Convert to PIL
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Remove background
    output = remove(pil_img)

    # Convert back to numpy
    output_array = np.array(output)

    print(f"✅ Input shape: {img.shape}")
    print(f"✅ Output shape: {output_array.shape} (includes alpha channel)")
    print(f"✅ Alpha channel min: {output_array[:,:,3].min()}, max: {output_array[:,:,3].max()}")

    # Save result
    output.save("rembg_test_output.png")
    print("📁 Saved: rembg_test_output.png")

    return output_array


def test_rembg_on_image(image_path: str):
    """Test rembg on a real image."""
    print(f"\n🖼️  Testing rembg on: {image_path}")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load image: {image_path}")
        return

    print(f"✅ Loaded image: {img.shape}")

    # Convert to PIL
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Remove background
    print("🔄 Removing background (may take a few seconds on first run)...")
    output = remove(pil_img)

    # Convert back to numpy (RGBA)
    output_array = np.array(output)

    # Extract alpha channel
    alpha = output_array[:, :, 3]

    # Create version with white background
    # Method: Composite RGBA over white
    rgb = output_array[:, :, :3]
    alpha_norm = alpha[:, :, np.newaxis] / 255.0

    white_bg = (rgb * alpha_norm + 255 * (1 - alpha_norm)).astype(np.uint8)

    # Convert back to BGR for OpenCV
    result_bgr = cv2.cvtColor(white_bg, cv2.COLOR_RGB2BGR)

    # Save results
    input_path = Path(image_path)
    output_dir = input_path.parent / "rembg_results"
    output_dir.mkdir(exist_ok=True)

    output_transparent = output_dir / f"{input_path.stem}_transparent.png"
    output_white_bg = output_dir / f"{input_path.stem}_white_bg.jpg"
    output_mask = output_dir / f"{input_path.stem}_mask.png"

    # Save transparent version
    output.save(str(output_transparent))
    print(f"📁 Saved (transparent): {output_transparent}")

    # Save white background version
    cv2.imwrite(str(output_white_bg), result_bgr)
    print(f"📁 Saved (white bg): {output_white_bg}")

    # Save mask
    cv2.imwrite(str(output_mask), alpha)
    print(f"📁 Saved (mask): {output_mask}")

    # Show stats
    foreground_pixels = np.sum(alpha > 128)
    total_pixels = alpha.size
    foreground_ratio = foreground_pixels / total_pixels

    print("\n📊 Statistics:")
    print(f"   Foreground ratio: {foreground_ratio:.1%}")
    print(f"   Background removed: {(1-foreground_ratio):.1%}")

    return output_array


def create_comparison_image(original_path: str):
    """Create side-by-side comparison."""
    print("\n🎨 Creating before/after comparison...")

    # Load original
    original = cv2.imread(original_path)
    if original is None:
        print(f"❌ Failed to load: {original_path}")
        return

    # Load processed (white bg version)
    input_path = Path(original_path)
    output_dir = input_path.parent / "rembg_results"
    processed_path = output_dir / f"{input_path.stem}_white_bg.jpg"

    if not processed_path.exists():
        print(f"❌ Processed image not found: {processed_path}")
        return

    processed = cv2.imread(str(processed_path))

    # Resize to same height
    h = min(original.shape[0], processed.shape[0])
    original = cv2.resize(original, (int(original.shape[1] * h / original.shape[0]), h))
    processed = cv2.resize(processed, (int(processed.shape[1] * h / processed.shape[0]), h))

    # Create side-by-side
    comparison = np.hstack([original, processed])

    # Add labels
    cv2.putText(comparison, "BEFORE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "AFTER (rembg)", (original.shape[1] + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save comparison
    comparison_path = output_dir / f"{input_path.stem}_comparison.jpg"
    cv2.imwrite(str(comparison_path), comparison)
    print(f"📁 Saved comparison: {comparison_path}")

    return comparison


def main():
    """Main demo function."""
    print("=" * 60)
    print("rembg Background Removal Demo")
    print("=" * 60)

    # Test 1: Basic functionality
    test_rembg_basic()

    # Test 2: Real image (if provided)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if Path(image_path).exists():
            test_rembg_on_image(image_path)
            create_comparison_image(image_path)
        else:
            print(f"❌ Image not found: {image_path}")
    else:
        print("\n💡 Tip: Run with an image path to test on real images:")
        print("   python test_rembg_demo.py path/to/your/image.jpg")

    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
