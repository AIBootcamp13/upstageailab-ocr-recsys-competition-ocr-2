#!/usr/bin/env python3
"""
Sample 20 test images and apply receipt filters and shadow removal filters.
Creates before/after comparisons for visual inspection.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# Import the filter functions from advanced_receipt_filters.py
def apply_receipt_filters(gray):
    """Optimized filter chain for receipt text detection."""
    # 1. CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Bilateral filter to reduce noise while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # 3. Morphological operations to clean up text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    # 4. Sharpening to enhance text edges
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    return enhanced


def apply_shadow_removal_filters(gray):
    """Filters to remove shadows and improve text visibility."""
    # Normalize lighting using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.medianBlur(background, 5)

    # Divide gray by background to normalize
    normalized = cv2.divide(gray, background, scale=255)

    # CLAHE on normalized image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    return enhanced


def create_comparison_image(original, filtered, title_original, title_filtered):
    """Create a side-by-side comparison image."""
    # Ensure both images are the same height
    h1, w1 = original.shape[:2]
    h2, w2 = filtered.shape[:2]

    # Resize if necessary to match heights
    if h1 != h2:
        if h1 > h2:
            filtered = cv2.resize(filtered, (int(w2 * h1 / h2), h1))
        else:
            original = cv2.resize(original, (int(w1 * h2 / h1), h2))

    # Concatenate horizontally
    comparison = cv2.hconcat([original, filtered])

    # Add titles
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    color = (255, 255, 255)  # White text

    # Add title for original
    cv2.putText(comparison, title_original, (10, 30), font, font_scale, color, font_thickness)

    # Add title for filtered (positioned on the right side)
    text_x = original.shape[1] + 10
    cv2.putText(comparison, title_filtered, (text_x, 30), font, font_scale, color, font_thickness)

    return comparison


def process_sample_images(input_dir, output_dir, num_samples=20, specific_files=None):
    """Process sample images with both filter types."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directories
    receipt_output = output_path / "receipt_filters"
    shadow_output = output_path / "shadow_removal_filters"
    comparison_output = output_path / "comparisons"

    receipt_output.mkdir(parents=True, exist_ok=True)
    shadow_output.mkdir(parents=True, exist_ok=True)
    comparison_output.mkdir(parents=True, exist_ok=True)

    # Get image files
    if specific_files:
        # Use specific files provided
        image_files = [input_path / filename for filename in specific_files]
        # Filter to only existing files
        image_files = [f for f in image_files if f.exists()]
        print(f"Processing {len(image_files)} specific images...")
    else:
        # Get all image files and sample
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

        # Sample images
        if len(image_files) > num_samples:
            sampled_files = random.sample(image_files, num_samples)
        else:
            sampled_files = image_files

        image_files = sampled_files
        print(f"Processing {len(image_files)} sample images...")

    for image_file in tqdm(image_files, desc="Applying filters"):
        try:
            # Read image
            image = cv2.imread(str(image_file))
            if image is None:
                continue

            # Convert to grayscale for processing
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Apply receipt filters
            receipt_filtered = apply_receipt_filters(gray)

            # Apply shadow removal filters
            shadow_filtered = apply_shadow_removal_filters(gray)

            # Save individual results
            base_name = image_file.stem

            # Save receipt filter result
            receipt_filename = f"{base_name}_receipt_filtered.jpg"
            cv2.imwrite(str(receipt_output / receipt_filename), receipt_filtered)

            # Save shadow removal result
            shadow_filename = f"{base_name}_shadow_filtered.jpg"
            cv2.imwrite(str(shadow_output / shadow_filename), shadow_filtered)

            # Create and save comparison images
            receipt_comparison = create_comparison_image(gray, receipt_filtered, "Original", "Receipt Filters")
            shadow_comparison = create_comparison_image(gray, shadow_filtered, "Original", "Shadow Removal")

            cv2.imwrite(str(comparison_output / f"{base_name}_receipt_comparison.jpg"), receipt_comparison)
            cv2.imwrite(str(comparison_output / f"{base_name}_shadow_comparison.jpg"), shadow_comparison)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    # Create an index HTML file for easy viewing
    create_index_html(output_path, len(image_files))

    print("\nFilter comparison completed!")
    print("Individual results saved to:")
    print(f"  - Receipt filters: {receipt_output}")
    print(f"  - Shadow removal: {shadow_output}")
    print(f"  - Comparisons: {comparison_output}")
    print(f"  - Index: {output_path}/index.html")


def create_index_html(output_dir, num_images):
    """Create an HTML index file for viewing results."""
    # Get comparison images
    receipt_comparisons = []
    shadow_comparisons = []

    comparisons_dir = output_dir / "comparisons"
    if comparisons_dir.exists():
        for img_file in sorted(comparisons_dir.glob("*_receipt_comparison.jpg")):
            receipt_comparisons.append(img_file.name)
        for img_file in sorted(comparisons_dir.glob("*_shadow_comparison.jpg")):
            shadow_comparisons.append(img_file.name)

    # Create HTML content
    receipt_html = ""
    for img in receipt_comparisons:
        receipt_html += f"""
        <div class="image-item">
            <h3>{img.replace("_receipt_comparison.jpg", "")}</h3>
            <img src="comparisons/{img}" alt="{img}">
        </div>"""

    shadow_html = ""
    for img in shadow_comparisons:
        shadow_html += f"""
        <div class="image-item">
            <h3>{img.replace("_shadow_comparison.jpg", "")}</h3>
            <img src="comparisons/{img}" alt="{img}">
        </div>"""

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Advanced Receipt Filters - Sample Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .section {{ margin-bottom: 40px; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        .image-item {{ border: 1px solid #ddd; padding: 10px; }}
        .image-item img {{ max-width: 100%; height: auto; }}
        .image-item h3 {{ margin-top: 0; color: #333; }}
        h1, h2 {{ color: #2c3e50; }}
    </style>
</head>
<body>
    <h1>Advanced Receipt Filters - Sample Results ({num_images} images)</h1>

    <div class="section">
        <h2>Receipt Optimized Filters</h2>
        <p>CLAHE + Bilateral Filter + Morphological Operations + Sharpening</p>
        <div class="image-grid">
            {receipt_html}
        </div>
    </div>

    <div class="section">
        <h2>Shadow Removal Filters</h2>
        <p>Morphological lighting normalization + CLAHE</p>
        <div class="image-grid">
            {shadow_html}
        </div>
    </div>
</body>
</html>
"""

    with open(output_dir / "index.html", "w") as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description="Sample images with advanced receipt filters")
    parser.add_argument(
        "--input_dir",
        default="/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/images/test",
        help="Input directory with test images",
    )
    parser.add_argument("--output_dir", default="filter_samples", help="Output directory for filtered results")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of images to sample")
    parser.add_argument("--specific_files", nargs="*", help="Specific image filenames to process (overrides num_samples)")

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist")
        return 1

    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    process_sample_images(input_path, output_path, args.num_samples, args.specific_files)
    return 0


if __name__ == "__main__":
    exit(main())
