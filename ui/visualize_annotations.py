#!/usr/bin/env python3
"""
Visualize bounding boxes on processed images

This script loads the WebP images and their corresponding annotations,
then draws bounding boxes to show the detected text regions.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def load_annotation(annotation_path: str) -> dict:
    """Load annotation JSON file"""
    with open(annotation_path, encoding="utf-8") as f:
        return json.load(f)


def draw_bounding_boxes(image_path: str, annotations: dict, max_boxes: int = 20) -> Image.Image:
    """Draw bounding boxes on image"""
    # Load image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()

    # Draw bounding boxes
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

    for i, annotation in enumerate(annotations["annotations"][:max_boxes]):
        polygon = annotation["polygon"]
        text = annotation["text"]

        # Convert polygon to bounding box (rectangle)
        if len(polygon) >= 4:
            x_coords = [point[0] for point in polygon]
            y_coords = [point[1] for point in polygon]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Draw rectangle
            color = colors[i % len(colors)]
            draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=2)

            # Draw text label
            label = f"{text[:15]}{'...' if len(text) > 15 else ''}"
            draw.text((x_min, y_min - 15), label, fill=color, font=font)

    return image


def visualize_sample_images(
    base_dir: str = "outputs/upstage_processed", dataset: str = "cord-v2", max_images: int = 2, save_path: str | None = None
):
    """Visualize sample images with bounding boxes"""

    dataset_dir = Path(base_dir) / dataset
    images_dir = dataset_dir / "images_webp"
    annotations_dir = dataset_dir / "annotations"

    if not images_dir.exists() or not annotations_dir.exists():
        print(f"Directories not found: {images_dir} or {annotations_dir}")
        return

    # Get available annotation files first
    annotation_files = list(annotations_dir.glob("*.json"))
    if not annotation_files:
        print("No annotation files found")
        return

    # Find corresponding images
    image_files = []
    for ann_file in annotation_files[:max_images]:
        image_file = images_dir / f"{ann_file.stem}.webp"
        if image_file.exists():
            image_files.append((image_file, ann_file))

    if not image_files:
        print("No matching image files found for annotations")
        return

    print(f"Found {len(image_files)} image-annotation pairs to visualize")

    # Create subplots
    fig, axes = plt.subplots(1, len(image_files), figsize=(15, 8))
    if len(image_files) == 1:
        axes = [axes]

    for i, (image_file, annotation_file) in enumerate(image_files):
        # Load annotation
        annotations = load_annotation(str(annotation_file))

        # Draw bounding boxes
        image_with_boxes = draw_bounding_boxes(str(image_file), annotations)

        # Display
        axes[i].imshow(image_with_boxes)
        axes[i].set_title(f"Image: {image_file.name}\n{len(annotations['annotations'])} text regions")
        axes[i].axis("off")

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


def show_annotation_stats(base_dir: str = "outputs/upstage_processed", dataset: str = "cord-v2"):
    """Show statistics about annotations"""

    dataset_dir = Path(base_dir) / dataset
    annotations_dir = dataset_dir / "annotations"

    if not annotations_dir.exists():
        print(f"Annotations directory not found: {annotations_dir}")
        return

    annotation_files = list(annotations_dir.glob("*.json"))

    print(f"\n📊 Annotation Statistics for {dataset}:")
    print(f"Total annotation files: {len(annotation_files)}")

    if annotation_files:
        # Load first annotation to show sample
        sample_annotation = load_annotation(str(annotation_files[0]))
        num_annotations = len(sample_annotation["annotations"])

        print(f"Sample file has {num_annotations} text regions")

        # Show first few annotations
        print("\n📝 Sample annotations:")
        for i, ann in enumerate(sample_annotation["annotations"][:5]):
            text = ann["text"][:30] + "..." if len(ann["text"]) > 30 else ann["text"]
            bbox = ann["polygon"]
            print(f"  {i + 1}. '{text}' - bbox: ({bbox[0][0]:.0f},{bbox[0][1]:.0f}) to ({bbox[2][0]:.0f},{bbox[2][1]:.0f})")


if __name__ == "__main__":
    import sys

    # Check if we're in the right directory
    if not Path("outputs/upstage_processed").exists():
        print("Please run this script from the project root directory")
        sys.exit(1)

    # Show statistics
    show_annotation_stats()

    # Visualize images
    print("\n🖼️  Visualizing sample images with bounding boxes...")
    save_file = "sample_bounding_boxes.png"
    visualize_sample_images(save_path=save_file)
    print(f"Visualization saved as: {save_file}")
