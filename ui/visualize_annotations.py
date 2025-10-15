#!/usr/bin/env python3
"""
Visualize bounding boxes on processed images

This script loads the WebP images and their corresponding annotations,
then draws bounding boxes to show the detected text regions.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr.utils.orientation import normalize_pil_image, remap_polygons


def load_annotation(annotation_path: str) -> dict:
    """Load annotation JSON file"""
    with open(annotation_path, encoding="utf-8") as f:
        return json.load(f)


def _extract_polygons(annotation_payload: dict) -> list[dict[str, Any]]:
    """Return valid polygons alongside their source annotations."""
    polygons: list[dict[str, object]] = []
    for item in annotation_payload.get("annotations", []):
        polygon = item.get("polygon")
        if not polygon:
            continue
        coords = np.asarray(polygon, dtype=np.float32)
        if coords.size < 6:
            continue
        polygons.append({"coords": coords.reshape(-1, 2), "annotation": item})
    return polygons


def _normalize_image_and_polygons(image_path: str, annotations: dict) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Normalize the image via EXIF metadata and remap polygons accordingly."""
    pil_image = Image.open(image_path)
    normalized_image, orientation = normalize_pil_image(pil_image)

    if normalized_image.mode != "RGB":
        display_image = normalized_image.convert("RGB")
    else:
        display_image = normalized_image.copy()

    raw_width, raw_height = pil_image.size

    polygon_entries = _extract_polygons(annotations)
    if polygon_entries and orientation != 1:
        coords_list = [np.asarray(entry["coords"], dtype=np.float32) for entry in polygon_entries]
        transformed = remap_polygons(coords_list, raw_width, raw_height, orientation)
        for entry, remapped in zip(polygon_entries, transformed, strict=True):
            entry["coords"] = np.asarray(remapped, dtype=np.float32).reshape(-1, 2)

    # Close original images after copying to avoid resource leaks
    if normalized_image is not pil_image:
        pil_image.close()
        normalized_image.close()
    else:
        pil_image.close()

    return display_image, polygon_entries


def draw_bounding_boxes(image_path: str, annotations: dict, max_boxes: int = 20) -> Image.Image:
    """Draw polygon overlays aligned with EXIF-aware orientation."""
    image, polygon_entries = _normalize_image_and_polygons(image_path, annotations)
    draw = ImageDraw.Draw(image, "RGBA")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()  # type: ignore[assignment]

    colors = [
        (255, 0, 0),
        (0, 0, 255),
        (0, 128, 0),
        (255, 165, 0),
        (128, 0, 128),
        (165, 42, 42),
        (255, 192, 203),
        (128, 128, 128),
        (128, 128, 0),
        (0, 255, 255),
    ]

    for index, entry in enumerate(polygon_entries[:max_boxes]):
        color = colors[index % len(colors)]
        polygon = entry["coords"]
        annotation = entry["annotation"]
        points = [(float(x), float(y)) for x, y in np.asarray(polygon).reshape(-1, 2)]
        if len(points) < 3:
            continue

        outline = color + (255,)
        fill = color + (50,)
        draw.polygon(points, outline=outline, fill=fill)

        text = ""
        if isinstance(annotation, dict):
            text = annotation.get("text", "")
        if points:
            x_min = min(point[0] for point in points)
            y_min = min(point[1] for point in points)
            label = f"{text[:15]}{'...' if len(text) > 15 else ''}" if text else f"T{index + 1}"
            draw.text((x_min, y_min - 15), label, fill=outline, font=font)

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
