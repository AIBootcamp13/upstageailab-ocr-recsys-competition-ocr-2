#!/usr/bin/env python3
"""
OCR Evaluation Viewer Demo

This script demonstrates the evaluation viewer functionality
by loading and analyzing prediction data.
"""

import pandas as pd

# Setup project paths automatically
from ocr.utils.path_utils import get_outputs_path, setup_project_paths

setup_project_paths()

from ui.utils.config_parser import ConfigParser


def main():
    """Run the evaluation viewer demo."""
    print("🔍 OCR Evaluation Viewer Demo")
    print("=" * 50)

    # Initialize config parser
    ConfigParser()

    # Load prediction data
    predictions_path = get_outputs_path() / "predictions" / "submission.csv"
    if not predictions_path.exists():
        print("❌ Prediction file not found. Please run a prediction first.")
        return

    print("📊 Loading prediction data...")
    df = pd.read_csv(predictions_path)

    print(f"✅ Loaded {len(df)} images with predictions")

    # Dataset overview
    print("\n📈 Dataset Overview:")
    print(f"   Total Images: {len(df)}")

    total_polygons = sum(len(str(row["polygons"]).split("|")) if pd.notna(row["polygons"]) else 0 for _, row in df.iterrows())
    print(f"   Total Predictions: {total_polygons}")

    avg_polygons = total_polygons / len(df) if len(df) > 0 else 0
    print(f"   Average Predictions/Image: {avg_polygons:.1f}")

    empty_predictions = len([row for _, row in df.iterrows() if pd.isna(row["polygons"]) or not str(row["polygons"]).strip()])
    print(f"   Images with No Predictions: {empty_predictions}")

    # Prediction analysis
    print("\n🎯 Prediction Analysis:")

    all_polygons = []
    for _, row in df.iterrows():
        if pd.notna(row["polygons"]) and str(row["polygons"]).strip():
            polygons = str(row["polygons"]).split("|")
            for polygon in polygons:
                coords = [float(x) for x in polygon.split()]
                if len(coords) >= 8:  # At least 4 points
                    all_polygons.append(coords)

    if all_polygons:
        print(f"   Valid Polygons: {len(all_polygons)}")

        # Calculate statistics
        areas = []
        aspect_ratios = []
        for coords in all_polygons:
            xs = coords[0::2]
            ys = coords[1::2]
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            area = width * height
            aspect_ratio = width / height if height > 0 else 0

            areas.append(area)
            aspect_ratios.append(aspect_ratio)

        print(f"   Average Bounding Box Area: {sum(areas) / len(areas):.0f} pixels²")
        print(f"   Median Bounding Box Area: {sorted(areas)[len(areas) // 2]:.0f} pixels²")
        print(f"   Average Aspect Ratio: {sum(aspect_ratios) / len(aspect_ratios):.2f}")
        print(f"   Median Aspect Ratio: {sorted(aspect_ratios)[len(aspect_ratios) // 2]:.2f}")
    else:
        print("   No valid polygons found")

    # Sample predictions
    print("\n🖼️ Sample Predictions:")
    for _, row in df.head(3).iterrows():
        filename = row["filename"]
        polygons_str = str(row["polygons"]) if pd.notna(row["polygons"]) else ""
        if polygons_str.strip():
            polygon_count = len(polygons_str.split("|"))
            print(f"   {filename}: {polygon_count} text regions")
        else:
            print(f"   {filename}: No predictions")

    print("\n✨ Demo completed! Run 'python run_ui.py evaluation_viewer' to launch the interactive UI.")
    print("\nThe evaluation viewer provides:")
    print("• 📊 Dataset statistics and distributions")
    print("• 🎯 Prediction analysis and metrics")
    print("• 🖼️ Interactive image viewer with bounding boxes")
    print("• 📈 Charts and visualizations")
    print("• 🔍 Filtering and sorting capabilities")


if __name__ == "__main__":
    main()
