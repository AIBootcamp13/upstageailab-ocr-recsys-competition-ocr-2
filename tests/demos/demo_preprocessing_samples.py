#!/usr/bin/env python3
"""
Demo script to visualize offline preprocessing samples.

This script displays the generated samples with before/after comparisons
to demonstrate the Microsoft Lens-style preprocessing effects.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_sample_data(sample_dir: Path, sample_idx: int) -> dict:
    """Load sample data for a given index."""
    # Find the sample files
    original_files = list((sample_dir / "original").glob(f"sample_{sample_idx:03d}_*_original.jpg"))
    processed_files = list((sample_dir / "processed").glob(f"sample_{sample_idx:03d}_*_processed.jpg"))
    metadata_files = list((sample_dir / "processed").glob(f"sample_{sample_idx:03d}_*_metadata.json"))

    if not original_files or not processed_files or not metadata_files:
        raise FileNotFoundError(f"Sample {sample_idx} not found")

    # Load images
    original_img = Image.open(original_files[0])
    processed_img = Image.open(processed_files[0])

    # Load metadata
    with open(metadata_files[0]) as f:
        metadata = json.load(f)

    return {
        "original": np.array(original_img),
        "processed": np.array(processed_img),
        "metadata": metadata,
        "sample_name": original_files[0].stem.replace("_original", ""),
    }


def display_sample_comparison(sample_data: dict, figsize: tuple = (16, 8)):
    """Display a before/after comparison of a sample."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original image
    axes[0].imshow(sample_data["original"])
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Processed image
    axes[1].imshow(sample_data["processed"])
    axes[1].set_title("After Lens-Style Preprocessing", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # Processing information
    axes[2].axis("off")
    metadata = sample_data["metadata"]

    info_text = (
        ".1f"
        ".1f"
        f"""
Processing Applied:
{chr(10).join("• " + step.replace("_", " ").title() for step in metadata.get("processing_steps", []))}

Enhancements:
{chr(10).join("• " + enh.replace("_", " ").title() for enh in metadata.get("enhancement_applied", []))}

Original Shape: {metadata.get("original_shape", "N/A")}
Final Shape: {metadata.get("final_shape", "N/A")}
"""
    )

    axes[2].text(
        0.05,
        0.95,
        info_text,
        transform=axes[2].transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgray", "alpha": 0.8},
    )

    plt.suptitle(f"Sample: {sample_data['sample_name']}", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.show()


def display_multiple_samples(sample_dir: Path, num_samples: int = 5, cols: int = 3):
    """Display multiple samples in a grid layout."""
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows * 2, cols, figsize=(6 * cols, 4 * rows * 2))
    if rows == 1:
        axes = axes.reshape(2, -1)

    sample_idx = 0
    for row in range(rows):
        for col in range(cols):
            if sample_idx >= num_samples:
                axes[row * 2, col].axis("off")
                axes[row * 2 + 1, col].axis("off")
                continue

            try:
                sample_data = load_sample_data(sample_dir, sample_idx)

                # Original
                axes[row * 2, col].imshow(sample_data["original"])
                axes[row * 2, col].set_title(f"Sample {sample_idx}\nOriginal", fontsize=10)
                axes[row * 2, col].axis("off")

                # Processed
                axes[row * 2 + 1, col].imshow(sample_data["processed"])
                axes[row * 2 + 1, col].set_title("Processed", fontsize=10)
                axes[row * 2 + 1, col].axis("off")

            except FileNotFoundError:
                axes[row * 2, col].text(
                    0.5,
                    0.5,
                    f"Sample {sample_idx}\nNot Found",
                    ha="center",
                    va="center",
                    transform=axes[row * 2, col].transAxes,
                )
                axes[row * 2, col].axis("off")
                axes[row * 2 + 1, col].axis("off")

            sample_idx += 1

    plt.suptitle(
        "Microsoft Lens-Style Preprocessing Samples",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.show()


def display_generation_stats(sample_dir: Path):
    """Display generation statistics."""
    report_path = sample_dir / "generation_report.json"

    if not report_path.exists():
        print("❌ Generation report not found. Run sample generation first.")
        return

    with open(report_path) as f:
        report = json.load(f)

    print("\n" + "=" * 60)
    print("📊 SAMPLE GENERATION STATISTICS")
    print("=" * 60)

    summary = report["generation_summary"]
    print(f"Total Samples: {summary['total_samples']}")
    print(f"Successful: {summary['successful_samples']}")
    print(f"Failed: {summary['failed_samples']}")
    print(".1f")

    config = report["preprocessing_config"]
    print("\n🔧 Preprocessing Configuration:")
    print(f"   Document Detection: {config['enable_document_detection']}")
    print(f"   Perspective Correction: {config['enable_perspective_correction']}")
    print(f"   Image Enhancement: {config['enable_enhancement']}")
    print(f"   Text Enhancement: {config['enable_text_enhancement']}")
    print(f"   Target Size: {config['target_size']}")

    if "processing_steps_stats" in report and report["processing_steps_stats"]:
        print("\n📈 Processing Steps Applied:")
        for step, count in report["processing_steps_stats"].items():
            print(f"   {step.replace('_', ' ').title()}: {count} samples")

    print(f"\n📁 Output Directory: {sample_dir}")
    print("=" * 60)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Visualize offline preprocessing samples")
    parser.add_argument(
        "--sample-dir",
        default="outputs/samples",
        help="Directory containing generated samples",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "grid", "stats"],
        default="grid",
        help="Display mode",
    )
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index for single mode")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to display in grid mode",
    )
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in grid mode")

    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)

    if not sample_dir.exists():
        print(f"❌ Sample directory not found: {sample_dir}")
        print("Run 'python generate_offline_samples.py' first to generate samples.")
        return

    if args.mode == "stats":
        display_generation_stats(sample_dir)
    elif args.mode == "single":
        try:
            sample_data = load_sample_data(sample_dir, args.sample_idx)
            display_sample_comparison(sample_data)
        except FileNotFoundError:
            print(f"❌ Sample {args.sample_idx} not found")
    elif args.mode == "grid":
        display_multiple_samples(sample_dir, args.num_samples, args.cols)


if __name__ == "__main__":
    main()
