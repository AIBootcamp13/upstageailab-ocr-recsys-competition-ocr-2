#!/usr/bin/env python3
"""
Offline Sample Generation Script

This script generates processed sample images using the Microsoft Lens-style
preprocessing pipeline for demonstration and testing purposes.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from ocr.datasets.preprocessing import DocumentPreprocessor
from ocr.utils.logging import logger


class OfflineSampleGenerator:
    """Generate offline samples with preprocessing applied."""

    def __init__(
        self,
        source_dir: str = "data/datasets/images",
        output_dir: str = "outputs/samples",
        num_samples: int = 10,
        preprocessing_config: dict | None = None,
    ):
        """
        Initialize the sample generator.

        Args:
            source_dir: Directory containing source images
            output_dir: Directory to save processed samples
            num_samples: Number of samples to generate
            preprocessing_config: Configuration for preprocessing pipeline
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples

        # Default preprocessing configuration
        default_config = {
            "enable_document_detection": True,
            "enable_perspective_correction": True,
            "enable_enhancement": True,
            "enable_text_enhancement": True,
            "target_size": (640, 640),
        }

        self.preprocessing_config = {**default_config, **(preprocessing_config or {})}
        self.preprocessor = DocumentPreprocessor(**self.preprocessing_config)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "original").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)
        (self.output_dir / "comparison").mkdir(exist_ok=True)

        logger.info(f"Initialized sample generator with {num_samples} samples")
        logger.info(f"Source: {source_dir}")
        logger.info(f"Output: {output_dir}")

    def collect_sample_images(self) -> list[Path]:
        """Collect sample images from the dataset."""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        # Collect all image files
        all_images = []
        for ext in image_extensions:
            all_images.extend(self.source_dir.rglob(f"*{ext}"))
            all_images.extend(self.source_dir.rglob(f"*{ext.upper()}"))

        if not all_images:
            raise ValueError(f"No images found in {self.source_dir}")

        # Randomly select samples
        if len(all_images) <= self.num_samples:
            selected_images = all_images
        else:
            selected_images = random.sample(all_images, self.num_samples)

        logger.info(f"Selected {len(selected_images)} images out of {len(all_images)} available")
        return selected_images

    def process_sample(self, image_path: Path) -> dict:
        """
        Process a single sample image.

        Args:
            image_path: Path to the source image

        Returns:
            Processing results and metadata
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            original_array = np.array(image)

            # Apply preprocessing
            result = self.preprocessor(original_array)

            return {
                "image_path": image_path,
                "original_image": original_array,
                "processed_image": result["image"],
                "metadata": result["metadata"],
                "success": True,
            }

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            return {"image_path": image_path, "error": str(e), "success": False}

    def save_sample(self, sample_data: dict, sample_idx: int):
        """Save a processed sample to disk."""
        if not sample_data["success"]:
            logger.warning(f"Skipping failed sample {sample_idx}")
            return

        base_name = f"sample_{sample_idx:03d}_{sample_data['image_path'].stem}"

        # Save original image
        original_path = self.output_dir / "original" / f"{base_name}_original.jpg"
        Image.fromarray(sample_data["original_image"]).save(original_path)

        # Save processed image
        processed_path = self.output_dir / "processed" / f"{base_name}_processed.jpg"
        Image.fromarray(sample_data["processed_image"]).save(processed_path)

        # Save metadata
        metadata_path = self.output_dir / "processed" / f"{base_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(sample_data["metadata"], f, indent=2, default=str)

        # Create comparison visualization
        self._create_comparison_visualization(
            sample_data["original_image"],
            sample_data["processed_image"],
            sample_data["metadata"],
            self.output_dir / "comparison" / f"{base_name}_comparison.jpg",
        )

    def _create_comparison_visualization(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        metadata: dict,
        output_path: Path,
    ):
        """Create a side-by-side comparison visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Original image
        axes[0].imshow(original)
        axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        # Processed image
        axes[1].imshow(processed)
        axes[1].set_title("After Lens-Style Preprocessing", fontsize=14, fontweight="bold")
        axes[1].axis("off")

        # Add processing info
        if processing_steps := metadata.get("processing_steps", []):
            info_text = f"Applied: {', '.join(processing_steps)}"
            fig.suptitle(info_text, fontsize=12, y=0.98)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def generate_samples(self):
        """Generate all offline samples."""
        logger.info("Starting offline sample generation...")

        # Collect sample images
        sample_images = self.collect_sample_images()

        # Process each sample
        processed_samples = []
        for i, image_path in enumerate(tqdm(sample_images, desc="Processing samples")):
            sample_data = self.process_sample(image_path)
            processed_samples.append(sample_data)

            # Save sample
            self.save_sample(sample_data, i)

        # Generate summary report
        self._generate_summary_report(processed_samples)

        logger.info(f"Generated {len(processed_samples)} samples in {self.output_dir}")

    def _generate_summary_report(self, samples: list[dict]):
        """Generate a summary report of the sample generation."""
        successful_samples = [s for s in samples if s["success"]]
        failed_samples = [s for s in samples if not s["success"]]

        report = {
            "generation_summary": {
                "total_samples": len(samples),
                "successful_samples": len(successful_samples),
                "failed_samples": len(failed_samples),
                "success_rate": (len(successful_samples) / len(samples) if samples else 0),
            },
            "preprocessing_config": self.preprocessing_config,
            "processing_steps_stats": {},
        }

        # Collect processing steps statistics
        if successful_samples:
            all_steps = {}
            for sample in successful_samples:
                steps = sample["metadata"].get("processing_steps", [])
                for step in steps:
                    all_steps[step] = all_steps.get(step, 0) + 1

            report["processing_steps_stats"] = all_steps

        # Save report
        report_path = self.output_dir / "generation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Print summary
        print("\n" + "=" * 60)
        print("📊 OFFLINE SAMPLE GENERATION REPORT")
        print("=" * 60)
        print(f"Total Samples: {report['generation_summary']['total_samples']}")
        print(f"Successful: {report['generation_summary']['successful_samples']}")
        print(f"Failed: {report['generation_summary']['failed_samples']}")
        print(".1f")

        if report["processing_steps_stats"]:
            print("\n🔧 Processing Steps Applied:")
            for step, count in report["processing_steps_stats"].items():
                print(f"   {step}: {count} samples")

        print(f"\n📁 Output Directory: {self.output_dir}")
        print("=" * 60)


def main():
    """Main entry point for offline sample generation."""
    parser = argparse.ArgumentParser(description="Generate offline preprocessing samples")
    parser.add_argument(
        "--source-dir",
        default="data/datasets/images",
        help="Directory containing source images",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/samples",
        help="Directory to save processed samples",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument(
        "--no-document-detection",
        action="store_true",
        help="Disable document boundary detection",
    )
    parser.add_argument(
        "--no-perspective-correction",
        action="store_true",
        help="Disable perspective correction",
    )
    parser.add_argument("--no-enhancement", action="store_true", help="Disable image enhancement")
    parser.add_argument(
        "--no-text-enhancement",
        action="store_true",
        help="Disable text-specific enhancement",
    )

    args = parser.parse_args()

    # Configure preprocessing
    preprocessing_config = {
        "enable_document_detection": not args.no_document_detection,
        "enable_perspective_correction": not args.no_perspective_correction,
        "enable_enhancement": not args.no_enhancement,
        "enable_text_enhancement": not args.no_text_enhancement,
    }

    # Generate samples
    generator = OfflineSampleGenerator(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        preprocessing_config=preprocessing_config,
    )

    generator.generate_samples()


if __name__ == "__main__":
    main()
