#!/usr/bin/env python3
"""
Comprehensive preprocessing testing script for OCR project.

Tests doctr rcrop geometry, camscanner detection, orientation correction,
and provides systematic result logging and analysis.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Import preprocessing components
from ocr.datasets.preprocessing.pipeline import DocumentPreprocessor


class PreprocessingTester:
    """Comprehensive tester for preprocessing features."""

    def __init__(self, output_dir: Path, log_level: str = "INFO"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(self.output_dir / "preprocessing_test.log"), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

        # Test configurations
        self.test_configs = [
            {
                "name": "doctr_rcrop_with_fallback",
                "use_doctr_geometry": True,
                "document_detection_use_fallback_box": True,
                "enable_orientation_correction": False,
                "document_detection_use_camscanner": False,
            },
            {
                "name": "doctr_rcrop_without_fallback",
                "use_doctr_geometry": True,
                "document_detection_use_fallback_box": False,
                "enable_orientation_correction": False,
                "document_detection_use_camscanner": False,
            },
            {
                "name": "doctr_rcrop_with_orientation",
                "use_doctr_geometry": True,
                "document_detection_use_fallback_box": True,
                "enable_orientation_correction": True,
                "document_detection_use_camscanner": False,
            },
            {
                "name": "camscanner_detection",
                "use_doctr_geometry": False,
                "document_detection_use_fallback_box": True,
                "enable_orientation_correction": False,
                "document_detection_use_camscanner": True,
            },
            {
                "name": "doctr_text_detection",
                "use_doctr_geometry": False,
                "document_detection_use_fallback_box": True,
                "enable_orientation_correction": False,
                "document_detection_use_camscanner": False,
                "document_detection_use_doctr_text": True,
            },
        ]

        self.results: list[dict[str, Any]] = []

    def load_sample_images(self, image_dir: Path, max_samples: int = 10) -> list[np.ndarray]:
        """Load sample images for testing."""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = []

        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(f"**/*{ext}")))

        # Limit number of samples
        image_files = image_files[:max_samples]

        images = []
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                img_array = np.array(img)
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    images.append(img_array)
                    self.logger.info(f"Loaded image: {img_path.name}")
                else:
                    self.logger.warning(f"Skipping non-RGB image: {img_path.name}")
            except Exception as e:
                self.logger.error(f"Failed to load {img_path}: {e}")

        return images

    def draw_detection_overlay(self, image: np.ndarray, corners: np.ndarray | None, method: str | None) -> np.ndarray:
        """Draw detection overlay with highly visible markings."""
        overlay = image.copy()

        if corners is not None and len(corners) == 4:
            # Draw thick colored borders around the entire detected region
            # Create a mask for the detected document area
            mask = np.zeros_like(image, dtype=np.uint8)

            # Fill the quadrilateral with semi-transparent color
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], (0, 255, 0))  # Green fill
            overlay = cv2.addWeighted(overlay, 1.0, mask, 0.3, 0)  # Semi-transparent overlay

            # Draw thick white border
            cv2.polylines(overlay, [pts], True, (255, 255, 255), 8, cv2.LINE_AA)

            # Draw thick colored border
            cv2.polylines(overlay, [pts], True, (0, 255, 0), 4, cv2.LINE_AA)

            # Draw corner circles with numbers
            for i, corner in enumerate(corners):
                x, y = int(corner[0]), int(corner[1])
                # Draw large colored circle
                cv2.circle(overlay, (x, y), 20, (255, 0, 0), -1)  # Red filled circle
                cv2.circle(overlay, (x, y), 20, (255, 255, 255), 3)  # White border

                # Draw corner number
                cv2.putText(overlay, str(i), (x + 25, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(overlay, str(i), (x + 25, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)

            # Add method label with better visibility
            if method:
                # Draw semi-transparent background for text
                text = f"METHOD: {method.upper()}"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                cv2.rectangle(overlay, (8, 8), (8 + text_width + 10, 8 + text_height + 10), (0, 0, 0), -1)  # Black background
                cv2.putText(overlay, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        else:
            # No corners found - make this very obvious
            h, w = image.shape[:2]

            # Draw big red X across the image
            cv2.line(overlay, (0, 0), (w, h), (0, 0, 255), 8, cv2.LINE_AA)
            cv2.line(overlay, (w, 0), (0, h), (0, 0, 255), 8, cv2.LINE_AA)

            # Add clear "NO DOCUMENT DETECTED" message
            text = "NO DOCUMENT DETECTED"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)
            x = (w - text_width) // 2
            y = (h + text_height) // 2

            # Draw text background
            cv2.rectangle(overlay, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), (0, 0, 0), -1)

            cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        return overlay

    def test_preprocessing_config(self, config: dict[str, Any], images: list[np.ndarray]) -> dict[str, Any]:
        """Test a specific preprocessing configuration."""
        self.logger.info(f"Testing configuration: {config['name']}")

        # Create preprocessor
        preprocessor = DocumentPreprocessor(
            enable_document_detection=True,
            enable_perspective_correction=True,
            enable_enhancement=False,  # Disable for cleaner testing
            target_size=(640, 640),
            enable_final_resize=True,
            enable_orientation_correction=config.get("enable_orientation_correction", False),
            use_doctr_geometry=config.get("use_doctr_geometry", False),
            enable_padding_cleanup=False,
            document_detection_use_fallback_box=config.get("document_detection_use_fallback_box", True),
            document_detection_use_camscanner=config.get("document_detection_use_camscanner", False),
        )

        config_results = {
            "config_name": config["name"],
            "config": config,
            "samples": [],
            "summary": {
                "total_samples": len(images),
                "successful_detections": 0,
                "failed_detections": 0,
                "orientation_corrections": 0,
                "processing_times": [],
            },
        }

        for idx, image in enumerate(images):
            sample_result = {
                "sample_idx": idx,
                "original_shape": image.shape,
                "detection_success": False,
                "processing_time": 0.0,
                "metadata": {},
                "corners": None,
                "method": None,
            }

            try:
                start_time = time.time()
                result: dict[str, Any] = preprocessor(image)
                processing_time = time.time() - start_time

                sample_result["processing_time"] = processing_time
                sample_result["metadata"] = result["metadata"]
                sample_result["final_shape"] = result["image"].shape

                # Save processed image
                processed_filename = f"sample_{idx:03d}_{config['name']}_processed.jpg"
                processed_path = self.output_dir / "processed_images" / processed_filename
                processed_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(processed_path), cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR))
                sample_result["processed_image_path"] = str(processed_path)

                # Extract detection info
                metadata = result["metadata"]
                if "document_corners" in metadata:
                    sample_result["corners"] = metadata["document_corners"]
                    sample_result["method"] = metadata.get("document_detection_method")
                    sample_result["detection_success"] = True
                    config_results["summary"]["successful_detections"] += 1
                else:
                    config_results["summary"]["failed_detections"] += 1

                # Create and save overlay image with detected corners
                corners = sample_result.get("corners")
                method = sample_result.get("method")
                overlay = self.draw_detection_overlay(image, corners, method)
                overlay_filename = f"sample_{idx:03d}_{config['name']}_overlay.jpg"
                overlay_path = self.output_dir / "overlays" / overlay_filename
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                sample_result["overlay_image_path"] = str(overlay_path)

                if "orientation" in metadata:
                    config_results["summary"]["orientation_corrections"] += 1

                config_results["summary"]["processing_times"].append(processing_time)

                # Log detailed information
                self.logger.info(
                    f"Sample {idx}: {sample_result['method']} - "
                    f"Shape: {sample_result['original_shape']} -> {sample_result['final_shape']} - "
                    f"Time: {processing_time:.3f}s"
                )

                if sample_result["corners"] is not None:
                    self.logger.info(f"  Corners: {sample_result['corners'].tolist()}")

                if "orientation" in metadata:
                    orientation = metadata["orientation"]
                    self.logger.info(f"  Orientation: {orientation}")

            except Exception as e:
                self.logger.error(f"Sample {idx} failed: {e}")
                sample_result["error"] = str(e)
                config_results["summary"]["failed_detections"] += 1

            config_results["samples"].append(sample_result)

        # Calculate summary statistics
        times = config_results["summary"]["processing_times"]
        if times:
            config_results["summary"]["avg_processing_time"] = np.mean(times)
            config_results["summary"]["std_processing_time"] = np.std(times)
            config_results["summary"]["min_processing_time"] = np.min(times)
            config_results["summary"]["max_processing_time"] = np.max(times)

        config_results["summary"]["detection_success_rate"] = config_results["summary"]["successful_detections"] / len(images)

        return config_results

    def run_all_tests(self, images: list[np.ndarray]) -> None:
        """Run all test configurations."""
        self.logger.info(f"Starting comprehensive preprocessing tests with {len(images)} images")

        for config in self.test_configs:
            try:
                result = self.test_preprocessing_config(config, images)
                self.results.append(result)

                # Save individual config results
                output_file = self.output_dir / f"results_{config['name']}.json"
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2, default=str)

                self.logger.info(
                    f"Completed {config['name']}: {result['summary']['successful_detections']}/{len(images)} detections successful"
                )

            except Exception as e:
                self.logger.error(f"Configuration {config['name']} failed: {e}")

        # Save consolidated results
        consolidated_file = self.output_dir / "consolidated_results.json"
        with open(consolidated_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

    def generate_comparison_report(self) -> None:
        """Generate a comparison report across all configurations."""
        if not self.results:
            self.logger.warning("No results to compare")
            return

        report = {
            "test_summary": {
                "total_configurations": len(self.results),
                "total_samples_per_config": len(self.results[0]["samples"]) if self.results else 0,
            },
            "configuration_comparison": [],
        }

        for result in self.results:
            summary = result["summary"]
            config_summary = {
                "config_name": result["config_name"],
                "detection_success_rate": summary["detection_success_rate"],
                "avg_processing_time": summary.get("avg_processing_time", 0),
                "successful_detections": summary["successful_detections"],
                "orientation_corrections": summary["orientation_corrections"],
            }
            report["configuration_comparison"].append(config_summary)

        # Save comparison report
        report_file = self.output_dir / "comparison_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info("Generated comparison report")

    def create_visual_comparison(self, images: list[np.ndarray], max_visual_samples: int = 3) -> None:
        """Create visual comparison of results."""
        if not self.results:
            return

        visual_samples = images[:max_visual_samples]

        for sample_idx, image in enumerate(visual_samples):
            fig, axes = plt.subplots(2, len(self.results), figsize=(6 * len(self.results), 12))

            if len(self.results) == 1:
                axes = axes.reshape(2, -1)

            for config_idx, result in enumerate(self.results):
                sample_result = result["samples"][sample_idx]

                # Original image with overlay
                ax_orig = axes[0, config_idx]
                corners = sample_result.get("corners")
                method = sample_result.get("method")
                overlay = self.draw_detection_overlay(image, corners, method)
                ax_orig.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                ax_orig.set_title(f"{result['config_name']}\nDetection: {method or 'Failed'}")
                ax_orig.axis("off")

                # Processed image
                ax_proc = axes[1, config_idx]
                processed_img = cv2.imread(sample_result.get("processed_image_path", ""))
                if processed_img is not None:
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    ax_proc.imshow(processed_img)
                else:
                    ax_proc.text(
                        0.5,
                        0.5,
                        f"Processed Image\n{sample_result.get('final_shape', 'N/A')}",
                        ha="center",
                        va="center",
                        transform=ax_proc.transAxes,
                    )
                ax_proc.set_title("Processed Result")
                ax_proc.axis("off")

            plt.suptitle(f"Sample {sample_idx} - Preprocessing Comparison", fontsize=16)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"visual_comparison_sample_{sample_idx}.png", dpi=150, bbox_inches="tight")
            plt.close()

        self.logger.info(f"Created visual comparisons for {len(visual_samples)} samples")


def main():
    parser = argparse.ArgumentParser(description="Test preprocessing features systematically")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing test images")
    parser.add_argument("--output-dir", type=Path, default=Path("preprocessing_test_results"), help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum number of images to test")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")

    args = parser.parse_args()

    # Create tester
    tester = PreprocessingTester(args.output_dir, args.log_level)

    # Load images
    images = tester.load_sample_images(args.image_dir, args.max_samples)
    if not images:
        tester.logger.error("No valid images found")
        return

    # Run tests
    tester.run_all_tests(images)

    # Generate reports
    tester.generate_comparison_report()
    tester.create_visual_comparison(images)

    tester.logger.info(f"Testing complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
