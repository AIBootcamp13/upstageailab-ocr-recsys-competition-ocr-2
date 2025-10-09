"""Testing framework for advanced document detection."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .advanced_detector import AdvancedDetectionConfig, AdvancedDocumentDetector


class AdvancedDetectorTester:
    """Testing framework for advanced document detection with ground truth validation."""

    def __init__(
        self,
        output_dir: str | Path,
        logger: logging.Logger | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)

        # Create subdirectories for different test types
        self.ground_truth_dir = self.output_dir / "ground_truth"
        self.results_dir = self.output_dir / "results"
        self.debug_dir = self.output_dir / "debug"

        for dir_path in [self.ground_truth_dir, self.results_dir, self.debug_dir]:
            dir_path.mkdir(exist_ok=True)

    def create_synthetic_test_dataset(self, num_samples: int = 50) -> list[dict[str, Any]]:
        """Create synthetic test dataset with known ground truth document boundaries."""
        test_cases = []

        for i in range(num_samples):
            # Generate random document parameters
            img_width = np.random.randint(800, 1200)
            img_height = np.random.randint(600, 900)

            # Create base image
            image = np.full((img_height, img_width, 3), 255, dtype=np.uint8)  # White background

            # Generate random document rectangle
            doc_width = np.random.randint(int(img_width * 0.6), int(img_width * 0.9))
            doc_height = np.random.randint(int(img_height * 0.5), int(img_height * 0.8))

            # Random position (ensure document fits)
            x_offset = np.random.randint(0, img_width - doc_width)
            y_offset = np.random.randint(0, img_height - doc_height)

            # Define ground truth corners
            gt_corners = np.array(
                [
                    [x_offset, y_offset],  # top-left
                    [x_offset + doc_width, y_offset],  # top-right
                    [x_offset + doc_width, y_offset + doc_height],  # bottom-right
                    [x_offset, y_offset + doc_height],  # bottom-left
                ],
                dtype=np.float32,
            )

            # Draw document rectangle
            cv2.rectangle(
                image,
                (x_offset, y_offset),
                (x_offset + doc_width, y_offset + doc_height),
                (200, 200, 200),  # Light gray
                -1,  # Filled
            )

            # Add some noise/texture to make detection more realistic
            noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
            image = cv2.add(image, noise).astype(np.uint8)

            # Add subtle border
            cv2.rectangle(
                image,
                (x_offset, y_offset),
                (x_offset + doc_width, y_offset + doc_height),
                (150, 150, 150),  # Darker gray border
                2,
            )

            # Apply slight perspective distortion (optional)
            if np.random.random() < 0.3:  # 30% chance
                image, gt_corners = self._apply_random_perspective(image, gt_corners)

            # Save test case
            case_id = "04d"
            image_path = self.ground_truth_dir / f"test_{case_id}.png"
            cv2.imwrite(str(image_path), image)

            test_case = {
                "id": case_id,
                "image_path": str(image_path),
                "ground_truth_corners": gt_corners.tolist(),
                "image_shape": image.shape,
                "document_area": doc_width * doc_height,
                "has_perspective_distortion": np.random.random() < 0.3,
            }

            test_cases.append(test_case)

        # Save ground truth metadata
        metadata_path = self.ground_truth_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(test_cases, f, indent=2)

        self.logger.info(f"Created {num_samples} synthetic test cases in {self.ground_truth_dir}")
        return test_cases

    def _apply_random_perspective(
        self,
        image: np.ndarray,
        corners: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply random perspective distortion to test case."""
        height, width = image.shape[:2]

        # Define destination points (slight distortion)
        dst_points = corners.copy()

        # Add small random distortions
        max_distortion = 20
        for i in range(4):
            dst_points[i][0] += np.random.randint(-max_distortion, max_distortion)
            dst_points[i][1] += np.random.randint(-max_distortion, max_distortion)

        # Ensure points stay within bounds
        dst_points = np.clip(dst_points, [0, 0], [width - 1, height - 1])

        # Apply perspective transform
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_points.astype(np.float32))
        warped = cv2.warpPerspective(image, M, (width, height))

        return warped, dst_points

    def run_individual_feature_tests(self) -> dict[str, Any]:
        """Run individual feature tests for ablation study."""
        self.logger.info("Running individual feature tests...")

        # Test configurations for ablation study
        test_configs = {
            "harris_only": AdvancedDetectionConfig(),
            "shi_tomasi_only": AdvancedDetectionConfig(),
            "combined_corners": AdvancedDetectionConfig(),
            "contour_refinement": AdvancedDetectionConfig(),
            "strict_geometric": AdvancedDetectionConfig(
                min_geometric_confidence=0.9,
                max_aspect_ratio_deviation=0.3,
            ),
            "relaxed_geometric": AdvancedDetectionConfig(
                min_geometric_confidence=0.6,
                max_aspect_ratio_deviation=0.8,
            ),
        }

        results = {}

        # Create test dataset if it doesn't exist
        if not (self.ground_truth_dir / "metadata.json").exists():
            test_cases = self.create_synthetic_test_dataset(20)
        else:
            with open(self.ground_truth_dir / "metadata.json") as f:
                test_cases = json.load(f)

        for config_name, config in test_configs.items():
            self.logger.info(f"Testing configuration: {config_name}")
            detector = AdvancedDocumentDetector(config=config, logger=self.logger)

            config_results = []
            for test_case in test_cases[:10]:  # Test on subset for speed
                result = self._test_single_case(detector, test_case)
                config_results.append(result)

            results[config_name] = {
                "config": config.__dict__,
                "results": config_results,
                "summary": self._summarize_results(config_results),
            }

        # Save results
        results_path = self.results_dir / "feature_ablation_results.json"
        with open(results_path, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._make_json_serializable(results)
            json.dump(json_results, f, indent=2)

        return results

    def run_ground_truth_validation(self) -> dict[str, Any]:
        """Run ground truth validation tests."""
        self.logger.info("Running ground truth validation...")

        # Load or create test dataset
        if not (self.ground_truth_dir / "metadata.json").exists():
            test_cases = self.create_synthetic_test_dataset(50)
        else:
            with open(self.ground_truth_dir / "metadata.json") as f:
                test_cases = json.load(f)

        # Test with default configuration
        detector = AdvancedDocumentDetector(logger=self.logger)

        validation_results = []
        for test_case in test_cases:
            result = self._test_single_case(detector, test_case)
            validation_results.append(result)

            # Save debug visualization
            self._save_debug_visualization(test_case, result)

        # Calculate metrics
        metrics = self._calculate_validation_metrics(validation_results, test_cases)

        results = {
            "test_cases": len(test_cases),
            "successful_detections": sum(1 for r in validation_results if r["detected"]),
            "average_confidence": np.mean([r["confidence"] for r in validation_results if r["detected"]]),
            "average_iou": np.mean([r["iou"] for r in validation_results if r["detected"]]),
            "metrics": metrics,
            "detailed_results": validation_results,
        }

        # Save validation results
        validation_path = self.results_dir / "ground_truth_validation.json"
        with open(validation_path, "w") as f:
            json_results = self._make_json_serializable(results)
            json.dump(json_results, f, indent=2)

        self.logger.info(f"Ground truth validation completed. Results saved to {validation_path}")
        return results

    def _test_single_case(
        self,
        detector: AdvancedDocumentDetector,
        test_case: dict[str, Any],
    ) -> dict[str, Any]:
        """Test detection on a single test case."""
        # Load image
        image = cv2.imread(test_case["image_path"])
        if image is None:
            return {"error": "Could not load image", "detected": False}

        # Run detection
        corners, method, metadata = detector.detect_document(image)

        detected = corners is not None
        confidence = metadata.get("confidence", 0.0) if detected else 0.0

        # Calculate IoU with ground truth
        iou = 0.0
        if detected and corners is not None:
            gt_corners = np.array(test_case["ground_truth_corners"])
            iou = self._calculate_iou(corners, gt_corners, (image.shape[0], image.shape[1]))

        result = {
            "test_case_id": test_case["id"],
            "detected": detected,
            "method": method,
            "confidence": confidence,
            "iou": iou,
            "metadata": metadata,
            "ground_truth_corners": test_case["ground_truth_corners"],
            "detected_corners": corners.tolist() if corners is not None else None,
        }

        return result

    def _calculate_iou(
        self,
        detected_corners: np.ndarray,
        gt_corners: np.ndarray,
        image_shape: tuple[int, int],
    ) -> float:
        """Calculate Intersection over Union (IoU) between detected and ground truth quadrilaterals."""
        # Create masks for both quadrilaterals
        mask1 = np.zeros(image_shape[:2], dtype=np.uint8)
        mask2 = np.zeros(image_shape[:2], dtype=np.uint8)

        cv2.fillPoly(mask1, [detected_corners.astype(np.int32)], (1,))
        cv2.fillPoly(mask2, [gt_corners.astype(np.int32)], (1,))

        # Calculate intersection and union
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        return intersection / union

    def _calculate_validation_metrics(
        self,
        results: list[dict[str, Any]],
        test_cases: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate comprehensive validation metrics."""
        detected_results = [r for r in results if r["detected"]]

        if not detected_results:
            return {"error": "No detections succeeded"}

        ious = [r["iou"] for r in detected_results]
        confidences = [r["confidence"] for r in detected_results]

        # IoU thresholds for precision/recall
        iou_thresholds = [0.5, 0.7, 0.9]

        metrics = {
            "total_test_cases": len(test_cases),
            "successful_detections": len(detected_results),
            "detection_rate": len(detected_results) / len(test_cases),
            "average_iou": np.mean(ious),
            "median_iou": np.median(ious),
            "average_confidence": np.mean(confidences),
            "iou_std": np.std(ious),
            "confidence_std": np.std(confidences),
        }

        # Precision at different IoU thresholds
        for threshold in iou_thresholds:
            correct = sum(1 for iou in ious if iou >= threshold)
            precision = correct / len(detected_results) if detected_results else 0.0
            metrics[f"precision_at_iou_{threshold}"] = precision

        return metrics

    def _summarize_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize results for a configuration."""
        detected = [r for r in results if r["detected"]]

        if not detected:
            return {"detection_rate": 0.0, "error": "No detections"}

        return {
            "detection_rate": len(detected) / len(results),
            "average_confidence": np.mean([r["confidence"] for r in detected]),
            "average_iou": np.mean([r["iou"] for r in detected]),
            "methods_used": {r["method"] for r in detected},
        }

    def _save_debug_visualization(
        self,
        test_case: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Save debug visualization for a test case."""
        image = cv2.imread(test_case["image_path"])
        if image is None:
            return

        # Draw ground truth
        gt_corners = np.array(test_case["ground_truth_corners"])
        cv2.polylines(image, [gt_corners.astype(np.int32)], True, (0, 255, 0), 2)  # Green for GT

        # Draw detection if successful
        if result["detected"] and result["detected_corners"]:
            det_corners = np.array(result["detected_corners"])
            cv2.polylines(image, [det_corners.astype(np.int32)], True, (255, 0, 0), 2)  # Red for detection

            # Add confidence text
            confidence_text = ".2f"
            cv2.putText(image, confidence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Save debug image
        debug_path = self.debug_dir / f"debug_{test_case['id']}.png"
        cv2.imwrite(str(debug_path), image)

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, int | float | str | bool | type(None)):
            return obj
        else:
            return str(obj)  # Fallback to string representation


# Convenience functions for running tests
def run_feature_ablation_test(output_dir: str | Path) -> dict[str, Any]:
    """Run feature ablation testing."""
    tester = AdvancedDetectorTester(output_dir)
    return tester.run_individual_feature_tests()


def run_ground_truth_validation_test(output_dir: str | Path) -> dict[str, Any]:
    """Run ground truth validation testing."""
    tester = AdvancedDetectorTester(output_dir)
    return tester.run_ground_truth_validation()


def create_test_dataset(output_dir: str | Path, num_samples: int = 50) -> list[dict[str, Any]]:
    """Create synthetic test dataset."""
    tester = AdvancedDetectorTester(output_dir)
    return tester.create_synthetic_test_dataset(num_samples)
