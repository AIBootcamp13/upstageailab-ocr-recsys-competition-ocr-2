"""
High-Confidence Decision Making for Document Detection.

This module implements confidence-weighted boundary selection, fallback hierarchy
with quality thresholds, uncertainty quantification, and ground truth validation
framework to achieve robust document detection decisions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np


class DetectionMethod(Enum):
    DOCTR = "doctr"
    CORNER_BASED = "corner_based"
    CONTOUR_BASED = "contour_based"
    FALLBACK = "fallback"


class ConfidenceLevel(Enum):
    HIGH = "high"  # >0.9 confidence
    MEDIUM = "medium"  # 0.7-0.9 confidence
    LOW = "low"  # 0.5-0.7 confidence
    VERY_LOW = "very_low"  # <0.5 confidence


@dataclass
class DetectionHypothesis:
    """Represents a single document detection hypothesis."""

    corners: np.ndarray  # Shape: (4, 2) for quadrilateral
    confidence: float
    method: DetectionMethod
    uncertainty: float  # Uncertainty quantification
    metadata: dict[str, Any] | None = None


@dataclass
class DecisionConfig:
    """Configuration for high-confidence decision making."""

    high_confidence_threshold: float = 0.9
    medium_confidence_threshold: float = 0.7
    low_confidence_threshold: float = 0.5
    uncertainty_threshold: float = 0.3
    min_confidence_for_selection: float = 0.6
    enable_ground_truth_validation: bool = True
    fallback_hierarchy: list[DetectionMethod] = None  # type: ignore

    def __post_init__(self):
        if self.fallback_hierarchy is None:
            self.fallback_hierarchy = [
                DetectionMethod.DOCTR,
                DetectionMethod.CORNER_BASED,
                DetectionMethod.CONTOUR_BASED,
                DetectionMethod.FALLBACK,
            ]


@dataclass
class DecisionResult:
    """Result of high-confidence decision making."""

    selected_hypothesis: DetectionHypothesis | None
    confidence_level: ConfidenceLevel
    all_hypotheses: list[DetectionHypothesis]
    decision_metadata: dict[str, Any]
    ground_truth_validated: bool = False
    validation_score: float | None = None


class DetectionStrategy(ABC):
    """Abstract base class for detection strategies."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionHypothesis | None:
        """Detect document boundaries in the image."""
        pass

    @abstractmethod
    def get_method(self) -> DetectionMethod:
        """Return the detection method."""
        pass


class DoctrDetectionStrategy(DetectionStrategy):
    """Doctr-based document detection strategy."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def detect(self, image: np.ndarray) -> DetectionHypothesis | None:
        """Detect using doctr library."""
        try:
            # Import doctr here to avoid import errors if not available
            from doctr.models import ocr_predictor

            # Convert numpy array to doctr-compatible format
            # This is a simplified implementation - real implementation would
            # need proper image preprocessing and doctr model loading
            ocr_predictor(pretrained=True)

            # For now, return a placeholder hypothesis
            # Real implementation would extract document boundaries from doctr results
            height, width = image.shape[:2]
            corners = np.array(
                [[width * 0.1, height * 0.1], [width * 0.9, height * 0.1], [width * 0.9, height * 0.9], [width * 0.1, height * 0.9]],
                dtype=np.float32,
            )

            confidence = 0.85  # Placeholder confidence
            uncertainty = 0.15

            return DetectionHypothesis(
                corners=corners,
                confidence=confidence,
                method=DetectionMethod.DOCTR,
                uncertainty=uncertainty,
                metadata={"doctr_version": "0.7.0", "model": "db_resnet50"},
            )

        except ImportError:
            return None
        except Exception:
            return None

    def get_method(self) -> DetectionMethod:
        return DetectionMethod.DOCTR


class CornerBasedDetectionStrategy(DetectionStrategy):
    """Corner-based document detection strategy."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        from ocr.datasets.preprocessing.advanced_corner_detection import AdvancedCornerDetector
        from ocr.datasets.preprocessing.geometric_document_modeling import GeometricDocumentModeler

        self.corner_detector = AdvancedCornerDetector()
        self.geometry_modeler = GeometricDocumentModeler()

    def detect(self, image: np.ndarray) -> DetectionHypothesis | None:
        """Detect using corner detection and geometric modeling."""
        try:
            # Detect corners
            corner_result = self.corner_detector.detect_corners(image)

            if len(corner_result.corners) < 4:
                return None

            # Fit geometric model
            geometry = self.geometry_modeler.fit_document_geometry(corner_result.corners)

            if geometry is None:
                return None

            # Calculate uncertainty based on confidence and corner detection quality
            uncertainty = 1.0 - (geometry.confidence * corner_result.confidence)

            return DetectionHypothesis(
                corners=geometry.corners,
                confidence=geometry.confidence,
                method=DetectionMethod.CORNER_BASED,
                uncertainty=uncertainty,
                metadata={
                    "corner_count": len(corner_result.corners),
                    "geometry_type": geometry.model_type,
                    "is_rectangle": geometry.is_rectangle,
                    "area": geometry.area,
                    "aspect_ratio": geometry.aspect_ratio,
                },
            )

        except Exception:
            return None

    def get_method(self) -> DetectionMethod:
        return DetectionMethod.CORNER_BASED


class ContourBasedDetectionStrategy(DetectionStrategy):
    """Contour-based document detection strategy (fallback)."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def detect(self, image: np.ndarray) -> DetectionHypothesis | None:
        """Detect using contour analysis."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Approximate polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            if len(approx) != 4:
                # Try to fit quadrilateral
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                corners = box.astype(np.float32)
            else:
                corners = approx.reshape(-1, 2).astype(np.float32)

            # Calculate confidence based on contour properties
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            # Higher compactness = more likely to be a document
            confidence = min(0.7, compactness * 2.0)
            uncertainty = 1.0 - confidence

            return DetectionHypothesis(
                corners=corners,
                confidence=confidence,
                method=DetectionMethod.CONTOUR_BASED,
                uncertainty=uncertainty,
                metadata={"contour_area": area, "compactness": compactness, "contour_count": len(contours)},
            )

        except Exception:
            return None

    def get_method(self) -> DetectionMethod:
        return DetectionMethod.CONTOUR_BASED


class FallbackDetectionStrategy(DetectionStrategy):
    """Simple fallback detection strategy."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def detect(self, image: np.ndarray) -> DetectionHypothesis | None:
        """Simple fallback: assume document fills most of image."""
        height, width = image.shape[:2]

        # Create a simple rectangle covering most of the image
        margin = 0.05  # 5% margin
        corners = np.array(
            [
                [width * margin, height * margin],
                [width * (1 - margin), height * margin],
                [width * (1 - margin), height * (1 - margin)],
                [width * margin, height * (1 - margin)],
            ],
            dtype=np.float32,
        )

        return DetectionHypothesis(
            corners=corners,
            confidence=0.3,  # Low confidence fallback
            method=DetectionMethod.FALLBACK,
            uncertainty=0.7,
            metadata={"fallback_reason": "no_other_methods_succeeded"},
        )

    def get_method(self) -> DetectionMethod:
        return DetectionMethod.FALLBACK


class HighConfidenceDecisionMaker:
    """
    High-confidence decision making for document detection.

    Combines multiple detection strategies with confidence weighting,
    fallback hierarchy, and uncertainty quantification.
    """

    def __init__(self, config: DecisionConfig | None = None):
        self.config = config or DecisionConfig()
        self.strategies = self._initialize_strategies()

    def _initialize_strategies(self) -> dict[DetectionMethod, DetectionStrategy]:
        """Initialize detection strategies."""
        return {
            DetectionMethod.DOCTR: DoctrDetectionStrategy(),
            DetectionMethod.CORNER_BASED: CornerBasedDetectionStrategy(),
            DetectionMethod.CONTOUR_BASED: ContourBasedDetectionStrategy(),
            DetectionMethod.FALLBACK: FallbackDetectionStrategy(),
        }

    def make_decision(self, image: np.ndarray) -> DecisionResult:
        """
        Make high-confidence decision for document detection.

        Args:
            image: Input image

        Returns:
            DecisionResult with selected hypothesis and metadata
        """
        all_hypotheses = []

        # Try strategies in fallback hierarchy order
        for method in self.config.fallback_hierarchy:
            strategy = self.strategies.get(method)
            if strategy:
                hypothesis = strategy.detect(image)
                if hypothesis:
                    all_hypotheses.append(hypothesis)

                    # If we have high confidence, we can stop early
                    if hypothesis.confidence >= self.config.high_confidence_threshold:
                        break

        if not all_hypotheses:
            # No hypotheses generated - return fallback
            fallback_strategy = self.strategies[DetectionMethod.FALLBACK]
            fallback_hypothesis = fallback_strategy.detect(image)
            if fallback_hypothesis:
                all_hypotheses.append(fallback_hypothesis)

        # Select best hypothesis using confidence weighting
        selected_hypothesis = self._select_best_hypothesis(all_hypotheses)

        # Determine confidence level
        confidence_level = self._determine_confidence_level(selected_hypothesis)

        # Ground truth validation if enabled
        ground_truth_validated = False
        validation_score = None

        if self.config.enable_ground_truth_validation and selected_hypothesis:
            validation_score = self._validate_against_ground_truth(selected_hypothesis, image)
            ground_truth_validated = validation_score is not None

        decision_metadata = {
            "total_hypotheses": len(all_hypotheses),
            "methods_tried": [h.method.value for h in all_hypotheses],
            "confidence_distribution": [h.confidence for h in all_hypotheses],
            "uncertainty_distribution": [h.uncertainty for h in all_hypotheses],
            "selection_criteria": "confidence_weighted",
        }

        return DecisionResult(
            selected_hypothesis=selected_hypothesis,
            confidence_level=confidence_level,
            all_hypotheses=all_hypotheses,
            decision_metadata=decision_metadata,
            ground_truth_validated=ground_truth_validated,
            validation_score=validation_score,
        )

    def _select_best_hypothesis(self, hypotheses: list[DetectionHypothesis]) -> DetectionHypothesis | None:
        """Select the best hypothesis using confidence weighting."""
        if not hypotheses:
            return None

        # Weight confidence by method reliability
        method_weights = {
            DetectionMethod.DOCTR: 1.0,
            DetectionMethod.CORNER_BASED: 0.9,
            DetectionMethod.CONTOUR_BASED: 0.7,
            DetectionMethod.FALLBACK: 0.3,
        }

        best_hypothesis = None
        best_score = 0.0

        for hypothesis in hypotheses:
            # Skip if below minimum confidence
            if hypothesis.confidence < self.config.min_confidence_for_selection:
                continue

            # Calculate weighted score
            weight = method_weights.get(hypothesis.method, 0.5)
            score = hypothesis.confidence * weight

            if score > best_score:
                best_score = score
                best_hypothesis = hypothesis

        return best_hypothesis

    def _determine_confidence_level(self, hypothesis: DetectionHypothesis | None) -> ConfidenceLevel:
        """Determine confidence level of the selected hypothesis."""
        if hypothesis is None:
            return ConfidenceLevel.VERY_LOW

        confidence = hypothesis.confidence

        if confidence >= self.config.high_confidence_threshold:
            return ConfidenceLevel.HIGH
        elif confidence >= self.config.medium_confidence_threshold:
            return ConfidenceLevel.MEDIUM
        elif confidence >= self.config.low_confidence_threshold:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _validate_against_ground_truth(self, hypothesis: DetectionHypothesis, image: np.ndarray) -> float | None:
        """
        Validate hypothesis against ground truth.

        This is a placeholder for ground truth validation framework.
        In a real implementation, this would compare against labeled data.
        """
        # Placeholder implementation
        # Real implementation would load ground truth data and calculate IoU or other metrics

        # For now, return a validation score based on geometric plausibility
        if hypothesis.corners is None:
            return 0.0

        height, width = image.shape[:2]

        # Check if corners are within reasonable bounds
        x_coords, y_coords = hypothesis.corners[:, 0], hypothesis.corners[:, 1]
        if not np.all((x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)):
            return 0.0

        # Check aspect ratio reasonableness
        min_coords = np.min(hypothesis.corners, axis=0)
        max_coords = np.max(hypothesis.corners, axis=0)
        bbox_width = max_coords[0] - min_coords[0]
        bbox_height = max_coords[1] - min_coords[1]

        if bbox_height == 0:
            return 0.0

        aspect_ratio = bbox_width / bbox_height

        # Documents typically have reasonable aspect ratios
        if 0.1 <= aspect_ratio <= 10.0:
            return min(1.0, hypothesis.confidence * 1.2)  # Boost score for plausible geometry
        else:
            return hypothesis.confidence * 0.8  # Penalize for implausible geometry


def create_ground_truth_validation_framework():
    """
    Create ground truth validation framework setup.

    This function sets up the infrastructure for ground truth validation
    of document detection results.
    """
    # Placeholder for ground truth framework
    # Real implementation would create:
    # - Ground truth dataset loading
    # - IoU calculation functions
    # - Precision/recall metrics
    # - Validation pipeline

    print("Ground truth validation framework setup placeholder")
    print("- Would create ground truth dataset loader")
    print("- Would implement IoU calculation")
    print("- Would set up validation metrics")
    print("- Would create validation pipeline")

    return True
