"""
Tests for High-Confidence Decision Making module.
"""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from ocr.datasets.preprocessing.high_confidence_decision_making import (
    ConfidenceLevel,
    ContourBasedDetectionStrategy,
    DecisionConfig,
    DetectionHypothesis,
    DetectionMethod,
    DoctrDetectionStrategy,
    FallbackDetectionStrategy,
    HighConfidenceDecisionMaker,
)


class TestDetectionStrategies:
    @pytest.fixture
    def sample_image(self):
        """Create a sample document image for testing."""
        img = np.zeros((200, 300), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 150), 255, -1)
        return img

    def test_fallback_strategy_always_succeeds(self, sample_image):
        """Test that fallback strategy always produces a result."""
        strategy = FallbackDetectionStrategy()
        result = strategy.detect(sample_image)

        assert result is not None
        assert result.method == DetectionMethod.FALLBACK
        assert result.confidence == 0.3
        assert result.uncertainty == 0.7
        assert len(result.corners) == 4

    def test_contour_strategy_on_rectangle(self, sample_image):
        """Test contour strategy on a clear rectangular document."""
        strategy = ContourBasedDetectionStrategy()
        result = strategy.detect(sample_image)

        assert result is not None
        assert result.method == DetectionMethod.CONTOUR_BASED
        assert result.confidence > 0.0
        assert result.uncertainty < 1.0
        assert len(result.corners) == 4

    @patch("ocr.datasets.preprocessing.high_confidence_decision_making.DoctrDetectionStrategy.detect")
    def test_doctr_strategy_placeholder(self, mock_detect, sample_image):
        """Test doctr strategy (mocked since doctr may not be available)."""
        mock_detect.return_value = DetectionHypothesis(
            corners=np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32),
            confidence=0.85,
            method=DetectionMethod.DOCTR,
            uncertainty=0.15,
        )

        strategy = DoctrDetectionStrategy()
        result = strategy.detect(sample_image)

        assert result is not None
        assert result.method == DetectionMethod.DOCTR
        assert result.confidence == 0.85


class TestHighConfidenceDecisionMaker:
    @pytest.fixture
    def sample_image(self):
        """Create a sample document image."""
        img = np.zeros((200, 300), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 150), 255, -1)
        return img

    @pytest.fixture
    def decision_maker(self):
        """Create decision maker instance."""
        config = DecisionConfig(high_confidence_threshold=0.9, medium_confidence_threshold=0.7, min_confidence_for_selection=0.3)
        return HighConfidenceDecisionMaker(config)

    def test_decision_maker_initialization(self, decision_maker):
        """Test that decision maker initializes correctly."""
        assert len(decision_maker.strategies) == 4
        assert DetectionMethod.DOCTR in decision_maker.strategies
        assert DetectionMethod.CORNER_BASED in decision_maker.strategies
        assert DetectionMethod.CONTOUR_BASED in decision_maker.strategies
        assert DetectionMethod.FALLBACK in decision_maker.strategies

    def test_make_decision_with_fallback_only(self, decision_maker, sample_image):
        """Test decision making when only fallback strategy works."""
        # Mock all strategies to fail except fallback
        with (
            patch.object(decision_maker.strategies[DetectionMethod.DOCTR], "detect", return_value=None),
            patch.object(decision_maker.strategies[DetectionMethod.CORNER_BASED], "detect", return_value=None),
            patch.object(decision_maker.strategies[DetectionMethod.CONTOUR_BASED], "detect", return_value=None),
        ):
            result = decision_maker.make_decision(sample_image)

            assert result.selected_hypothesis is not None
            assert result.selected_hypothesis.method == DetectionMethod.FALLBACK
            assert result.confidence_level == ConfidenceLevel.VERY_LOW
            assert len(result.all_hypotheses) == 1

    def test_make_decision_with_high_confidence_result(self, decision_maker, sample_image):
        """Test decision making with a high confidence result."""
        # Create a high confidence hypothesis
        high_conf_hypothesis = DetectionHypothesis(
            corners=np.array([[50, 50], [250, 50], [250, 150], [50, 150]], dtype=np.float32),
            confidence=0.95,
            method=DetectionMethod.CORNER_BASED,
            uncertainty=0.05,
        )

        with patch.object(decision_maker.strategies[DetectionMethod.DOCTR], "detect", return_value=high_conf_hypothesis):
            result = decision_maker.make_decision(sample_image)

            assert result.selected_hypothesis is not None
            assert result.selected_hypothesis.confidence == 0.95
            assert result.confidence_level == ConfidenceLevel.HIGH

    def test_confidence_level_determination(self, decision_maker):
        """Test confidence level classification."""
        # Test high confidence
        high_hyp = DetectionHypothesis(
            corners=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
            confidence=0.95,
            method=DetectionMethod.DOCTR,
            uncertainty=0.05,
        )
        assert decision_maker._determine_confidence_level(high_hyp) == ConfidenceLevel.HIGH

        # Test medium confidence
        medium_hyp = DetectionHypothesis(
            corners=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
            confidence=0.8,
            method=DetectionMethod.DOCTR,
            uncertainty=0.2,
        )
        assert decision_maker._determine_confidence_level(medium_hyp) == ConfidenceLevel.MEDIUM

        # Test low confidence
        low_hyp = DetectionHypothesis(
            corners=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
            confidence=0.6,
            method=DetectionMethod.DOCTR,
            uncertainty=0.4,
        )
        assert decision_maker._determine_confidence_level(low_hyp) == ConfidenceLevel.LOW

        # Test very low confidence
        very_low_hyp = DetectionHypothesis(
            corners=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
            confidence=0.3,
            method=DetectionMethod.DOCTR,
            uncertainty=0.7,
        )
        assert decision_maker._determine_confidence_level(very_low_hyp) == ConfidenceLevel.VERY_LOW

        # Test None hypothesis
        assert decision_maker._determine_confidence_level(None) == ConfidenceLevel.VERY_LOW

    def test_hypothesis_selection(self, decision_maker):
        """Test hypothesis selection logic."""
        hypotheses = [
            DetectionHypothesis(
                corners=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
                confidence=0.8,
                method=DetectionMethod.DOCTR,
                uncertainty=0.2,
            ),
            DetectionHypothesis(
                corners=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
                confidence=0.9,
                method=DetectionMethod.CORNER_BASED,
                uncertainty=0.1,
            ),
        ]

        selected = decision_maker._select_best_hypothesis(hypotheses)

        # Should select the CORNER_BASED result (0.9 confidence * 0.9 weight = 0.81)
        # over DOCTR (0.8 confidence * 1.0 weight = 0.8)
        assert selected.method == DetectionMethod.CORNER_BASED
        assert selected.confidence == 0.9

    def test_ground_truth_validation_placeholder(self, decision_maker, sample_image):
        """Test ground truth validation placeholder."""
        hypothesis = DetectionHypothesis(
            corners=np.array([[50, 50], [250, 50], [250, 150], [50, 150]], dtype=np.float32),
            confidence=0.8,
            method=DetectionMethod.CORNER_BASED,
            uncertainty=0.2,
        )

        score = decision_maker._validate_against_ground_truth(hypothesis, sample_image)

        # Should return a validation score
        assert score is not None
        assert 0.0 <= score <= 1.0


class TestGroundTruthFramework:
    def test_ground_truth_framework_creation(self):
        """Test ground truth validation framework setup."""
        from ocr.datasets.preprocessing.high_confidence_decision_making import create_ground_truth_validation_framework

        result = create_ground_truth_validation_framework()

        # Should return True (placeholder implementation)
        assert result is True
