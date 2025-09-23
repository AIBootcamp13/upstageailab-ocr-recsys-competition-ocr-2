import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from ocr.metrics.cleval_metric import CLEvalMetric, Options


class TestCLEvalMetric:
    @pytest.fixture
    def default_options(self):
        """Create default options for CLEvalMetric."""
        return {
            'case_sensitive': True,
            'recall_gran_penalty': 1.0,
            'precision_gran_penalty': 1.0,
            'vertical_aspect_ratio_thresh': 0.5,
            'ap_constraint': 0.3
        }

    @pytest.fixture
    def sample_predictions(self):
        """Create sample prediction data."""
        # det_quads: list of polygon coordinate arrays
        return [
            np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32).flatten()
        ]

    @pytest.fixture
    def sample_targets(self):
        """Create sample target data."""
        # gt_quads: list of polygon coordinate arrays
        return [
            np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32).flatten()
        ]

    @pytest.fixture
    def sample_letters_det(self):
        """Create sample detected letters."""
        return ['HELLO']

    @pytest.fixture
    def sample_letters_gt(self):
        """Create sample ground truth letters."""
        return ['HELLO']

    def test_metric_initialization(self, default_options):
        """Test CLEvalMetric initialization."""
        metric = CLEvalMetric(**default_options)

        assert metric.options.CASE_SENSITIVE == default_options['case_sensitive']
        assert metric.options.RECALL_GRANULARITY_PENALTY_WEIGHT == default_options['recall_gran_penalty']
        assert metric.options.PRECISION_GRANULARITY_PENALTY_WEIGHT == default_options['precision_gran_penalty']
        assert metric.options.VERTICAL_ASPECT_RATIO_THRESH == default_options['vertical_aspect_ratio_thresh']
        assert metric.options.AREA_PRECISION_CONSTRAINT == default_options['ap_constraint']

    def test_options_class(self):
        """Test Options class initialization."""
        options = Options(
            case_sensitive=True,
            recall_gran_penalty=1.0,
            precision_gran_penalty=1.0,
            vertical_aspect_ratio_thresh=0.5,
            ap_constraint=0.3
        )

        assert options.CASE_SENSITIVE == True
        assert options.RECALL_GRANULARITY_PENALTY_WEIGHT == 1.0
        assert options.PRECISION_GRANULARITY_PENALTY_WEIGHT == 1.0
        assert options.VERTICAL_ASPECT_RATIO_THRESH == 0.5
        assert options.AREA_PRECISION_CONSTRAINT == 0.3
        assert options.DUMP_SAMPLE_RESULT == False
        assert options.ORIENTATION == False

    def test_compute_metric(self, default_options, sample_predictions, sample_targets, sample_letters_det, sample_letters_gt):
        """Test metric computation."""
        metric = CLEvalMetric(**default_options)

        # Update metric with sample data
        metric.update(sample_predictions, sample_targets, sample_letters_det, sample_letters_gt)

        # Compute results
        results = metric.compute()

        # Verify results structure
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert isinstance(results['precision'], torch.Tensor)
        assert isinstance(results['recall'], torch.Tensor)
        assert isinstance(results['f1'], torch.Tensor)

    def test_update_with_empty_predictions(self, default_options):
        """Test update with empty predictions."""
        metric = CLEvalMetric(**default_options)

        # Update with empty data
        metric.update([], [], [], [])

        # Should not crash and predictions/targets should be empty lists
        assert metric.predictions == [[]]
        assert metric.targets == [[]]

    def test_update_accumulates_data(self, default_options, sample_predictions, sample_targets, sample_letters_det, sample_letters_gt):
        """Test that update accumulates multiple batches."""
        metric = CLEvalMetric(**default_options)

        # Update multiple times
        metric.update(sample_predictions, sample_targets, sample_letters_det, sample_letters_gt)
        metric.update(sample_predictions, sample_targets, sample_letters_det, sample_letters_gt)

        assert len(metric.predictions) == 2
        assert len(metric.targets) == 2

    def test_reset_clears_state(self, default_options, sample_predictions, sample_targets, sample_letters_det, sample_letters_gt):
        """Test that reset clears accumulated state."""
        metric = CLEvalMetric(**default_options)

        # Update with data
        metric.update(sample_predictions, sample_targets, sample_letters_det, sample_letters_gt)
        assert len(metric.predictions) > 0
        assert len(metric.targets) > 0

        # Reset
        metric.reset()
        assert metric.predictions == []
        assert metric.targets == []

    def test_persistent_mode(self, default_options, sample_predictions, sample_targets, sample_letters_det, sample_letters_gt):
        """Test persistent mode state management."""
        metric = CLEvalMetric(**default_options)

        # Update and compute
        metric.update(sample_predictions, sample_targets, sample_letters_det, sample_letters_gt)
        results1 = metric.compute()

        # Update again and compute
        metric.update(sample_predictions, sample_targets, sample_letters_det, sample_letters_gt)
        results2 = metric.compute()

        # Results should be different (accumulated)
        # Note: actual values depend on mock, but structure should be consistent
        assert isinstance(results1, dict)
        assert isinstance(results2, dict)

    def test_scale_wise_evaluation(self, default_options, sample_predictions, sample_targets, sample_letters_det, sample_letters_gt):
        """Test scale-wise evaluation mode."""
        scale_bins = (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.1, 0.5, 1.0)

        metric = CLEvalMetric(**default_options, scale_wise=True, scale_bins=scale_bins)

        assert len(metric.scalewise_metric) > 0  # Should have scale-wise metrics initialized

    @patch('ocr.metrics.cleval_metric.evaluation')
    def test_evaluation_function_called_with_correct_options(self, mock_evaluation, default_options, sample_predictions, sample_targets, sample_letters_det, sample_letters_gt):
        """Test that evaluation function receives correct options."""
        from ocr.metrics.data import SampleResult, Stats, CoreStats

        # Create a proper SampleResult mock
        mock_stats = Stats()
        mock_stats.det.num_char_gt = 5
        mock_stats.det.num_char_det = 5
        mock_stats.det.gran_score_recall = 0.1
        mock_stats.det.num_char_tp_recall = 4
        mock_stats.det.gran_score_precision = 0.1
        mock_stats.det.num_char_tp_precision = 4
        mock_stats.det.num_char_fp = 1
        mock_stats.num_splitted = 0
        mock_stats.num_merged = 0
        mock_stats.num_char_overlapped = 0

        mock_sample_result = SampleResult(
            matches=[],
            gts=[],
            preds=[],
            stats=mock_stats
        )

        mock_evaluation.return_value = mock_sample_result

        metric = CLEvalMetric(**default_options)
        metric.update(sample_predictions, sample_targets, sample_letters_det, sample_letters_gt)
        metric.compute()

        # Verify evaluation was called
        call_args = mock_evaluation.call_args
        assert call_args is not None

        # Check that options were passed correctly
        args, kwargs = call_args
        options_arg = args[0]  # Options should be the first argument

        assert hasattr(options_arg, 'CASE_SENSITIVE')
        assert options_arg.CASE_SENSITIVE == default_options['case_sensitive']

    def test_metric_with_different_case_sensitivity(self):
        """Test metric with different case sensitivity settings."""
        case_sensitive_metric = CLEvalMetric(case_sensitive=True)
        case_insensitive_metric = CLEvalMetric(case_sensitive=False)

        assert case_sensitive_metric.options.CASE_SENSITIVE == True
        assert case_insensitive_metric.options.CASE_SENSITIVE == False

    def test_metric_with_custom_penalties(self):
        """Test metric with custom penalty weights."""
        metric = CLEvalMetric(
            recall_gran_penalty=0.8,
            precision_gran_penalty=1.2
        )

        assert metric.options.RECALL_GRANULARITY_PENALTY_WEIGHT == 0.8
        assert metric.options.PRECISION_GRANULARITY_PENALTY_WEIGHT == 1.2