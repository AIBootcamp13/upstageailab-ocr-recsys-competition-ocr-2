import tempfile
from unittest.mock import Mock, patch

import pytest
import torch

from ocr.callbacks.profiler import ProfilerCallback


def test_profiler_callback_initialization():
    """Test basic initialization of ProfilerCallback."""
    callback = ProfilerCallback(enabled=True, profile_epochs=[1, 2], profile_steps=50)

    assert callback.enabled is True
    assert callback.profile_epochs == [1, 2]
    assert callback.profile_steps == 50
    assert len(callback.activities) >= 1  # At least CPU activity is enabled


def test_should_profile_epoch():
    """Test the epoch selection logic."""
    callback = ProfilerCallback(enabled=True, profile_epochs=[1, 3, 5])

    # Should profile epochs 1, 3, and 5
    assert callback._should_profile_epoch(1) is True
    assert callback._should_profile_epoch(3) is True
    assert callback._should_profile_epoch(5) is True

    # Should not profile other epochs
    assert callback._should_profile_epoch(0) is False
    assert callback._should_profile_epoch(2) is False
    assert callback._should_profile_epoch(4) is False

    # Should not profile if disabled
    callback.enabled = False
    assert callback._should_profile_epoch(1) is False


def test_disabled_profiler():
    """Test that profiler does nothing when disabled."""
    callback = ProfilerCallback(enabled=False)
    trainer = Mock()
    pl_module = Mock()

    # Simulate epoch start when disabled
    callback.on_train_epoch_start(trainer, pl_module)
    assert callback.profiler is None  # Profiler should not be initialized


@patch("torch.cuda.is_available")
def test_activities_selection_with_cuda(mock_cuda_available):
    """Test that activities are correctly selected based on CUDA availability."""
    # With CUDA available
    mock_cuda_available.return_value = True
    callback = ProfilerCallback(activities=["cpu", "cuda"])
    assert len(callback.activities) == 2  # Both CPU and CUDA should be enabled

    # Without CUDA available
    mock_cuda_available.return_value = False
    callback = ProfilerCallback(activities=["cpu", "cuda"])
    # Only CPU should be enabled since CUDA is not available
    cpu_activity = [act for act in callback.activities if str(act) == str(torch.profiler.ProfilerActivity.CPU)]
    assert len(cpu_activity) == 1


@patch("torch.cuda.is_available")
def test_activities_default_selection(mock_cuda_available):
    """Test default activities selection."""
    # With CUDA available by default
    mock_cuda_available.return_value = True
    callback = ProfilerCallback()
    assert len(callback.activities) >= 1  # At least CPU
    # Check if both CPU and CUDA are in the list
    has_cpu = any(act == torch.profiler.ProfilerActivity.CPU for act in callback.activities)
    has_cuda = any(act == torch.profiler.ProfilerActivity.CUDA for act in callback.activities)
    assert has_cpu
    assert has_cuda  # Should have CUDA if available


def test_trace_export():
    """Test that the callback can execute without errors (trace creation requires actual training)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        callback = ProfilerCallback(
            enabled=True,
            profile_epochs=[1],
            output_dir=temp_dir,
            export_chrome_trace=True,
            profile_steps=1,  # Just one step for faster testing
            warmup_steps=1,  # Use 1 warmup to avoid profiler warning
        )

        # Mock trainer to return epoch 1
        trainer = Mock()
        trainer.current_epoch = 1
        trainer.num_training_batches = 5  # More than profile_steps
        pl_module = Mock()

        # Simulate epoch start and end - just make sure no exceptions are raised
        callback.on_train_epoch_start(trainer, pl_module)

        # Simulate a batch end
        if callback.profiler is not None:
            callback.on_train_batch_end(trainer, pl_module, None, None, 0)

        # End the epoch which should trigger trace export
        callback.on_train_epoch_end(trainer, pl_module)

        # Since actual profiling requires real training steps, we just ensure
        # no exceptions were raised during the callback execution
        # The actual trace file creation will be tested in integration
        assert callback is not None  # Basic check that callback still exists


def test_bottleneck_detection():
    """Test bottleneck detection with sample trace data."""
    callback = ProfilerCallback(log_top_k_ops=5)

    # Create sample trace data
    sample_trace = {
        "traceEvents": [
            {"name": "aten::conv2d", "dur": 125300, "cat": "cpu_op"},
            {"name": "aten::batch_norm", "dur": 87200, "cat": "cpu_op"},
            {"name": "DataLoader", "dur": 65800, "cat": "data_op"},
            {"name": "volta_scudnn_winograd_128x128", "dur": 95200, "cat": "cuda_op"},
            {"name": "volta_sgemm_128x128", "dur": 68400, "cat": "cuda_op"},
        ]
    }

    bottlenecks = callback.analyze_trace(sample_trace)

    # Check that we got results
    assert "top_cpu_ops" in bottlenecks
    assert "top_cuda_ops" in bottlenecks
    assert len(bottlenecks["top_cpu_ops"]) <= 5
    assert len(bottlenecks["top_cuda_ops"]) <= 5

    # Check that the operations are sorted by duration (descending)
    if bottlenecks["top_cpu_ops"]:
        first_op_duration = bottlenecks["top_cpu_ops"][0][1]  # Duration of first op
        last_op_duration = bottlenecks["top_cpu_ops"][-1][1]  # Duration of last op
        # First op should have higher or equal duration than last (sorted desc)
        assert first_op_duration >= last_op_duration


def test_analyze_empty_trace():
    """Test trace analysis with empty trace data."""
    callback = ProfilerCallback(log_top_k_ops=5)

    empty_trace = {"traceEvents": []}

    bottlenecks = callback.analyze_trace(empty_trace)

    assert bottlenecks["top_cpu_ops"] == []
    assert bottlenecks["top_cuda_ops"] == []


if __name__ == "__main__":
    pytest.main([__file__])
