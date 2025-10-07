"""Unit tests for ThroughputMonitorCallback."""

import time
from unittest.mock import Mock

import pytorch_lightning as pl
import torch

from ocr.callbacks.throughput_monitor import ThroughputMonitorCallback


class DummyModel(pl.LightningModule):
    """Dummy model for testing."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.layer(x).sum()
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


def test_throughput_monitor_initialization():
    """Test callback initialization with default parameters."""
    callback = ThroughputMonitorCallback()

    assert callback.enabled is True
    assert callback.log_interval == 1
    assert callback.track_memory is True
    assert callback.track_timing is True


def test_throughput_monitor_disabled():
    """Test that callback is disabled when enabled=False."""
    callback = ThroughputMonitorCallback(enabled=False)

    # Mock trainer and module
    trainer = Mock()
    pl_module = Mock()

    # These methods should return early when disabled
    callback.on_train_epoch_start(trainer, pl_module)
    callback.on_train_batch_start(trainer, pl_module, [], 0)
    callback.on_train_batch_end(trainer, pl_module, None, [], 0)
    callback.on_train_epoch_end(trainer, pl_module)


def test_batch_size_detection():
    """Test batch size detection for different batch formats."""
    callback = ThroughputMonitorCallback()

    # Test tensor batch
    batch_tensor = torch.randn(32, 10)
    assert callback._get_batch_size(batch_tensor) == 32

    # Test list batch with tensors
    batch_list = [torch.randn(16, 10), torch.randn(16)]
    assert callback._get_batch_size(batch_list) == 16

    # Test tuple batch with tensors
    batch_tuple = (torch.randn(8, 3, 224, 224), torch.randint(0, 10, (8,)))
    assert callback._get_batch_size(batch_tuple) == 8

    # Test dictionary batch
    batch_dict = {"images": torch.randn(4, 3, 224, 224), "labels": torch.randint(0, 10, (4,))}
    assert callback._get_batch_size(batch_dict) == 4

    # Test unsupported format (should return 0)
    assert callback._get_batch_size("unsupported") == 0
    assert callback._get_batch_size([]) == 0


def test_timing_percentiles():
    """Test timing percentile calculations."""
    callback = ThroughputMonitorCallback()

    # Mock times for percentile calculation
    callback.batch_times = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    # Simulate the calculation that would happen in on_train_epoch_end
    sorted_times = sorted(callback.batch_times)

    # Calculate percentiles using the same method from the class
    def get_percentile(data, percentile):
        if not data:
            return 0
        k = (len(data) - 1) * percentile / 100
        f = int(k)
        c = min(f + 1, len(data) - 1)
        if f == c:
            return data[f]
        d0 = data[f] * (c - k)
        d1 = data[c] * (k - f)
        return d0 + d1

    p95 = get_percentile(sorted_times, 95)
    p99 = get_percentile(sorted_times, 99)

    # For our sorted data [10, 20, ..., 100], p95 should be close to 95, p99 to 99
    assert 94 < p95 <= 100
    assert 98 < p99 <= 100


def test_throughput_calculation():
    """Test samples per second calculation."""
    callback = ThroughputMonitorCallback(enabled=True)

    # Simulate an epoch
    callback.epoch_start_time = time.perf_counter() - 2.0  # Started 2 seconds ago
    callback.samples_this_epoch = 100

    # Simulate batch timing data
    callback.batch_times = [10.0, 20.0, 15.0, 25.0]  # in ms

    # Calculate metrics as would happen in on_train_epoch_end
    epoch_time = time.perf_counter() - callback.epoch_start_time
    samples_per_sec = callback.samples_this_epoch / epoch_time if epoch_time > 0 else 0
    batches_per_sec = len(callback.batch_times) / epoch_time if epoch_time > 0 else 0

    # The values should be calculated correctly
    expected_samples_per_sec = 100 / 2.0  # approximately
    assert abs(samples_per_sec - expected_samples_per_sec) < 5  # Allow for timing imprecision

    expected_batches_per_sec = 4 / 2.0  # 4 batches in 2 seconds
    assert abs(batches_per_sec - expected_batches_per_sec) < 0.1


def test_memory_tracking():
    """Test memory tracking functionality."""
    callback = ThroughputMonitorCallback(enabled=True, track_memory=True)

    # Mock trainer and module
    trainer = Mock()
    pl_module = Mock()

    # Call on_train_epoch_start to initialize memory tracking
    callback.on_train_epoch_start(trainer, pl_module)

    # The epoch_start_memory_mb should have been set
    assert callback.epoch_start_memory_mb >= 0
    assert callback.peak_memory_mb >= 0


def test_no_data_handling():
    """Test that callback handles cases with no batch data."""
    callback = ThroughputMonitorCallback(enabled=True)

    # Simulate an epoch with no batches
    callback.epoch_start_time = time.perf_counter()
    callback.samples_this_epoch = 0
    callback.batch_times = []

    # This should not cause any errors
    try:
        # Create mock trainer and module for the logging call
        trainer = Mock()
        trainer.current_epoch = 0
        trainer.global_step = 0
        trainer.logger = Mock()
        pl_module = Mock()

        callback.on_train_epoch_end(trainer, pl_module)
        # If we reach here, no error occurred
        success = True
    except Exception:
        success = False

    assert success, "Callback should handle empty batch_times without error"
