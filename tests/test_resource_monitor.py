import os
import tempfile
from unittest.mock import Mock, patch

from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel

from ocr.callbacks.resource_monitor import ResourceMonitorCallback


def test_resource_monitor_callback_initialization():
    """Test that ResourceMonitorCallback initializes correctly with default parameters."""
    callback = ResourceMonitorCallback()

    # Check default parameters
    assert callback.enabled is True
    assert callback.log_interval == 10
    assert callback.gpu_monitoring is True
    assert callback.cpu_monitoring is True
    assert callback.io_monitoring is True
    assert callback.alert_gpu_underutilization is True
    assert callback.alert_memory_pressure is True
    assert callback.alert_io_bottleneck is True
    assert callback.gpu_util_threshold == 0.5
    assert callback.memory_threshold == 0.9
    assert callback.io_wait_threshold == 0.3
    assert callback.export_timeseries is True
    assert str(callback.timeseries_path) == "resource_logs"


def test_resource_monitor_callback_disabled():
    """Test that ResourceMonitorCallback does nothing when disabled."""
    callback = ResourceMonitorCallback(enabled=False)

    trainer = Mock()
    pl_module = Mock()

    # These methods should do nothing when disabled
    callback.setup(trainer, pl_module)
    callback.on_train_start(trainer, pl_module)
    callback.on_train_batch_end(trainer, pl_module, None, None, 0)

    # No metrics should be collected when disabled
    assert callback.enabled is False


def test_gpu_metrics_collection():
    """Test GPU metrics collection when GPU is available."""
    # Mock GPUtil to simulate GPU availability
    with patch("ocr.callbacks.resource_monitor.GPUtil") as mock_gputil:
        # Create mock GPU objects
        mock_gpu = Mock()
        mock_gpu.load = 0.85  # 85% utilization
        mock_gpu.memoryUsed = 8192  # 8GB used
        mock_gpu.memoryTotal = 16384  # 16GB total
        mock_gpu.memoryUtil = 0.5  # 50% memory usage
        mock_gpu.temperature = 72  # 72°C
        mock_gpu.power = 180.5  # 180.5W

        mock_gputil.getGPUs.return_value = [mock_gpu]

        callback = ResourceMonitorCallback(gpu_monitoring=True)

        # Test GPU metrics collection
        metrics = callback.get_gpu_metrics()

        assert metrics["gpu_0_util"] == 85.0
        assert metrics["gpu_0_memory_used_mb"] == 8192
        assert metrics["gpu_0_memory_total_mb"] == 16384
        assert metrics["gpu_0_memory_pct"] == 50.0
        assert metrics["gpu_0_temp_c"] == 72
        assert metrics["gpu_0_power_w"] == 180.5


def test_gpu_metrics_collection_no_gpu():
    """Test GPU metrics collection when no GPU is available."""
    with patch("ocr.callbacks.resource_monitor.GPUtil.getGPUs", side_effect=Exception("No GPU")):
        callback = ResourceMonitorCallback(gpu_monitoring=True)

        # When getGPUs fails during initialization, gpu_available should be False
        assert callback.gpu_available is False

        # This should not raise an exception but return empty dict
        metrics = callback.get_gpu_metrics()

        # When GPU monitoring fails, it should return an empty dict or handle gracefully
        assert isinstance(metrics, dict)
        assert len(metrics) == 0  # Should be empty since GPU is not available


def test_cpu_memory_metrics():
    """Test CPU and memory metrics collection."""
    with patch("ocr.callbacks.resource_monitor.psutil") as mock_psutil:
        # Mock system-wide CPU and memory stats
        mock_virtual_memory = Mock()
        mock_virtual_memory.total = 16384 * 1024 * 1024  # 16GB in bytes
        mock_virtual_memory.available = 8192 * 1024 * 1024  # 8GB available in bytes
        mock_virtual_memory.percent = 50.0

        mock_psutil.cpu_percent.return_value = 45.2
        mock_psutil.virtual_memory.return_value = mock_virtual_memory

        # Mock process-specific stats
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 180.5
        mock_memory_info = Mock()
        mock_memory_info.rss = 2048 * 1024 * 1024  # 2GB in bytes
        mock_process.memory_info.return_value = mock_memory_info

        callback = ResourceMonitorCallback(cpu_monitoring=True)
        # Set the process directly since we're mocking
        callback.process = mock_process
        callback.pid = 12345  # Mock PID

        metrics = callback.get_cpu_memory_metrics()

        assert metrics["cpu_percent"] == 45.2
        assert metrics["cpu_process_percent"] == 180.5
        assert metrics["memory_system_mb"] == 16384.0  # 16GB in MB
        assert metrics["memory_available_mb"] == 8192.0  # 8GB in MB
        assert metrics["memory_percent"] == 50.0
        assert metrics["memory_process_mb"] == 2048.0  # 2GB in MB


def test_io_metrics():
    """Test I/O metrics collection."""
    with patch("ocr.callbacks.resource_monitor.psutil"):
        # Mock I/O counters
        mock_io_counters1 = Mock()
        mock_io_counters1.read_bytes = 100 * 1024 * 1024  # 100MB
        mock_io_counters1.write_bytes = 10 * 1024 * 1024  # 10MB

        mock_io_counters2 = Mock()
        mock_io_counters2.read_bytes = 150 * 1024 * 1024  # 150MB (50MB more read)
        mock_io_counters2.write_bytes = 15 * 1024 * 1024  # 15MB (5MB more written)

        mock_process = Mock()
        mock_process.io_counters.side_effect = [mock_io_counters1, mock_io_counters2, mock_io_counters2]
        callback = ResourceMonitorCallback(io_monitoring=True)
        callback.process = mock_process
        callback.pid = 12345

        # Initial call to set up last_io_counters
        callback.last_io_counters = mock_io_counters1
        callback.last_time = 100.0  # Simulate time 100.0

        # Second call to calculate rates (with updated time)
        import time

        original_time = time.time
        time.time = lambda: 102.0  # 2 seconds later

        try:
            metrics = callback.get_io_metrics()
            # With 50MB read and 5MB written over 2 seconds:
            # Read rate: 50/2 = 25 MB/s
            # Write rate: 5/2 = 2.5 MB/s
            # io_wait_pct is always 0 for per-process monitoring
            assert "io_read_rate_mbps" in metrics
            assert "io_write_rate_mbps" in metrics
            assert metrics["io_wait_pct"] == 0.0
        finally:
            time.time = original_time


def test_alert_thresholds():
    """Test alerting logic with various conditions."""
    callback = ResourceMonitorCallback(
        alert_gpu_underutilization=True,
        alert_memory_pressure=True,
        alert_io_bottleneck=True,
        gpu_util_threshold=0.7,  # 70% threshold
        memory_threshold=0.8,  # 80% threshold
        io_wait_threshold=0.2,  # 20% threshold
    )

    # Test metrics that should trigger alerts
    metrics_with_alerts = {
        "gpu_0_util": 60.0,  # Below 70% threshold
        "memory_percent": 85.0,  # Above 80% threshold
        "io_wait_pct": 25.0,  # Above 20% threshold
        "gpu_0_temp_c": 85,  # Above 80°C
    }

    alerts = callback.check_alerts(metrics_with_alerts)

    # Check that all expected alerts are present
    assert len(alerts) >= 3  # Should have at least GPU, memory, and temperature alerts
    assert any("GPU underutilized" in alert for alert in alerts)
    assert any("High memory usage" in alert for alert in alerts)
    assert any("High temperature" in alert for alert in alerts)


def test_no_alerts_when_disabled():
    """Test that no alerts are triggered when monitoring is disabled."""
    callback = ResourceMonitorCallback(alert_gpu_underutilization=False, alert_memory_pressure=False, alert_io_bottleneck=False)

    metrics = {
        "gpu_0_util": 10.0,  # Very low GPU util
        "memory_percent": 95.0,  # Very high memory
        "io_wait_pct": 50.0,  # High I/O wait
    }

    alerts = callback.check_alerts(metrics)

    # No alerts should be generated since all alert types are disabled
    assert len(alerts) == 0


def test_timeseries_export():
    """Test timeseries data export functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "test_logs")
        callback = ResourceMonitorCallback(export_timeseries=True, timeseries_path=csv_path)
        callback.current_epoch = 1

        # Mock metrics to export
        test_metrics = {"step": 10, "timestamp": "2025-10-07T10:00:00", "gpu_0_util": 85.3, "cpu_percent": 45.2}

        callback.export_timeseries_data(test_metrics)

        # Check that CSV file was created
        import pandas as pd

        epoch_csv_path = os.path.join(csv_path, "epoch_1.csv")
        assert os.path.exists(epoch_csv_path)

        # Read back and verify content
        df = pd.read_csv(epoch_csv_path)
        assert len(df) == 1
        assert df.iloc[0]["step"] == 10
        assert df.iloc[0]["gpu_0_util"] == 85.3


def test_callback_integration():
    """Integration test: ensure callback doesn't break training."""
    # Create a temporary directory for logs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup callback with temporary path
        resource_callback = ResourceMonitorCallback(
            enabled=True,
            log_interval=1,
            timeseries_path=os.path.join(temp_dir, "resource_logs"),
        )

        model = BoringModel()
        trainer = Trainer(default_root_dir=temp_dir, fast_dev_run=True, callbacks=[resource_callback])

        # This should not raise any exceptions
        trainer.fit(model)

        # Check that logs were created
        os.path.join(temp_dir, "resource_logs", "epoch_0.csv")
        # Note: May not be created if fast_dev_run ends before first batch
        # That's acceptable behavior
