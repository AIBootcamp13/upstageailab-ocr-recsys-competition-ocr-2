"""
Simple test script to verify ResourceMonitorCallback functionality.
"""

from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.demos.boring_classes import BoringModel

from ocr.callbacks.resource_monitor import ResourceMonitorCallback


def test_callback_with_simple_model():
    """Test ResourceMonitorCallback with a simple model."""
    print("Creating model and callback...")

    model = BoringModel()
    callback = ResourceMonitorCallback(
        enabled=True,
        log_interval=2,  # Log every 2 batches
        gpu_monitoring=False,  # Disable GPU monitoring to avoid issues on CPU-only systems
        cpu_monitoring=True,
        io_monitoring=True,
        alert_gpu_underutilization=False,  # Disable GPU alerting
        export_timeseries=True,
        timeseries_path="test_resource_logs",
    )

    print("Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[callback],
        limit_train_batches=10,  # Only run a few batches for testing
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="cpu",  # Use CPU to avoid GPU issues
    )

    print("Starting training...")
    trainer.fit(model)

    print("Training completed. Checking for exported logs...")

    # Check if the logs were exported
    log_path = Path("test_resource_logs")
    if log_path.exists():
        print(f"Found log directory: {log_path}")
        for file in log_path.glob("*.csv"):
            print(f"  Found log file: {file}")
            with open(file) as f:
                lines = f.readlines()
                print(f"    Number of entries: {len(lines)-1}")  # -1 for header
                if len(lines) > 1:
                    print(f"    Sample entry: {lines[1].strip()}")
    else:
        print("No log directory found")

    print("Test completed successfully!")


if __name__ == "__main__":
    test_callback_with_simple_model()
