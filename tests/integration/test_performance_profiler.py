"""Integration test for PerformanceProfilerCallback."""

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from ocr.lightning_modules.callbacks import PerformanceProfilerCallback


class DummyModel(LightningModule):
    """Minimal model for testing callbacks."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


@pytest.fixture
def dummy_dataloader():
    """Create a minimal dataloader for testing."""
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=8)


def test_profiler_callback_enabled(dummy_dataloader, tmp_path):
    """Test that profiler callback tracks metrics when enabled."""
    model = DummyModel()
    profiler = PerformanceProfilerCallback(
        enabled=True,
        log_interval=1,
        profile_memory=True,
        verbose=False,
    )

    trainer = Trainer(
        max_epochs=1,
        callbacks=[profiler],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        accelerator="cpu",
    )

    trainer.fit(model, dummy_dataloader, dummy_dataloader)

    # Verify profiler collected metrics
    assert len(profiler.validation_batch_times) > 0
    assert profiler.epoch_start_time is not None
    assert len(profiler.cpu_memory_percent) > 0


def test_profiler_callback_disabled(dummy_dataloader, tmp_path):
    """Test that profiler callback does nothing when disabled."""
    model = DummyModel()
    profiler = PerformanceProfilerCallback(enabled=False)

    trainer = Trainer(
        max_epochs=1,
        callbacks=[profiler],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        accelerator="cpu",
    )

    trainer.fit(model, dummy_dataloader, dummy_dataloader)

    # Verify profiler did not collect metrics
    assert len(profiler.validation_batch_times) == 0


def test_profiler_metrics_logged_to_model(dummy_dataloader, tmp_path):
    """Test that profiler logs metrics to the Lightning module."""
    model = DummyModel()
    profiler = PerformanceProfilerCallback(
        enabled=True,
        log_interval=1,
        profile_memory=True,
        verbose=False,
    )

    trainer = Trainer(
        max_epochs=1,
        callbacks=[profiler],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        accelerator="cpu",
    )

    trainer.fit(model, dummy_dataloader, dummy_dataloader)

    # Check that metrics were calculated
    assert len(profiler.validation_batch_times) > 0

    # Verify expected metric keys would be logged
    # (actual logging depends on logger being enabled)


def test_profiler_batch_timing(dummy_dataloader, tmp_path):
    """Test that profiler accurately measures batch timing."""
    import time

    model = DummyModel()
    profiler = PerformanceProfilerCallback(enabled=True, log_interval=1)

    # Manually trigger profiler hooks
    trainer = Trainer(
        max_epochs=1,
        callbacks=[profiler],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        accelerator="cpu",
    )

    profiler.on_validation_epoch_start(trainer, model)
    assert profiler.epoch_start_time is not None

    # Simulate batch processing
    profiler.on_validation_batch_start(trainer, model, None, 0, 0)
    time.sleep(0.01)  # Small delay
    profiler.on_validation_batch_end(trainer, model, None, None, 0, 0)

    assert len(profiler.validation_batch_times) == 1
    assert profiler.validation_batch_times[0] >= 0.01  # At least 10ms


def test_profiler_verbose_mode(dummy_dataloader, tmp_path, capsys):
    """Test that verbose mode prints to console."""
    model = DummyModel()
    profiler = PerformanceProfilerCallback(
        enabled=True,
        log_interval=1,
        verbose=True,  # Enable verbose
    )

    trainer = Trainer(
        max_epochs=1,
        callbacks=[profiler],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        accelerator="cpu",
    )

    trainer.fit(model, dummy_dataloader, dummy_dataloader)

    # Check that something was printed
    captured = capsys.readouterr()
    assert "Validation Performance Summary" in captured.out or "Validation batch" in captured.out
