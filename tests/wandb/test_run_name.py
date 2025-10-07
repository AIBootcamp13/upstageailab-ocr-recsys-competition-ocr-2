from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from ocr.utils.wandb_utils import generate_run_name


@pytest.fixture(autouse=True)
def reset_env(monkeypatch: pytest.MonkeyPatch):
    """Ensure WANDB_USER is cleared between tests."""
    monkeypatch.delenv("WANDB_USER", raising=False)
    yield
    monkeypatch.delenv("WANDB_USER", raising=False)


def test_generate_run_name_full_descriptor(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WANDB_USER", "tester")

    # Ensure architecture components are registered for fallback defaults
    import ocr.models.architectures.dbnet  # noqa: F401

    cfg = OmegaConf.create(
        {
            "wandb": {"experiment_tag": "Decoder Bench"},
            "dataloaders": {"train_dataloader": {"batch_size": 12}},
            "model": {
                "architecture_name": "dbnet",
                "optimizer": {"lr": 1e-3},
                "component_overrides": {
                    "encoder": {"model_name": "resnet18"},
                    "decoder": {"name": "unet"},
                },
                "encoder": {"model_name": "resnet18"},
                "decoder": {"name": "unet"},
                "head": {"_target_": "ocr.models.head.db_head.DBHead"},
                "loss": {"_target_": "ocr.models.loss.db_loss.DBLoss"},
            },
        }
    )

    run_name = generate_run_name(cfg)

    assert run_name.endswith("_SCORE_PLACEHOLDER")
    assert run_name.startswith("tester_decoder-bench_")
    assert "dbnet" in run_name
    assert "resnet18" in run_name
    assert "unet" in run_name
    assert "bs12" in run_name
    assert "lr1e-3" in run_name


def test_generate_run_name_without_architecture():
    cfg = OmegaConf.create(
        {
            "dataloaders": {"train_dataloader": {"batch_size": 8}},
            "model": {
                "optimizer": {"lr": 0.01},
                "encoder": {"_target_": "ocr.models.encoder.custom.CustomEncoder"},
                "decoder": {"_target_": "ocr.models.decoder.custom.CustomDecoder"},
            },
        }
    )

    run_name = generate_run_name(cfg)

    assert run_name.startswith("user_")
    assert "customencoder" in run_name
    assert "customdecoder" in run_name
    assert "bs8" in run_name
    assert "lr0.01" in run_name or "lr0-01" in run_name
