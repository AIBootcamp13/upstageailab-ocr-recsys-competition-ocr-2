"""Integration tests for Checkpoint Catalog V2 fallback hierarchy.

This module tests the complete fallback chain: YAML → Wandb → Config → Checkpoint
and verifies end-to-end behavior of the V2 catalog system.

Test Scenarios:
    - Fast path: Metadata YAML files available
    - Wandb fallback: No YAML, but run ID available
    - Config fallback: No YAML, no Wandb, use config files
    - Checkpoint fallback: Load checkpoint state dict (slowest)
    - Mixed scenarios: Some checkpoints with metadata, some without
    - Error recovery: Corrupt files, missing data, offline mode
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import yaml

from ui.apps.inference.services.checkpoint.catalog import CheckpointCatalogBuilder, build_catalog
from ui.apps.inference.services.checkpoint.metadata_loader import save_metadata
from ui.apps.inference.services.checkpoint.types import (
    CheckpointingConfig,
    CheckpointMetadataV1,
    DecoderInfo,
    EncoderInfo,
    HeadInfo,
    LossInfo,
    MetricsInfo,
    ModelInfo,
    TrainingInfo,
)


class TestFallbackHierarchy:
    """Test the complete fallback hierarchy: YAML → Wandb → Config → Checkpoint."""

    @pytest.fixture
    def outputs_dir(self):
        """Create temporary outputs directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def _create_experiment_structure(self, outputs_dir: Path, exp_name: str = "test_exp"):
        """Helper to create experiment directory structure."""
        exp_dir = outputs_dir / exp_name
        exp_dir.mkdir(parents=True)

        checkpoints_dir = exp_dir / "checkpoints"
        checkpoints_dir.mkdir()

        hydra_dir = exp_dir / ".hydra"
        hydra_dir.mkdir()

        return exp_dir, checkpoints_dir, hydra_dir

    def _create_checkpoint_file(self, checkpoint_path: Path, epoch: int = 10, include_metrics: bool = True):
        """Helper to create a checkpoint file with state dict."""
        state_dict = {
            "state_dict": {
                "encoder.model.conv1.weight": torch.randn(64, 3, 7, 7),
                "encoder.model.layer1.0.conv1.weight": torch.randn(64, 64, 3, 3),
            },
            "epoch": epoch,
        }

        if include_metrics:
            state_dict["cleval_metrics"] = {
                "hmean": 0.85,
                "precision": 0.88,
                "recall": 0.82,
            }

        torch.save(state_dict, checkpoint_path)

    def _create_config_file(self, config_path: Path, architecture: str = "dbnet", encoder: str = "resnet50"):
        """Helper to create config YAML file."""
        config = {
            "model": {
                "architecture": architecture,
                "encoder": {"model_name": encoder, "pretrained": True, "frozen": False},
                "decoder": {"name": "fpn_decoder"},
                "head": {"name": "detection_head"},
                "loss": {"name": "db_loss"},
            },
            "trainer": {"max_epochs": 20},
        }

        with config_path.open("w") as f:
            yaml.safe_dump(config, f)

    def _create_metadata_yaml(self, checkpoint_path: Path, exp_name: str = "test_exp", epoch: int = 10, hmean: float = 0.85):
        """Helper to create metadata YAML file."""
        metadata = CheckpointMetadataV1(
            schema_version="1.0",
            checkpoint_path=str(checkpoint_path),
            exp_name=exp_name,
            created_at=datetime.now().isoformat(),
            training=TrainingInfo(epoch=epoch, global_step=epoch * 100, training_phase="training", max_epochs=20),
            model=ModelInfo(
                architecture="dbnet",
                encoder=EncoderInfo(model_name="resnet50", pretrained=True, frozen=False),
                decoder=DecoderInfo(name="fpn_decoder"),
                head=HeadInfo(name="detection_head"),
                loss=LossInfo(name="db_loss"),
            ),
            metrics=MetricsInfo(hmean=hmean, precision=0.88, recall=0.82, validation_loss=0.15),
            checkpointing=CheckpointingConfig(monitor="val/hmean", mode="max", save_top_k=3, save_last=True),
        )

        save_metadata(metadata, checkpoint_path)

    def test_fast_path_metadata_yaml(self, outputs_dir):
        """Test fast path: All checkpoints have metadata YAML files."""
        exp_dir, ckpt_dir, _ = self._create_experiment_structure(outputs_dir)

        # Create 3 checkpoints with metadata
        for i in range(1, 4):
            ckpt_path = ckpt_dir / f"epoch={i}.ckpt"
            ckpt_path.touch()
            self._create_metadata_yaml(ckpt_path, epoch=i, hmean=0.80 + i * 0.01)

        # Build catalog
        catalog = build_catalog(outputs_dir, use_cache=False, use_wandb_fallback=False)

        # Verify all loaded via fast path
        assert catalog.total_count == 3
        assert catalog.metadata_available_count == 3
        assert all(entry.has_metadata for entry in catalog.entries)

        # Should be very fast
        assert catalog.catalog_build_time_seconds < 1.0

        # Verify sorting by epoch
        assert catalog.entries[0].epochs == 1
        assert catalog.entries[1].epochs == 2
        assert catalog.entries[2].epochs == 3

    def test_legacy_path_no_metadata(self, outputs_dir):
        """Test legacy path: No metadata files, use checkpoint loading."""
        exp_dir, ckpt_dir, hydra_dir = self._create_experiment_structure(outputs_dir)

        # Create config file
        config_path = hydra_dir / "config.yaml"
        self._create_config_file(config_path)

        # Create checkpoint WITHOUT metadata YAML
        ckpt_path = ckpt_dir / "epoch=5.ckpt"
        self._create_checkpoint_file(ckpt_path, epoch=5, include_metrics=True)

        # Build catalog
        catalog = build_catalog(outputs_dir, use_cache=False, use_wandb_fallback=False)

        # Verify loaded via legacy path
        assert catalog.total_count == 1
        assert catalog.metadata_available_count == 0
        assert not catalog.entries[0].has_metadata

        # Should still extract metrics and architecture
        entry = catalog.entries[0]
        # Note: epoch comes from checkpoint file, which has epoch=5
        assert entry.epochs == 5  # From checkpoint state dict
        assert entry.architecture == "dbnet"
        assert entry.backbone == "resnet50"
        assert entry.hmean == 0.85

    def test_mixed_metadata_availability(self, outputs_dir):
        """Test mixed scenario: Some checkpoints with metadata, some without."""
        exp_dir, ckpt_dir, hydra_dir = self._create_experiment_structure(outputs_dir)

        # Create config file for legacy fallback
        config_path = hydra_dir / "config.yaml"
        self._create_config_file(config_path)

        # Create checkpoint 1 with metadata (fast path)
        ckpt1 = ckpt_dir / "epoch=1.ckpt"
        ckpt1.touch()
        self._create_metadata_yaml(ckpt1, epoch=1)

        # Create checkpoint 2 without metadata (legacy path)
        ckpt2 = ckpt_dir / "epoch=2.ckpt"
        self._create_checkpoint_file(ckpt2, epoch=2, include_metrics=True)

        # Create checkpoint 3 with metadata (fast path)
        ckpt3 = ckpt_dir / "epoch=3.ckpt"
        ckpt3.touch()
        self._create_metadata_yaml(ckpt3, epoch=3)

        # Build catalog
        catalog = build_catalog(outputs_dir, use_cache=False, use_wandb_fallback=False)

        # Verify mixed loading
        assert catalog.total_count == 3
        assert catalog.metadata_available_count == 2  # Only ckpt1 and ckpt3

        # Check individual entries
        entries_by_epoch = {e.epochs: e for e in catalog.entries}
        assert entries_by_epoch[1].has_metadata is True
        assert entries_by_epoch[2].has_metadata is False
        assert entries_by_epoch[3].has_metadata is True

    def test_corrupt_metadata_fallback(self, outputs_dir):
        """Test fallback when metadata YAML is corrupt."""
        exp_dir, ckpt_dir, hydra_dir = self._create_experiment_structure(outputs_dir)

        # Create config file for fallback
        config_path = hydra_dir / "config.yaml"
        self._create_config_file(config_path)

        # Create checkpoint
        ckpt_path = ckpt_dir / "epoch=5.ckpt"
        self._create_checkpoint_file(ckpt_path, epoch=5)

        # Create corrupt metadata file
        metadata_path = ckpt_path.with_suffix(".metadata.yaml")
        with metadata_path.open("w") as f:
            f.write("{ corrupt yaml: [unclosed")

        # Build catalog - should fall back to legacy path
        catalog = build_catalog(outputs_dir, use_cache=False, use_wandb_fallback=False)

        # Should still work via fallback
        assert catalog.total_count == 1
        assert catalog.metadata_available_count == 0
        assert catalog.entries[0].epochs == 5

    def test_wandb_fallback_path(self, outputs_dir):
        """Test Wandb API fallback when metadata YAML unavailable."""
        exp_dir, ckpt_dir, hydra_dir = self._create_experiment_structure(outputs_dir)

        # Create checkpoint
        ckpt_path = ckpt_dir / "epoch=5.ckpt"
        ckpt_path.touch()

        # Create Hydra config with wandb run ID
        config_path = hydra_dir / "config.yaml"
        config = {
            "model": {"architecture": "dbnet", "encoder": {"model_name": "resnet50"}},
            "logger": {"wandb": {"id": "abc123"}},
        }
        with config_path.open("w") as f:
            yaml.safe_dump(config, f)

        # Mock Wandb client
        with patch.dict("os.environ", {"WANDB_API_KEY": "test_key"}):
            with patch("ui.apps.inference.services.checkpoint.wandb_client.WandbClient") as MockWandbClient:
                # Create mock client
                mock_client = Mock()

                # Mock successful metadata retrieval
                mock_metadata = CheckpointMetadataV1(
                    schema_version="1.0",
                    checkpoint_path=str(ckpt_path),
                    exp_name="test_exp",
                    created_at=datetime.now().isoformat(),
                    training=TrainingInfo(epoch=5, global_step=500, training_phase="training", max_epochs=20),
                    model=ModelInfo(
                        architecture="dbnet",
                        encoder=EncoderInfo(model_name="resnet50", pretrained=True, frozen=False),
                        decoder=DecoderInfo(name="fpn_decoder"),
                        head=HeadInfo(name="detection_head"),
                        loss=LossInfo(name="db_loss"),
                    ),
                    metrics=MetricsInfo(hmean=0.87, precision=0.90, recall=0.84),
                    checkpointing=CheckpointingConfig(monitor="val/hmean", mode="max"),
                    wandb_run_id="abc123",
                )

                mock_client.get_metadata_from_wandb.return_value = mock_metadata
                MockWandbClient.return_value = mock_client

                # Build catalog with Wandb fallback enabled
                builder = CheckpointCatalogBuilder(outputs_dir, use_cache=False, use_wandb_fallback=True)
                builder.wandb_client = mock_client

                catalog = builder.build_catalog()

                # Verify loaded via Wandb fallback
                assert catalog.total_count == 1
                assert catalog.metadata_available_count == 1
                entry = catalog.entries[0]
                assert entry.has_metadata is True
                assert entry.hmean == 0.87

    def test_invalid_checkpoint_filtering(self, outputs_dir):
        """Test that invalid checkpoints (epoch=0) are filtered out."""
        exp_dir, ckpt_dir, _ = self._create_experiment_structure(outputs_dir)

        # Create checkpoint with epoch=0
        ckpt_path = ckpt_dir / "invalid.ckpt"
        self._create_checkpoint_file(ckpt_path, epoch=0, include_metrics=False)

        # Build catalog
        catalog = build_catalog(outputs_dir, use_cache=False, use_wandb_fallback=False)

        # Should be filtered out
        assert catalog.total_count == 0

    def test_multiple_experiments_in_outputs(self, outputs_dir):
        """Test catalog building with multiple experiments in outputs directory."""
        # Create experiment 1
        exp1_dir, ckpt1_dir, _ = self._create_experiment_structure(outputs_dir, "experiment_001")
        ckpt1 = ckpt1_dir / "epoch=5.ckpt"
        ckpt1.touch()
        self._create_metadata_yaml(ckpt1, exp_name="experiment_001", epoch=5)

        # Create experiment 2
        exp2_dir, ckpt2_dir, _ = self._create_experiment_structure(outputs_dir, "experiment_002")
        ckpt2 = ckpt2_dir / "epoch=10.ckpt"
        ckpt2.touch()
        self._create_metadata_yaml(ckpt2, exp_name="experiment_002", epoch=10)

        # Build catalog
        catalog = build_catalog(outputs_dir, use_cache=False, use_wandb_fallback=False)

        # Should find both
        assert catalog.total_count == 2
        assert {entry.exp_name for entry in catalog.entries} == {"experiment_001", "experiment_002"}


class TestCacheInvalidation:
    """Test cache invalidation and expiration mechanisms."""

    @pytest.fixture
    def outputs_dir(self):
        """Create temporary outputs directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_cache_invalidates_on_new_checkpoint(self, outputs_dir):
        """Test that adding a new checkpoint invalidates cache."""
        from ui.apps.inference.services.checkpoint.cache import clear_global_cache

        # Clear cache before test to ensure clean state
        clear_global_cache()

        exp_dir = outputs_dir / "test_exp"
        exp_dir.mkdir()
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir()

        # Create first checkpoint
        ckpt1 = ckpt_dir / "epoch=1.ckpt"
        ckpt1.touch()

        metadata1 = CheckpointMetadataV1(
            schema_version="1.0",
            checkpoint_path=str(ckpt1),
            exp_name="test_exp",
            created_at=datetime.now().isoformat(),
            training=TrainingInfo(epoch=1, global_step=100, training_phase="training"),
            model=ModelInfo(
                architecture="dbnet",
                encoder=EncoderInfo(model_name="resnet50", pretrained=True, frozen=False),
                decoder=DecoderInfo(name="fpn"),
                head=HeadInfo(name="head"),
                loss=LossInfo(name="loss"),
            ),
            metrics=MetricsInfo(hmean=0.85, precision=0.88, recall=0.82),
            checkpointing=CheckpointingConfig(monitor="val/hmean", mode="max"),
        )
        save_metadata(metadata1, ckpt1)

        # Build catalog (will be cached)
        catalog1 = build_catalog(outputs_dir, use_cache=True, use_wandb_fallback=False)
        assert catalog1.total_count == 1

        # Sleep to ensure mtime will be different
        import time

        time.sleep(1.1)  # Ensure at least 1 second passes for mtime granularity

        # Add new checkpoint
        ckpt2 = ckpt_dir / "epoch=2.ckpt"
        ckpt2.touch()

        metadata2 = metadata1.model_copy(
            update={
                "checkpoint_path": str(ckpt2),
                "training": TrainingInfo(epoch=2, global_step=200, training_phase="training"),
            }
        )
        save_metadata(metadata2, ckpt2)

        # Touch the OUTPUTS directory to invalidate cache
        # (cache key is based on outputs_dir mtime, not subdirectories)
        outputs_dir.touch()

        # Rebuild catalog - should detect mtime change and rebuild
        catalog2 = build_catalog(outputs_dir, use_cache=True, use_wandb_fallback=False)

        # Should reflect new checkpoint
        assert catalog2.total_count == 2

    def test_cache_cleared_manually(self, outputs_dir):
        """Test manual cache clearing."""
        from ui.apps.inference.services.checkpoint.cache import clear_global_cache

        exp_dir = outputs_dir / "test_exp"
        exp_dir.mkdir()
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir()

        ckpt1 = ckpt_dir / "epoch=1.ckpt"
        ckpt1.touch()

        metadata = CheckpointMetadataV1(
            schema_version="1.0",
            checkpoint_path=str(ckpt1),
            exp_name="test_exp",
            created_at=datetime.now().isoformat(),
            training=TrainingInfo(epoch=1, global_step=100, training_phase="training"),
            model=ModelInfo(
                architecture="dbnet",
                encoder=EncoderInfo(model_name="resnet50", pretrained=True, frozen=False),
                decoder=DecoderInfo(name="fpn"),
                head=HeadInfo(name="head"),
                loss=LossInfo(name="loss"),
            ),
            metrics=MetricsInfo(hmean=0.85),
            checkpointing=CheckpointingConfig(monitor="val/hmean", mode="max"),
        )
        save_metadata(metadata, ckpt1)

        # Build and cache
        catalog1 = build_catalog(outputs_dir, use_cache=True, use_wandb_fallback=False)

        # Clear cache
        clear_global_cache()

        # Rebuild - should not use cache
        catalog2 = build_catalog(outputs_dir, use_cache=True, use_wandb_fallback=False)

        # Catalogs should have same content but potentially different instances
        assert catalog2.total_count == catalog1.total_count


class TestErrorRecovery:
    """Test error handling and recovery for various failure scenarios."""

    @pytest.fixture
    def outputs_dir(self):
        """Create temporary outputs directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_missing_config_file(self, outputs_dir):
        """Test handling of missing config files."""
        exp_dir = outputs_dir / "test_exp"
        exp_dir.mkdir()
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir()

        # Create checkpoint without config or metadata
        ckpt_path = ckpt_dir / "epoch=5.ckpt"
        ckpt_path.touch()

        # Build catalog - should use path inference
        catalog = build_catalog(outputs_dir, use_cache=False, use_wandb_fallback=False)

        # Should still create entry (with limited info)
        assert catalog.total_count == 1
        entry = catalog.entries[0]
        assert entry.epochs == 5  # Extracted from filename

    def test_permission_error_handling(self, outputs_dir):
        """Test handling of permission errors when reading files."""
        exp_dir = outputs_dir / "test_exp"
        exp_dir.mkdir()
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir()

        ckpt_path = ckpt_dir / "epoch=5.ckpt"
        ckpt_path.touch()

        # Make file unreadable (this may not work on all systems)
        try:
            ckpt_path.chmod(0o000)

            # Build catalog - should handle gracefully
            catalog = build_catalog(outputs_dir, use_cache=False, use_wandb_fallback=False)

            # May or may not include the checkpoint depending on error handling
            # At minimum, should not crash
            assert catalog is not None

        finally:
            # Restore permissions for cleanup
            ckpt_path.chmod(0o644)

    def test_empty_outputs_directory(self, outputs_dir):
        """Test handling of empty outputs directory."""
        catalog = build_catalog(outputs_dir, use_cache=False, use_wandb_fallback=False)

        assert catalog.total_count == 0
        assert catalog.metadata_available_count == 0
        assert len(catalog.entries) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
