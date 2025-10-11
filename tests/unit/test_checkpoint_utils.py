"""Unit tests for checkpoint_utils module."""

from ocr.lightning_modules.utils.checkpoint_utils import CheckpointHandler


class DummyModule:
    """A dummy module to simulate a Lightning module for testing purposes."""

    def __init__(self, checkpoint_metrics=None):
        if checkpoint_metrics is not None:
            self._checkpoint_metrics = checkpoint_metrics


class TestCheckpointHandler:
    """Tests for CheckpointHandler class."""

    def test_on_save_checkpoint_without_metrics(self):
        """Test on_save_checkpoint when module has no _checkpoint_metrics attribute."""
        module = DummyModule()
        checkpoint = {"epoch": 1, "model_state": "dummy_state"}

        result = CheckpointHandler.on_save_checkpoint(module, checkpoint)

        # The checkpoint should remain unchanged
        assert result == {"epoch": 1, "model_state": "dummy_state"}
        assert "cleval_metrics" not in result

    def test_on_save_checkpoint_with_metrics(self):
        """Test on_save_checkpoint when module has _checkpoint_metrics attribute."""
        metrics = {"precision": 0.95, "recall": 0.89, "hmean": 0.92}
        module = DummyModule(metrics)
        checkpoint = {"epoch": 1, "model_state": "dummy_state"}

        result = CheckpointHandler.on_save_checkpoint(module, checkpoint)

        # The checkpoint should include cleval_metrics
        assert result == {"epoch": 1, "model_state": "dummy_state", "cleval_metrics": {"precision": 0.95, "recall": 0.89, "hmean": 0.92}}

    def test_on_save_checkpoint_empty_metrics(self):
        """Test on_save_checkpoint with empty metrics dictionary."""
        metrics = {}
        module = DummyModule(metrics)
        checkpoint = {"epoch": 1, "model_state": "dummy_state"}

        result = CheckpointHandler.on_save_checkpoint(module, checkpoint)

        # The checkpoint should include empty cleval_metrics
        assert result == {"epoch": 1, "model_state": "dummy_state", "cleval_metrics": {}}

    def test_on_save_checkpoint_preserves_existing_checkpoint_data(self):
        """Test that on_save_checkpoint preserves all existing checkpoint data."""
        metrics = {"f1": 0.85}
        module = DummyModule(metrics)
        checkpoint = {
            "epoch": 5,
            "global_step": 1000,
            "model_state_dict": {"layer1.weight": [0.1, 0.2]},
            "optimizer_state_dict": {"lr": 0.001},
            "random_state": "some_state",
        }

        result = CheckpointHandler.on_save_checkpoint(module, checkpoint)

        # All original checkpoint data should be preserved
        expected = {
            "epoch": 5,
            "global_step": 1000,
            "model_state_dict": {"layer1.weight": [0.1, 0.2]},
            "optimizer_state_dict": {"lr": 0.001},
            "random_state": "some_state",
            "cleval_metrics": {"f1": 0.85},
        }
        assert result == expected

    def test_on_save_checkpoint_overwrites_existing_cleval_metrics(self):
        """Test that on_save_checkpoint overwrites any existing cleval_metrics."""
        metrics = {"new_metric": 0.99}
        module = DummyModule(metrics)
        checkpoint = {
            "epoch": 1,
            "cleval_metrics": {"old_metric": 0.5},  # This should be overwritten
        }

        result = CheckpointHandler.on_save_checkpoint(module, checkpoint)

        # The cleval_metrics should come from the module, not the original checkpoint
        assert result == {"epoch": 1, "cleval_metrics": {"new_metric": 0.99}}

    def test_on_load_checkpoint_without_cleval_metrics_in_checkpoint(self):
        """Test on_load_checkpoint when checkpoint has no cleval_metrics."""
        module = DummyModule()
        checkpoint = {"epoch": 1, "model_state": "dummy_state"}

        CheckpointHandler.on_load_checkpoint(module, checkpoint)

        # The module should not have _checkpoint_metrics attribute
        assert not hasattr(module, "_checkpoint_metrics")

    def test_on_load_checkpoint_with_cleval_metrics_in_checkpoint(self):
        """Test on_load_checkpoint when checkpoint has cleval_metrics."""
        module = DummyModule()
        checkpoint = {"epoch": 1, "model_state": "dummy_state", "cleval_metrics": {"precision": 0.95, "recall": 0.89, "hmean": 0.92}}

        CheckpointHandler.on_load_checkpoint(module, checkpoint)

        # The module should now have _checkpoint_metrics attribute with the values
        assert hasattr(module, "_checkpoint_metrics")
        assert module._checkpoint_metrics == {"precision": 0.95, "recall": 0.89, "hmean": 0.92}

    def test_on_load_checkpoint_empty_cleval_metrics(self):
        """Test on_load_checkpoint with empty cleval_metrics in checkpoint."""
        module = DummyModule()
        checkpoint = {"epoch": 1, "cleval_metrics": {}}

        CheckpointHandler.on_load_checkpoint(module, checkpoint)

        # The module should have _checkpoint_metrics as an empty dict
        assert hasattr(module, "_checkpoint_metrics")
        assert module._checkpoint_metrics == {}

    def test_on_load_checkpoint_overwrites_existing_metrics(self):
        """Test that on_load_checkpoint overwrites any existing _checkpoint_metrics."""
        # Module starts with some metrics
        module = DummyModule({"old_metric": 0.5})
        checkpoint = {
            "epoch": 1,
            "cleval_metrics": {"new_metric": 0.99},  # This should overwrite the old one
        }

        CheckpointHandler.on_load_checkpoint(module, checkpoint)

        # The module should have the new metrics, not the old ones
        assert hasattr(module, "_checkpoint_metrics")
        assert module._checkpoint_metrics == {"new_metric": 0.99}

    def test_on_load_checkpoint_preserves_other_checkpoint_data(self):
        """Test that on_load_checkpoint doesn't modify the checkpoint dictionary."""
        module = DummyModule()
        original_checkpoint = {"epoch": 1, "model_state": "dummy_state", "cleval_metrics": {"precision": 0.95}}
        checkpoint = original_checkpoint.copy()  # Make a copy to test with

        CheckpointHandler.on_load_checkpoint(module, checkpoint)

        # The checkpoint dictionary should remain unchanged
        assert checkpoint == original_checkpoint
        # The module should have the metrics loaded
        assert module._checkpoint_metrics == {"precision": 0.95}

    def test_on_save_checkpoint_then_on_load_checkpoint_roundtrip(self):
        """Test that saving and then loading metrics works correctly."""
        # First, save metrics to a checkpoint
        original_metrics = {"precision": 0.95, "recall": 0.89}
        module = DummyModule(original_metrics)
        checkpoint = {"epoch": 1, "model_state": "dummy_state"}

        # Save checkpoint
        saved_checkpoint = CheckpointHandler.on_save_checkpoint(module, checkpoint)

        # Create a new module and load the checkpoint
        new_module = DummyModule()
        CheckpointHandler.on_load_checkpoint(new_module, saved_checkpoint)

        # The new module should have the same metrics as the original
        assert hasattr(new_module, "_checkpoint_metrics")
        assert new_module._checkpoint_metrics == original_metrics

    def test_on_save_checkpoint_with_complex_metrics(self):
        """Test on_save_checkpoint with complex/nested metrics."""
        complex_metrics = {
            "precision": 0.95,
            "recall": 0.89,
            "nested": {"submetric1": 0.75, "submetric2": [0.1, 0.2, 0.3]},
            "list_metric": [1, 2, 3, 4],
        }
        module = DummyModule(complex_metrics)
        checkpoint = {"epoch": 1}

        result = CheckpointHandler.on_save_checkpoint(module, checkpoint)

        expected = {"epoch": 1, "cleval_metrics": complex_metrics}
        assert result == expected

    def test_on_load_checkpoint_with_complex_metrics(self):
        """Test on_load_checkpoint with complex/nested metrics."""
        complex_metrics = {
            "precision": 0.95,
            "recall": 0.89,
            "nested": {"submetric1": 0.75, "submetric2": [0.1, 0.2, 0.3]},
            "list_metric": [1, 2, 3, 4],
        }
        module = DummyModule()
        checkpoint = {"epoch": 1, "cleval_metrics": complex_metrics}

        CheckpointHandler.on_load_checkpoint(module, checkpoint)

        assert hasattr(module, "_checkpoint_metrics")
        assert module._checkpoint_metrics == complex_metrics
