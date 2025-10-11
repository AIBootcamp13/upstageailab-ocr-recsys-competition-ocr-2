"""Helpers for saving and restoring extra metrics in checkpoints."""

from __future__ import annotations

from typing import Any


class CheckpointHandler:
    """Handle checkpoint hooks for the Lightning module."""

    @staticmethod
    def on_save_checkpoint(module, checkpoint: dict[str, Any]) -> dict[str, Any]:
        """Inject CLEval metrics into the checkpoint payload when available."""
        if hasattr(module, "_checkpoint_metrics"):
            checkpoint["cleval_metrics"] = module._checkpoint_metrics
        return checkpoint

    @staticmethod
    def on_load_checkpoint(module, checkpoint: dict[str, Any]) -> None:
        """Restore CLEval metrics from the checkpoint when present."""
        if "cleval_metrics" in checkpoint:
            module._checkpoint_metrics = checkpoint["cleval_metrics"]
