"""Wandb API client for checkpoint metadata fallback.

This module provides functionality to retrieve checkpoint metadata from Wandb
when local .metadata.yaml files are unavailable. It implements caching and
offline fallback handling.

Performance: API calls are cached to minimize network overhead.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .types import CheckpointMetadataV1, MetricsInfo, TrainingInfo

LOGGER = logging.getLogger(__name__)


class WandbClient:
    """Client for retrieving checkpoint metadata from Wandb API."""

    def __init__(self, cache_size: int = 256):
        """Initialize Wandb client with caching.

        Args:
            cache_size: Maximum number of cached API responses
        """
        self.cache_size = cache_size
        self._api: Any | None = None
        self._is_available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if Wandb is available and configured.

        Returns:
            True if Wandb API can be used, False otherwise
        """
        # Check if WANDB_API_KEY is set
        if not os.getenv("WANDB_API_KEY"):
            LOGGER.debug("WANDB_API_KEY not set - Wandb fallback disabled")
            return False

        # Try to import wandb
        try:
            import wandb  # noqa: F401

            return True
        except ImportError:
            LOGGER.debug("wandb package not installed - fallback disabled")
            return False

    @property
    def api(self) -> Any | None:
        """Lazy initialization of Wandb API client.

        Returns:
            Wandb API instance, or None if unavailable
        """
        if not self._is_available:
            return None

        if self._api is None:
            try:
                import wandb  # type: ignore[import-untyped]

                self._api = wandb.Api()  # type: ignore[attr-defined]
                LOGGER.info("Wandb API initialized successfully")
            except Exception as exc:
                LOGGER.warning("Failed to initialize Wandb API: %s", exc)
                self._is_available = False
                return None

        return self._api

    # @lru_cache(maxsize=256) # memory-leark warnings
    def get_run_config(self, run_id: str) -> dict[str, Any] | None:
        """Fetch run configuration from Wandb API with caching.

        Args:
            run_id: Wandb run ID (format: "entity/project/run_id")

        Returns:
            Run config dict, or None if fetch fails
        """
        if not self._is_available or self.api is None:
            LOGGER.debug("Wandb API not available for run %s", run_id)
            return None

        try:
            run = self.api.run(run_id)
            # Filter out wandb internal keys
            config = {k: v for k, v in run.config.items() if not k.startswith("_")}
            LOGGER.info("Retrieved config for run %s", run_id)
            return config

        except Exception as exc:
            LOGGER.warning("Failed to fetch run config for %s: %s", run_id, exc)
            return None

    # @lru_cache(maxsize=256)
    def get_run_summary(self, run_id: str) -> dict[str, Any] | None:
        """Fetch run summary metrics from Wandb API with caching.

        Args:
            run_id: Wandb run ID (format: "entity/project/run_id")

        Returns:
            Run summary dict, or None if fetch fails
        """
        if not self._is_available or self.api is None:
            LOGGER.debug("Wandb API not available for run %s", run_id)
            return None

        try:
            run = self.api.run(run_id)
            summary = dict(run.summary)
            LOGGER.info("Retrieved summary for run %s", run_id)
            return summary

        except Exception as exc:
            LOGGER.warning("Failed to fetch run summary for %s: %s", run_id, exc)
            return None

    def get_metadata_from_wandb(
        self,
        run_id: str,
        checkpoint_path: Path,
    ) -> CheckpointMetadataV1 | None:
        """Construct checkpoint metadata from Wandb run data.

        Args:
            run_id: Wandb run ID (format: "entity/project/run_id")
            checkpoint_path: Path to checkpoint file (for metadata)

        Returns:
            Reconstructed metadata, or None if insufficient data
        """
        if not self._is_available:
            return None

        config = self.get_run_config(run_id)
        summary = self.get_run_summary(run_id)

        if config is None or summary is None:
            LOGGER.debug("Insufficient Wandb data for run %s", run_id)
            return None

        try:
            # Extract training info
            epoch = summary.get("epoch", 0)
            global_step = summary.get("global_step", summary.get("_step", 0))

            training_info = TrainingInfo(
                epoch=int(epoch) if epoch is not None else 0,
                global_step=int(global_step) if global_step is not None else 0,
                training_phase="training",
                max_epochs=config.get("trainer", {}).get("max_epochs"),
            )

            # Extract metrics info
            # Prefer test metrics, fallback to val metrics
            precision = summary.get("test/precision") or summary.get("val/precision")
            recall = summary.get("test/recall") or summary.get("val/recall")
            hmean = summary.get("test/hmean") or summary.get("val/hmean")
            val_loss = summary.get("val/loss")

            metrics_info = MetricsInfo(
                precision=float(precision) if precision is not None else None,
                recall=float(recall) if recall is not None else None,
                hmean=float(hmean) if hmean is not None else None,
                validation_loss=float(val_loss) if val_loss is not None else None,
                additional_metrics={
                    k: float(v) for k, v in summary.items() if k.startswith(("val/", "test/", "cleval/")) and isinstance(v, int | float)
                },
            )

            # Get model architecture from config
            model_cfg = config.get("model", {})
            architecture = model_cfg.get("architecture_name", "unknown")

            # Get encoder info
            encoder_cfg = model_cfg.get("encoder", {})
            encoder_name = encoder_cfg.get("model_name", "unknown")

            # Get experiment name from checkpoint path or config
            exp_name = checkpoint_path.parent.parent.name

            # Import model info types
            from .types import (
                CheckpointingConfig,
                DecoderInfo,
                EncoderInfo,
                HeadInfo,
                LossInfo,
                ModelInfo,
            )

            # Construct minimal metadata
            metadata = CheckpointMetadataV1(
                schema_version="1.0",
                checkpoint_path=str(checkpoint_path),
                exp_name=exp_name,
                created_at="1970-01-01T00:00:00",  # Placeholder - no timestamp available
                training=training_info,
                model=ModelInfo(
                    architecture=architecture,
                    encoder=EncoderInfo(
                        model_name=encoder_name,
                        pretrained=encoder_cfg.get("pretrained", True),
                        frozen=encoder_cfg.get("frozen", False),
                    ),
                    decoder=DecoderInfo(
                        name=model_cfg.get("decoder", {}).get("name", "unknown"),
                    ),
                    head=HeadInfo(
                        name=model_cfg.get("head", {}).get("name", "unknown"),
                    ),
                    loss=LossInfo(
                        name=model_cfg.get("loss", {}).get("name", "unknown"),
                    ),
                ),
                metrics=metrics_info,
                checkpointing=CheckpointingConfig(
                    monitor=config.get("callbacks", {}).get("model_checkpoint", {}).get("monitor", "val/hmean"),
                    mode=config.get("callbacks", {}).get("model_checkpoint", {}).get("mode", "max"),
                    save_top_k=config.get("callbacks", {}).get("model_checkpoint", {}).get("save_top_k", 1),
                    save_last=config.get("callbacks", {}).get("model_checkpoint", {}).get("save_last", True),
                ),
                wandb_run_id=run_id,
            )

            LOGGER.info("Constructed metadata from Wandb for run %s", run_id)
            return metadata

        except Exception as exc:
            LOGGER.warning("Failed to construct metadata from Wandb: %s", exc)
            return None

    # def clear_cache(self) -> None:
    #     """Clear all cached API responses."""
    #     self.get_run_config.cache_clear()
    #     self.get_run_summary.cache_clear()
    #     LOGGER.info("Cleared Wandb API cache")


# Singleton instance
_global_client: WandbClient | None = None


def get_wandb_client(cache_size: int = 256) -> WandbClient:
    """Get or create global Wandb client instance.

    Args:
        cache_size: Maximum number of cached API responses

    Returns:
        Wandb client singleton
    """
    global _global_client  # noqa: PLW0603

    if _global_client is None:
        _global_client = WandbClient(cache_size=cache_size)

    return _global_client


def extract_run_id_from_checkpoint(checkpoint_path: Path) -> str | None:
    """Extract Wandb run ID from checkpoint metadata or path.

    This function attempts to find the Wandb run ID by:
    1. Checking for .metadata.yaml file with wandb_run_id field
    2. Looking for run ID in .hydra/config.yaml
    3. Parsing experiment directory name for run ID pattern

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Wandb run ID if found, None otherwise
    """
    # Try loading metadata file first
    metadata_path = checkpoint_path.with_suffix(".metadata.yaml")
    if metadata_path.exists():
        try:
            import yaml

            with metadata_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            run_id = data.get("wandb_run_id")
            if run_id:
                return run_id

        except Exception as exc:
            LOGGER.debug("Failed to extract run_id from metadata: %s", exc)

    # Try Hydra config
    hydra_config = checkpoint_path.parent.parent / ".hydra" / "config.yaml"
    if hydra_config.exists():
        try:
            import yaml

            with hydra_config.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Check logger.wandb.id
            wandb_id = config.get("logger", {}).get("wandb", {}).get("id")
            if wandb_id:
                return wandb_id

        except Exception as exc:
            LOGGER.debug("Failed to extract run_id from Hydra config: %s", exc)

    # No run ID found
    return None
