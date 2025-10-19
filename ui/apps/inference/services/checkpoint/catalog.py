"""Checkpoint catalog builder V2.

This module provides the main orchestration for building checkpoint catalogs
using a fallback hierarchy: YAML metadata → Wandb API → config inference → checkpoint loading.

Performance targets:
    - Fast path (with .metadata.yaml): <10ms per checkpoint
    - Wandb fallback (with run ID): ~100-500ms per checkpoint (cached)
    - Legacy fallback: 2-5 seconds per checkpoint (current behavior)
    - Expected speedup: 40-100x for full metadata coverage
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any

from .cache import CatalogCache, get_global_cache
from .config_resolver import load_config, resolve_config_path
from .inference_engine import (
    infer_architecture_from_path,
    infer_encoder_from_path,
    infer_encoder_from_state,
    load_checkpoint,
)
from .metadata_loader import load_metadata
from .types import CheckpointCatalog, CheckpointCatalogEntry
from .validator import MetadataValidator
from .wandb_client import WandbClient, extract_run_id_from_checkpoint, get_wandb_client

LOGGER = logging.getLogger(__name__)


class CheckpointCatalogBuilder:
    """Builds checkpoint catalogs with fallback hierarchy.

    Fallback order:
        1. Fast path: Load .metadata.yaml file (~5-10ms)
        2. Wandb fallback: Fetch from Wandb API if run ID available (~100-500ms, cached)
        3. Medium path: Infer from config files + path patterns (~50-100ms)
        4. Slow path: Load PyTorch checkpoint and analyze state dict (~2-5s)
    """

    def __init__(
        self,
        outputs_dir: Path,
        use_cache: bool = True,
        use_wandb_fallback: bool = True,
        config_filenames: tuple[str, ...] = ("config.yaml", "hparams.yaml"),
    ):
        """Initialize catalog builder.

        Args:
            outputs_dir: Directory containing experiment outputs
            use_cache: Whether to enable catalog caching
            use_wandb_fallback: Whether to use Wandb API as fallback
            config_filenames: Config filenames to search for
        """
        self.outputs_dir = outputs_dir
        self.config_filenames = config_filenames
        self.validator = MetadataValidator()
        self.use_cache = use_cache
        self.use_wandb_fallback = use_wandb_fallback

        self.cache: CatalogCache | None
        if use_cache:
            self.cache = get_global_cache()
        else:
            self.cache = None

        # Initialize Wandb client if enabled
        self.wandb_client: WandbClient | None
        if use_wandb_fallback:
            self.wandb_client = get_wandb_client()
        else:
            self.wandb_client = None

    def build_catalog(self) -> CheckpointCatalog:
        """Build complete checkpoint catalog.

        Returns:
            Checkpoint catalog with all entries
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(self.outputs_dir)
            if cached is not None:
                return cached

        start_time = time()

        # List all checkpoints
        if not self.outputs_dir.exists():
            LOGGER.info("Outputs directory not found: %s", self.outputs_dir)
            return CheckpointCatalog(
                entries=[],
                total_count=0,
                metadata_available_count=0,
                catalog_build_time_seconds=0.0,
                outputs_dir=self.outputs_dir,
            )

        checkpoint_paths = sorted(self.outputs_dir.rglob("*.ckpt"))
        LOGGER.info("Found %d checkpoint files in %s", len(checkpoint_paths), self.outputs_dir)

        entries: list[CheckpointCatalogEntry] = []
        metadata_count = 0
        fast_path_count = 0
        wandb_path_count = 0
        slow_path_count = 0

        for ckpt_path in checkpoint_paths:
            entry, path_type = self._build_entry(ckpt_path)

            if entry and entry.epochs > 0:  # Filter out invalid checkpoints
                entries.append(entry)

                if path_type == "metadata":
                    metadata_count += 1
                    fast_path_count += 1
                elif path_type == "wandb":
                    metadata_count += 1
                    wandb_path_count += 1
                else:
                    slow_path_count += 1

        # Sort by architecture, backbone, epoch, filename
        entries.sort(key=lambda e: (e.architecture, e.backbone, e.epochs, e.checkpoint_path.name))

        build_time = time() - start_time

        LOGGER.info(
            "Catalog built: %d entries (%d YAML, %d Wandb, %d legacy) in %.3fs",
            len(entries),
            fast_path_count,
            wandb_path_count,
            slow_path_count,
            build_time,
        )

        catalog = CheckpointCatalog(
            entries=entries,
            total_count=len(entries),
            metadata_available_count=metadata_count,
            catalog_build_time_seconds=build_time,
            outputs_dir=self.outputs_dir,
        )

        # Cache result
        if self.cache:
            self.cache.set(self.outputs_dir, catalog)

        return catalog

    def _build_entry(
        self,
        checkpoint_path: Path,
    ) -> tuple[CheckpointCatalogEntry | None, str]:
        """Build catalog entry for a single checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Tuple of (catalog entry or None, path type: "metadata", "wandb", or "legacy")
        """
        # Fast path: Load metadata YAML
        metadata = load_metadata(checkpoint_path)

        if metadata is not None:
            try:
                # Validate metadata
                metadata = self.validator.validate_metadata(metadata)

                # Resolve config path
                config_path = resolve_config_path(checkpoint_path, self.config_filenames)

                # Format timestamp for display
                created_timestamp = self._format_timestamp(metadata.created_at)

                entry = CheckpointCatalogEntry(
                    checkpoint_path=checkpoint_path,
                    config_path=config_path,
                    display_name=checkpoint_path.stem,
                    architecture=metadata.model.architecture,
                    backbone=metadata.model.encoder.model_name,
                    exp_name=metadata.exp_name,
                    epochs=metadata.training.epoch,
                    created_timestamp=created_timestamp,
                    hmean=metadata.metrics.hmean,
                    precision=metadata.metrics.precision,
                    recall=metadata.metrics.recall,
                    monitor=metadata.checkpointing.monitor,
                    monitor_mode=metadata.checkpointing.mode,
                    has_metadata=True,
                )

                return entry, "metadata"

            except (ValueError, Exception) as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Metadata validation failed for %s: %s. Falling back to next option.",
                    checkpoint_path,
                    exc,
                )

        # Wandb fallback: Try to fetch from Wandb API
        if self.use_wandb_fallback and self.wandb_client:
            run_id = extract_run_id_from_checkpoint(checkpoint_path)

            if run_id:
                LOGGER.debug("Attempting Wandb fallback for %s (run_id=%s)", checkpoint_path, run_id)
                wandb_metadata = self.wandb_client.get_metadata_from_wandb(run_id, checkpoint_path)

                if wandb_metadata is not None:
                    try:
                        # Validate wandb metadata
                        wandb_metadata = self.validator.validate_metadata(wandb_metadata)

                        # Resolve config path
                        config_path = resolve_config_path(checkpoint_path, self.config_filenames)

                        # Format timestamp for display
                        created_timestamp = self._format_timestamp(wandb_metadata.created_at)

                        entry = CheckpointCatalogEntry(
                            checkpoint_path=checkpoint_path,
                            config_path=config_path,
                            display_name=checkpoint_path.stem,
                            architecture=wandb_metadata.model.architecture,
                            backbone=wandb_metadata.model.encoder.model_name,
                            exp_name=wandb_metadata.exp_name,
                            epochs=wandb_metadata.training.epoch,
                            created_timestamp=created_timestamp,
                            hmean=wandb_metadata.metrics.hmean,
                            precision=wandb_metadata.metrics.precision,
                            recall=wandb_metadata.metrics.recall,
                            monitor=wandb_metadata.checkpointing.monitor,
                            monitor_mode=wandb_metadata.checkpointing.mode,
                            has_metadata=True,
                        )

                        LOGGER.info("Successfully loaded metadata from Wandb for %s", checkpoint_path)
                        return entry, "wandb"

                    except (ValueError, Exception) as exc:  # noqa: BLE001
                        LOGGER.warning(
                            "Wandb metadata validation failed for %s: %s. Falling back to legacy path.",
                            checkpoint_path,
                            exc,
                        )

        # Slow path: Fallback to legacy inference
        legacy_entry = self._build_entry_legacy(checkpoint_path)
        return legacy_entry, "legacy"

    def _build_entry_legacy(
        self,
        checkpoint_path: Path,
    ) -> CheckpointCatalogEntry | None:
        """Build catalog entry for legacy checkpoint (no metadata file).

        This is the fallback path using inference from configs and checkpoint loading.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Catalog entry, or None if checkpoint is invalid
        """
        # Load config
        config_path = resolve_config_path(checkpoint_path, self.config_filenames)
        config_data = load_config(config_path)

        # Extract basic info from config
        architecture = "unknown"
        backbone = "unknown"
        epochs = None
        hmean = None
        precision = None
        recall = None
        max_epochs_from_config = None

        if config_data:
            model_cfg = config_data.get("model", {})
            architecture = model_cfg.get("architectures") or model_cfg.get("architecture", "unknown")

            encoder_cfg = model_cfg.get("encoder", {})
            backbone = encoder_cfg.get("model_name", "unknown")

            trainer_cfg = config_data.get("trainer", {})
            max_epochs_from_config = trainer_cfg.get("max_epochs")

        # Try to infer from path if config didn't help
        if architecture == "unknown":
            path_arch = infer_architecture_from_path(checkpoint_path)
            if path_arch:
                architecture = path_arch

        if backbone == "unknown":
            path_encoder = infer_encoder_from_path(checkpoint_path)
            if path_encoder:
                backbone = path_encoder

        # Load checkpoint for metrics and actual epoch (slow!)
        checkpoint_data = load_checkpoint(checkpoint_path)

        if checkpoint_data:
            # Try to get metrics from checkpoint
            cleval_metrics = checkpoint_data.get("cleval_metrics", {})
            if isinstance(cleval_metrics, dict):
                hmean = self._maybe_float(cleval_metrics.get("hmean"))
                precision = self._maybe_float(cleval_metrics.get("precision"))
                recall = self._maybe_float(cleval_metrics.get("recall"))

            # Get epoch from checkpoint (prioritize over config max_epochs)
            epochs = checkpoint_data.get("epoch")

            # Infer encoder from state dict if still unknown
            if backbone == "unknown":
                state_dict = checkpoint_data.get("state_dict")
                if state_dict:
                    inferred_encoder = infer_encoder_from_state(state_dict)
                    if inferred_encoder:
                        backbone = inferred_encoder

        # Fall back to config max_epochs only if checkpoint didn't have epoch
        if epochs is None and max_epochs_from_config is not None:
            epochs = max_epochs_from_config

        # Extract epoch from filename if still not found
        if epochs is None:
            epochs = self._extract_epoch_from_filename(checkpoint_path.stem)

        # Filter out invalid checkpoints
        if epochs is None or epochs == 0:
            LOGGER.debug("Skipping invalid checkpoint: %s (epoch=%s)", checkpoint_path, epochs)
            return None

        # Infer experiment name
        exp_name = checkpoint_path.parent.parent.name

        # Try to get creation timestamp from file stats
        created_timestamp = self._get_file_timestamp(checkpoint_path)

        return CheckpointCatalogEntry(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            display_name=checkpoint_path.stem,
            architecture=architecture,
            backbone=backbone,
            exp_name=exp_name,
            epochs=epochs,
            created_timestamp=created_timestamp,
            hmean=hmean,
            precision=precision,
            recall=recall,
            monitor=None,
            monitor_mode=None,
            has_metadata=False,
        )

    @staticmethod
    def _format_timestamp(iso_timestamp: str) -> str:
        """Format ISO timestamp for display.

        Args:
            iso_timestamp: ISO 8601 timestamp

        Returns:
            Formatted timestamp (YYYYmmdd_HHMM)
        """
        try:
            dt = datetime.fromisoformat(iso_timestamp)
            return dt.strftime("%Y%m%d_%H%M")
        except (ValueError, AttributeError):
            return ""

    @staticmethod
    def _get_file_timestamp(checkpoint_path: Path) -> str:
        """Get file creation/modification timestamp.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Formatted timestamp (YYYYmmdd_HHMM)
        """
        try:
            stat = checkpoint_path.stat()
            timestamp = getattr(stat, "st_birthtime", stat.st_mtime)
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y%m%d_%H%M")
        except (OSError, ValueError):
            return ""

    @staticmethod
    def _extract_epoch_from_filename(filename: str) -> int | None:
        """Extract epoch number from checkpoint filename.

        Args:
            filename: Checkpoint filename (without extension)

        Returns:
            Epoch number, or None if not found
        """
        import re

        # Special cases
        if filename.startswith("best"):
            return 999
        if filename.startswith("last"):
            return 998

        # Try to match epoch pattern
        pattern = re.compile(r"epoch[=\-_](?P<epoch>\d+)")
        match = pattern.search(filename)

        if match:
            try:
                return int(match.group("epoch"))
            except ValueError:
                return None

        return None

    @staticmethod
    def _maybe_float(value: Any) -> float | None:
        """Convert value to float if possible.

        Args:
            value: Value to convert

        Returns:
            Float value, or None if conversion fails
        """
        if value is None:
            return None

        if isinstance(value, int | float):
            return float(value)

        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:  # noqa: BLE001
                return None

        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None

        return None


# Public API
def build_catalog(
    outputs_dir: Path,
    use_cache: bool = True,
    use_wandb_fallback: bool = True,
    config_filenames: tuple[str, ...] = ("config.yaml", "hparams.yaml"),
) -> CheckpointCatalog:
    """Build checkpoint catalog for outputs directory.

    Args:
        outputs_dir: Directory containing experiment outputs
        use_cache: Whether to use catalog caching
        use_wandb_fallback: Whether to use Wandb API as fallback when metadata unavailable
        config_filenames: Config filenames to search for

    Returns:
        Complete checkpoint catalog

    Example:
        >>> from pathlib import Path
        >>> catalog = build_catalog(Path("outputs"))
        >>> print(f"Found {catalog.total_count} checkpoints")
        >>> print(f"Metadata coverage: {catalog.metadata_coverage_percent:.1f}%")
        >>> for entry in catalog.entries:
        ...     print(entry.to_display_option())
    """
    builder = CheckpointCatalogBuilder(
        outputs_dir,
        use_cache=use_cache,
        use_wandb_fallback=use_wandb_fallback,
        config_filenames=config_filenames,
    )
    return builder.build_catalog()
