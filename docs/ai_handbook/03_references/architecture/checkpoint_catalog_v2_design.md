# Checkpoint Catalog V2: Modular Architecture Design

**Date**: 2025-10-18
**Status**: Design Phase
**Related**: [Refactor Plan](../../../../checkpoint_catalog_refactor_plan.md) | [Analysis](../../05_changelog/2025-10/18_checkpoint_catalog_analysis.md)

## Overview

This document specifies the modular architecture for Checkpoint Catalog V2, designed to achieve **40-100x performance improvement** through YAML-based metadata and lazy checkpoint loading.

---

## Design Principles

1. **YAML-First**: Metadata stored in `.metadata.yaml` files, eliminating checkpoint loading
2. **Lazy Loading**: Only load checkpoints as last resort fallback
3. **Modular Design**: Clear separation of concerns across focused modules
4. **Type Safety**: Pydantic V2 for all data validation and serialization
5. **Backward Compatible**: Support legacy checkpoints without metadata files
6. **Cacheable**: LRU cache for repeated catalog builds

---

## Module Structure

```
ui/apps/inference/services/checkpoint/
├── __init__.py                    # Public API exports
├── types.py                       # Pydantic models (YAML schema)
├── metadata_loader.py             # YAML metadata loading
├── config_resolver.py             # Config file resolution & loading
├── validator.py                   # Schema validation & compatibility
├── wandb_client.py                # Wandb API integration (future)
├── inference_engine.py            # Checkpoint state dict analysis (fallback)
├── cache.py                       # Caching layer
└── catalog.py                     # Main catalog builder (orchestration)
```

---

## Data Models (Pydantic V2)

### Core Metadata Schema

```python
# ui/apps/inference/services/checkpoint/types.py

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CheckpointMetadataV1(BaseModel):
    """
    Checkpoint metadata schema V1.

    This is the primary data structure stored in `.metadata.yaml` files
    generated during training. Designed for fast loading without checkpoint access.
    """

    # Schema versioning
    schema_version: Literal["1.0"] = Field(
        default="1.0",
        description="Metadata schema version for migration support",
    )

    # === Checkpoint Identification ===
    checkpoint_path: str = Field(
        ...,
        description="Relative path to checkpoint file from outputs directory",
    )

    exp_name: str = Field(
        ...,
        description="Experiment name (directory name containing checkpoint)",
    )

    created_at: str = Field(
        ...,
        description="ISO 8601 timestamp when checkpoint was created",
    )

    # === Training Progress ===
    training: TrainingInfo = Field(
        ...,
        description="Training progress information",
    )

    # === Model Architecture ===
    model: ModelInfo = Field(
        ...,
        description="Model architecture and component configuration",
    )

    # === Performance Metrics ===
    metrics: MetricsInfo = Field(
        ...,
        description="Validation metrics (precision, recall, hmean, loss)",
    )

    # === Checkpointing Configuration ===
    checkpointing: CheckpointingConfig = Field(
        ...,
        description="ModelCheckpoint callback configuration",
    )

    # === Optional Metadata ===
    hydra_config_path: str | None = Field(
        default=None,
        description="Relative path to Hydra config.yaml from outputs dir",
    )

    wandb_run_id: str | None = Field(
        default=None,
        description="Wandb run ID for online artifact retrieval",
    )

    @field_validator("created_at")
    @classmethod
    def validate_iso_timestamp(cls, v: str) -> str:
        """Ensure created_at is valid ISO 8601 timestamp."""
        try:
            datetime.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"Invalid ISO 8601 timestamp: {v}") from exc
        return v

    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        str_strip_whitespace = True


class TrainingInfo(BaseModel):
    """Training progress information."""

    epoch: int = Field(
        ...,
        ge=0,
        description="Current training epoch (0-indexed)",
    )

    global_step: int = Field(
        ...,
        ge=0,
        description="Global training step across all epochs",
    )

    training_phase: Literal["training", "validation", "finetuning"] = Field(
        default="training",
        description="Training phase identifier",
    )

    max_epochs: int | None = Field(
        default=None,
        ge=1,
        description="Maximum epochs configured in trainer",
    )


class ModelInfo(BaseModel):
    """Model architecture and components."""

    architecture: str = Field(
        ...,
        description="Model architecture name (e.g., 'dbnet', 'craft', 'pan')",
    )

    encoder: EncoderInfo = Field(
        ...,
        description="Encoder/backbone configuration",
    )

    decoder: DecoderInfo = Field(
        ...,
        description="Decoder configuration and signature",
    )

    head: HeadInfo = Field(
        ...,
        description="Detection head configuration and signature",
    )

    loss: LossInfo = Field(
        ...,
        description="Loss function configuration",
    )


class EncoderInfo(BaseModel):
    """Encoder/backbone information."""

    model_name: str = Field(
        ...,
        description="Encoder model name (e.g., 'resnet50', 'mobilenetv3_large_100')",
    )

    pretrained: bool = Field(
        default=True,
        description="Whether encoder uses pretrained weights",
    )

    frozen: bool = Field(
        default=False,
        description="Whether encoder weights are frozen during training",
    )


class DecoderInfo(BaseModel):
    """Decoder configuration and signature."""

    name: str = Field(
        ...,
        description="Decoder type (e.g., 'pan_decoder', 'fpn_decoder', 'unet')",
    )

    in_channels: list[int] = Field(
        default_factory=list,
        description="Input channel dimensions from encoder feature pyramid",
    )

    inner_channels: int | None = Field(
        default=None,
        ge=1,
        description="Internal feature channels (for FPN/PAN)",
    )

    output_channels: int | None = Field(
        default=None,
        ge=1,
        description="Output feature channels",
    )

    params: dict[str, int | float | bool | str] = Field(
        default_factory=dict,
        description="Additional decoder-specific parameters",
    )


class HeadInfo(BaseModel):
    """Detection head configuration and signature."""

    name: str = Field(
        ...,
        description="Head type (e.g., 'db_head', 'craft_head')",
    )

    in_channels: int | None = Field(
        default=None,
        ge=1,
        description="Input channels from decoder",
    )

    params: dict[str, int | float | bool | str] = Field(
        default_factory=dict,
        description="Head-specific parameters (e.g., box_thresh, max_candidates)",
    )


class LossInfo(BaseModel):
    """Loss function configuration."""

    name: str = Field(
        ...,
        description="Loss function name (e.g., 'db_loss', 'craft_loss')",
    )

    params: dict[str, float] = Field(
        default_factory=dict,
        description="Loss-specific parameters (e.g., alpha, beta, weights)",
    )


class MetricsInfo(BaseModel):
    """Validation metrics."""

    precision: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Validation precision score",
    )

    recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Validation recall score",
    )

    hmean: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Validation harmonic mean (F1 score)",
    )

    validation_loss: float | None = Field(
        default=None,
        ge=0.0,
        description="Validation loss value",
    )

    # Extended metrics (optional)
    additional_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Additional tracked metrics (e.g., 'cleval/precision', 'val/iou')",
    )


class CheckpointingConfig(BaseModel):
    """ModelCheckpoint callback configuration."""

    monitor: str = Field(
        ...,
        description="Metric being monitored (e.g., 'val/hmean', 'val/loss')",
    )

    mode: Literal["min", "max"] = Field(
        ...,
        description="Optimization mode for monitored metric",
    )

    save_top_k: int = Field(
        default=1,
        ge=-1,
        description="Number of best checkpoints to save (-1 = all)",
    )

    save_last: bool = Field(
        default=True,
        description="Whether to save 'last.ckpt' checkpoint",
    )


# === Lightweight Catalog Models ===

class CheckpointCatalogEntry(BaseModel):
    """
    Lightweight catalog entry for UI dropdown display.

    This is the model returned by the catalog builder for fast UI rendering.
    Includes only essential display information without full metadata.
    """

    checkpoint_path: Path = Field(
        ...,
        description="Absolute path to checkpoint file",
    )

    config_path: Path | None = Field(
        default=None,
        description="Resolved config file path for inference",
    )

    display_name: str = Field(
        ...,
        description="Human-readable display name for UI",
    )

    # Model identification
    architecture: str = Field(
        ...,
        description="Model architecture (e.g., 'dbnet')",
    )

    backbone: str = Field(
        ...,
        description="Encoder model name (e.g., 'resnet50')",
    )

    # Training info
    exp_name: str = Field(
        ...,
        description="Experiment name",
    )

    epochs: int = Field(
        ...,
        ge=0,
        description="Training epoch number",
    )

    created_timestamp: str = Field(
        ...,
        description="Creation timestamp (YYYYmmdd_HHMM format)",
    )

    # Key metrics for display
    hmean: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Harmonic mean score",
    )

    precision: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Precision score",
    )

    recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Recall score",
    )

    # Checkpointing info
    monitor: str | None = Field(
        default=None,
        description="Monitored metric",
    )

    monitor_mode: Literal["min", "max"] | None = Field(
        default=None,
        description="Monitor optimization mode",
    )

    # Metadata availability
    has_metadata: bool = Field(
        default=False,
        description="Whether checkpoint has .metadata.yaml file",
    )

    def to_display_option(self) -> str:
        """Generate formatted display string for UI dropdown."""
        parts = [self.architecture, self.backbone]

        # Add experiment name if different from architecture
        if self.exp_name and self.exp_name not in parts:
            exp_short = self.exp_name[:20] + "..." if len(self.exp_name) > 23 else self.exp_name
            parts.append(exp_short)

        model_info = " · ".join(parts)

        # Training annotations
        annotations = [f"ep{self.epochs}"]

        if self.hmean is not None:
            annotations.append(f"hmean {self.hmean:.3f}")

        if self.created_timestamp:
            annotations.append(self.created_timestamp)

        return f"{model_info} ({' • '.join(annotations)})"

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True  # For Path objects


class CheckpointCatalog(BaseModel):
    """Complete checkpoint catalog with metadata."""

    entries: list[CheckpointCatalogEntry] = Field(
        default_factory=list,
        description="List of catalog entries",
    )

    total_count: int = Field(
        default=0,
        ge=0,
        description="Total number of checkpoints found",
    )

    metadata_available_count: int = Field(
        default=0,
        ge=0,
        description="Number of checkpoints with .metadata.yaml files",
    )

    catalog_build_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to build catalog",
    )

    outputs_dir: Path = Field(
        ...,
        description="Outputs directory scanned",
    )

    @property
    def metadata_coverage_percent(self) -> float:
        """Calculate percentage of checkpoints with metadata files."""
        if self.total_count == 0:
            return 0.0
        return (self.metadata_available_count / self.total_count) * 100
```

---

## YAML Metadata File Format

### Example: `.metadata.yaml`

```yaml
# Generated automatically by MetadataCallback during training
# Schema version for migration support
schema_version: "1.0"

# Checkpoint identification
checkpoint_path: "outputs/dbnet-resnet50-pan-20251018/checkpoints/epoch=10.ckpt"
exp_name: "dbnet-resnet50-pan-20251018"
created_at: "2025-10-18T14:32:15"

# Training progress
training:
  epoch: 10
  global_step: 5420
  training_phase: "training"
  max_epochs: 50

# Model architecture
model:
  architecture: "dbnet"

  encoder:
    model_name: "resnet50"
    pretrained: true
    frozen: false

  decoder:
    name: "pan_decoder"
    in_channels: [256, 512, 1024, 2048]
    inner_channels: 256
    output_channels: 128
    params:
      smooth: true
      num_fuse_layers: 4

  head:
    name: "db_head"
    in_channels: 128
    params:
      box_thresh: 0.3
      max_candidates: 300
      thresh: 0.2
      use_polygon: true

  loss:
    name: "db_loss"
    params:
      alpha: 1.0
      beta: 10.0

# Performance metrics (required: precision, recall, hmean, epoch)
metrics:
  precision: 0.8542
  recall: 0.8321
  hmean: 0.8430
  validation_loss: 0.0234
  additional_metrics:
    cleval/precision: 0.8542
    cleval/recall: 0.8321
    cleval/hmean: 0.8430
    val/loss: 0.0234
    val/iou: 0.7234

# Checkpointing configuration
checkpointing:
  monitor: "val/hmean"
  mode: "max"
  save_top_k: 3
  save_last: true

# Optional references
hydra_config_path: "outputs/dbnet-resnet50-pan-20251018/.hydra/config.yaml"
wandb_run_id: "abc123def456"
```

### Schema Size Estimate
- **Typical file size**: 2-5 KB (vs 500MB-2GB checkpoint)
- **Load time**: 5-10ms (vs 2-5 seconds for checkpoint)
- **Speedup**: **200-500x faster** per checkpoint

---

## Module Specifications

### 1. `metadata_loader.py`

**Responsibility**: Load and validate YAML metadata files

```python
"""Metadata loader for checkpoint catalog V2."""

from pathlib import Path
import yaml
from .types import CheckpointMetadataV1


def load_metadata(checkpoint_path: Path) -> CheckpointMetadataV1 | None:
    """
    Load metadata from .metadata.yaml file adjacent to checkpoint.

    Args:
        checkpoint_path: Path to .ckpt file

    Returns:
        Parsed and validated metadata, or None if file doesn't exist

    Raises:
        ValidationError: If YAML structure is invalid
    """
    metadata_path = checkpoint_path.with_suffix(".metadata.yaml")

    if not metadata_path.exists():
        return None

    with metadata_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return CheckpointMetadataV1.model_validate(data)


def load_metadata_batch(
    checkpoint_paths: list[Path],
) -> dict[Path, CheckpointMetadataV1 | None]:
    """
    Load metadata for multiple checkpoints in batch.

    Args:
        checkpoint_paths: List of checkpoint paths

    Returns:
        Dict mapping checkpoint paths to metadata (or None if unavailable)
    """
    return {path: load_metadata(path) for path in checkpoint_paths}
```

**API Contract**:
- **Input**: Checkpoint path
- **Output**: Pydantic model or None
- **No fallbacks**: Only reads YAML files
- **Fast**: <10ms per checkpoint

---

### 2. `config_resolver.py`

**Responsibility**: Resolve and load Hydra config files

```python
"""Config file resolution for checkpoint catalog."""

from pathlib import Path
from typing import Any
import yaml


def resolve_config_path(
    checkpoint_path: Path,
    config_filenames: tuple[str, ...] = ("config.yaml", "hparams.yaml"),
) -> Path | None:
    """
    Resolve config file path for a checkpoint.

    Search order:
    1. Sidecar config: {checkpoint}.config.yaml
    2. .hydra/config.yaml in experiment directory
    3. Fallback to specified config_filenames

    Args:
        checkpoint_path: Path to checkpoint file
        config_filenames: Tuple of config filenames to search for

    Returns:
        Resolved config path, or None if not found
    """
    # Check for sidecar config
    sidecar_config = checkpoint_path.with_suffix(".config.yaml")
    if sidecar_config.exists():
        return sidecar_config

    # Check .hydra directory
    hydra_config = checkpoint_path.parent.parent / ".hydra" / "config.yaml"
    if hydra_config.exists():
        return hydra_config

    # Search parent directories
    for parent in [checkpoint_path.parent, checkpoint_path.parent.parent]:
        for filename in config_filenames:
            candidate = parent / filename
            if candidate.exists():
                return candidate

    return None


def load_config(config_path: Path | None) -> dict[str, Any] | None:
    """
    Load Hydra config from YAML file.

    Args:
        config_path: Path to config file (or None)

    Returns:
        Parsed config dict, or None if path is None or invalid
    """
    if config_path is None or not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
```

---

### 3. `validator.py`

**Responsibility**: Schema validation and compatibility checking

```python
"""Schema validation for checkpoint metadata."""

from .types import CheckpointMetadataV1, CheckpointCatalogEntry


class MetadataValidator:
    """Validator for checkpoint metadata with schema compatibility checks."""

    def __init__(self, schema_version: str = "1.0"):
        """Initialize validator with target schema version."""
        self.schema_version = schema_version

    def validate_metadata(
        self,
        metadata: CheckpointMetadataV1,
    ) -> CheckpointMetadataV1:
        """
        Validate metadata against schema.

        Args:
            metadata: Metadata to validate

        Returns:
            Validated metadata

        Raises:
            ValidationError: If metadata is invalid
        """
        # Pydantic already validates on model creation
        # Additional business logic validation here

        # Ensure required metrics are present
        if metadata.metrics.hmean is None:
            raise ValueError("hmean metric is required for catalog entry")

        if metadata.training.epoch < 0:
            raise ValueError("Epoch cannot be negative")

        return metadata

    def validate_batch(
        self,
        metadata_list: list[CheckpointMetadataV1],
    ) -> list[CheckpointMetadataV1]:
        """
        Validate multiple metadata entries.

        Args:
            metadata_list: List of metadata to validate

        Returns:
            List of validated metadata
        """
        return [self.validate_metadata(m) for m in metadata_list]
```

---

### 4. `inference_engine.py`

**Responsibility**: Checkpoint state dict analysis (fallback only)

```python
"""Checkpoint state dict inference (fallback for legacy checkpoints)."""

from pathlib import Path
from typing import Any
import torch

from .types import DecoderInfo, HeadInfo


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    """
    Load PyTorch checkpoint file.

    IMPORTANT: This is a fallback operation for legacy checkpoints.
    Should only be called when metadata files are unavailable.

    Args:
        checkpoint_path: Path to .ckpt file

    Returns:
        Checkpoint dict, or None if loading fails
    """
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None


def infer_decoder_signature(state_dict: dict[str, Any]) -> DecoderInfo:
    """
    Infer decoder signature from state dict weight shapes.

    Args:
        state_dict: PyTorch state dict

    Returns:
        Inferred decoder information
    """
    # Implementation from current _extract_state_signatures_from_checkpoint
    # Lines 701-834 from checkpoint_catalog.py
    # (Omitted for brevity - copy existing logic)
    ...


def infer_head_signature(state_dict: dict[str, Any]) -> HeadInfo:
    """
    Infer head signature from state dict weight shapes.

    Args:
        state_dict: PyTorch state dict

    Returns:
        Inferred head information
    """
    # Implementation from existing function
    ...


def infer_encoder_from_state(state_dict: dict[str, Any]) -> str | None:
    """
    Infer encoder model from state dict weight shapes.

    Args:
        state_dict: PyTorch state dict

    Returns:
        Inferred encoder name, or None
    """
    # Implementation from _infer_encoder_from_checkpoint (lines 837-893)
    ...
```

---

### 5. `cache.py`

**Responsibility**: LRU cache for catalog results

```python
"""Caching layer for checkpoint catalog."""

from functools import lru_cache
from pathlib import Path
from typing import Callable

from .types import CheckpointCatalog


class CatalogCache:
    """LRU cache for checkpoint catalog builds."""

    def __init__(self, maxsize: int = 128):
        """Initialize cache with maximum size."""
        self.maxsize = maxsize
        self._build_fn: Callable | None = None

    def get_or_build(
        self,
        outputs_dir: Path,
        cache_key: str,
        build_fn: Callable[[], CheckpointCatalog],
    ) -> CheckpointCatalog:
        """
        Get catalog from cache or build and cache it.

        Args:
            outputs_dir: Outputs directory being cataloged
            cache_key: Unique cache key (e.g., hash of outputs_dir + mtime)
            build_fn: Function to build catalog if not cached

        Returns:
            Checkpoint catalog
        """
        # Use outputs_dir mtime as cache invalidation signal
        mtime = outputs_dir.stat().st_mtime
        full_key = f"{cache_key}:{mtime}"

        cached = self._get_cached(full_key)
        if cached is not None:
            return cached

        # Build and cache
        catalog = build_fn()
        self._set_cached(full_key, catalog)
        return catalog

    @lru_cache(maxsize=128)
    def _get_cached(self, key: str) -> CheckpointCatalog | None:
        """Internal cached getter."""
        return None  # LRU handles caching

    def _set_cached(self, key: str, catalog: CheckpointCatalog) -> None:
        """Cache catalog result."""
        self._get_cached(key)  # Populate LRU cache
        self._get_cached.cache_clear()  # Clear and re-cache
        self._get_cached.__wrapped__ = lambda k: catalog if k == key else None

    def clear(self) -> None:
        """Clear all cached catalogs."""
        self._get_cached.cache_clear()
```

---

### 6. `catalog.py`

**Responsibility**: Main orchestration layer

```python
"""Checkpoint catalog builder V2."""

from pathlib import Path
from time import time

from .types import CheckpointCatalog, CheckpointCatalogEntry
from .metadata_loader import load_metadata
from .config_resolver import resolve_config_path, load_config
from .validator import MetadataValidator
from .inference_engine import load_checkpoint, infer_decoder_signature
from .cache import CatalogCache


class CheckpointCatalogBuilder:
    """Builds checkpoint catalogs with fallback hierarchy."""

    def __init__(self, outputs_dir: Path, use_cache: bool = True):
        """
        Initialize catalog builder.

        Args:
            outputs_dir: Directory containing experiment outputs
            use_cache: Whether to enable catalog caching
        """
        self.outputs_dir = outputs_dir
        self.validator = MetadataValidator()
        self.cache = CatalogCache() if use_cache else None

    def build_catalog(self) -> CheckpointCatalog:
        """
        Build complete checkpoint catalog.

        Returns:
            Checkpoint catalog with all entries
        """
        start_time = time()

        # List all checkpoints
        checkpoint_paths = sorted(self.outputs_dir.rglob("*.ckpt"))

        entries: list[CheckpointCatalogEntry] = []
        metadata_count = 0

        for ckpt_path in checkpoint_paths:
            entry = self._build_entry(ckpt_path)
            if entry and entry.epochs > 0:  # Filter out invalid checkpoints
                entries.append(entry)
                if entry.has_metadata:
                    metadata_count += 1

        # Sort by architecture, backbone, epoch
        entries.sort(
            key=lambda e: (e.architecture, e.backbone, e.epochs, e.checkpoint_path.name)
        )

        build_time = time() - start_time

        return CheckpointCatalog(
            entries=entries,
            total_count=len(entries),
            metadata_available_count=metadata_count,
            catalog_build_time_seconds=build_time,
            outputs_dir=self.outputs_dir,
        )

    def _build_entry(self, checkpoint_path: Path) -> CheckpointCatalogEntry | None:
        """
        Build catalog entry for a single checkpoint.

        Fallback hierarchy:
        1. Load .metadata.yaml (fast path)
        2. Infer from config files
        3. Load checkpoint and analyze state dict (slow path)

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Catalog entry, or None if checkpoint is invalid
        """
        # Fast path: Load metadata YAML
        metadata = load_metadata(checkpoint_path)

        if metadata is not None:
            # Validate metadata
            metadata = self.validator.validate_metadata(metadata)

            # Resolve config path
            config_path = resolve_config_path(checkpoint_path)

            return CheckpointCatalogEntry(
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                display_name=checkpoint_path.stem,
                architecture=metadata.model.architecture,
                backbone=metadata.model.encoder.model_name,
                exp_name=metadata.exp_name,
                epochs=metadata.training.epoch,
                created_timestamp=metadata.created_at[:13].replace("T", "_").replace("-", ""),
                hmean=metadata.metrics.hmean,
                precision=metadata.metrics.precision,
                recall=metadata.metrics.recall,
                monitor=metadata.checkpointing.monitor,
                monitor_mode=metadata.checkpointing.mode,
                has_metadata=True,
            )

        # Slow path: Fallback to legacy inference
        return self._build_entry_legacy(checkpoint_path)

    def _build_entry_legacy(
        self,
        checkpoint_path: Path,
    ) -> CheckpointCatalogEntry | None:
        """
        Build catalog entry for legacy checkpoint (no metadata file).

        This is the fallback path using the current implementation logic.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Catalog entry, or None if checkpoint is invalid
        """
        # Load config
        config_path = resolve_config_path(checkpoint_path)
        config_data = load_config(config_path)

        # Extract basic info from config
        architecture = "unknown"
        backbone = "unknown"
        epochs = None

        if config_data:
            model_cfg = config_data.get("model", {})
            architecture = model_cfg.get("architecture", "unknown")
            encoder_cfg = model_cfg.get("encoder", {})
            backbone = encoder_cfg.get("model_name", "unknown")

            trainer_cfg = config_data.get("trainer", {})
            epochs = trainer_cfg.get("max_epochs")

        # Load checkpoint for metrics (slow!)
        checkpoint_data = load_checkpoint(checkpoint_path)

        hmean = None
        precision = None
        recall = None

        if checkpoint_data:
            cleval_metrics = checkpoint_data.get("cleval_metrics", {})
            hmean = cleval_metrics.get("hmean")
            precision = cleval_metrics.get("precision")
            recall = cleval_metrics.get("recall")
            epochs = epochs or checkpoint_data.get("epoch")

        if epochs is None or epochs == 0:
            return None  # Invalid checkpoint

        return CheckpointCatalogEntry(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            display_name=checkpoint_path.stem,
            architecture=architecture,
            backbone=backbone,
            exp_name=checkpoint_path.parent.parent.name,
            epochs=epochs,
            created_timestamp="",
            hmean=hmean,
            precision=precision,
            recall=recall,
            monitor=None,
            monitor_mode=None,
            has_metadata=False,
        )


# Public API
def build_catalog(outputs_dir: Path, use_cache: bool = True) -> CheckpointCatalog:
    """
    Build checkpoint catalog for outputs directory.

    Args:
        outputs_dir: Directory containing experiment outputs
        use_cache: Whether to use catalog caching

    Returns:
        Complete checkpoint catalog
    """
    builder = CheckpointCatalogBuilder(outputs_dir, use_cache=use_cache)
    return builder.build_catalog()
```

---

## API Surface

### Public Exports (`__init__.py`)

```python
"""Checkpoint catalog V2 - Fast YAML-based metadata system."""

# Primary API
from .catalog import build_catalog, CheckpointCatalogBuilder

# Data models
from .types import (
    CheckpointMetadataV1,
    CheckpointCatalog,
    CheckpointCatalogEntry,
    TrainingInfo,
    ModelInfo,
    MetricsInfo,
)

# Utilities
from .metadata_loader import load_metadata, load_metadata_batch
from .config_resolver import resolve_config_path, load_config
from .validator import MetadataValidator

__all__ = [
    # Primary API
    "build_catalog",
    "CheckpointCatalogBuilder",
    # Models
    "CheckpointMetadataV1",
    "CheckpointCatalog",
    "CheckpointCatalogEntry",
    "TrainingInfo",
    "ModelInfo",
    "MetricsInfo",
    # Utilities
    "load_metadata",
    "load_metadata_batch",
    "resolve_config_path",
    "load_config",
    "MetadataValidator",
]
```

---

## Performance Targets

### Fast Path (with .metadata.yaml)
- **Per checkpoint**: <10ms
- **20 checkpoints**: <200ms (0.2 seconds)
- **Speedup vs current**: **200-500x faster**

### Slow Path (legacy fallback)
- **Per checkpoint**: 2-5 seconds (current performance)
- **20 checkpoints**: 40-100 seconds (same as current)
- **Speedup**: None (maintains current behavior)

### Mixed Scenario (50% metadata coverage)
- **20 checkpoints**: ~20 seconds
- **Speedup**: **2-5x faster** than current

---

## Migration Strategy

1. **Phase 1**: Implement modules (no behavioral changes)
2. **Phase 2**: Add MetadataCallback to training pipeline
3. **Phase 3**: Generate metadata for new training runs
4. **Phase 4**: Build conversion tool for legacy checkpoints
5. **Phase 5**: Gradual rollout with feature flags

---

## Next Steps

1. ✅ Design complete
2. ⏭️ Implement `types.py` with Pydantic models
3. Implement core modules (`metadata_loader.py`, `config_resolver.py`)
4. Implement `catalog.py` orchestration layer
5. Unit tests for all modules
6. Integration testing

---

## References

- [Analysis Document](../../05_changelog/2025-10/18_checkpoint_catalog_analysis.md)
- [Current Implementation](../../../../ui/apps/inference/services/checkpoint_catalog.py)
- [Refactor Plan](../../../../checkpoint_catalog_refactor_plan.md)
