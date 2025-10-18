"""Checkpoint catalog V2 - Fast YAML-based metadata system.

This module provides a high-performance checkpoint catalog system that eliminates
the need to load PyTorch checkpoint files by using YAML metadata files generated
during training, with Wandb API fallback when metadata is unavailable.

Performance:
    - Fast path (with .metadata.yaml): <10ms per checkpoint
    - Wandb fallback (with run ID): ~100-500ms per checkpoint (cached)
    - Legacy fallback: 2-5 seconds per checkpoint (current behavior)
    - Expected speedup: 40-100x for full metadata coverage

Key Features:
    - YAML-first metadata loading
    - Wandb API fallback for missing metadata
    - Pydantic V2 validation
    - Modular architecture
    - LRU caching
    - Backward compatible with legacy checkpoints

Usage:
    >>> from ui.apps.inference.services.checkpoint import build_catalog
    >>> catalog = build_catalog(outputs_dir=Path("outputs"))
    >>> print(f"Found {catalog.total_count} checkpoints")
    >>> print(f"Metadata coverage: {catalog.metadata_coverage_percent:.1f}%")
"""

from __future__ import annotations

# Primary API
from .catalog import CheckpointCatalogBuilder, build_catalog

# Utilities
from .config_resolver import load_config, resolve_config_path
from .metadata_loader import load_metadata, load_metadata_batch

# Data models
from .types import (
    CheckpointCatalog,
    CheckpointCatalogEntry,
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
from .validator import MetadataValidator
from .wandb_client import WandbClient, extract_run_id_from_checkpoint, get_wandb_client

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
    "EncoderInfo",
    "DecoderInfo",
    "HeadInfo",
    "LossInfo",
    "CheckpointingConfig",
    # Utilities
    "load_metadata",
    "load_metadata_batch",
    "resolve_config_path",
    "load_config",
    "MetadataValidator",
    # Wandb integration
    "WandbClient",
    "get_wandb_client",
    "extract_run_id_from_checkpoint",
]

__version__ = "2.0.0"
