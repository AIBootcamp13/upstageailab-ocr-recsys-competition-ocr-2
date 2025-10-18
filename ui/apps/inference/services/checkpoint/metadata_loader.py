"""Metadata loader for checkpoint catalog V2.

This module handles loading and parsing of `.metadata.yaml` files generated
during training. This is the fast path for catalog building.

Performance: ~5-10ms per checkpoint (vs 2-5 seconds for checkpoint loading)
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from .types import CheckpointMetadataV1

LOGGER = logging.getLogger(__name__)


def load_metadata(checkpoint_path: Path) -> CheckpointMetadataV1 | None:
    """Load metadata from .metadata.yaml file adjacent to checkpoint.

    Args:
        checkpoint_path: Path to .ckpt file

    Returns:
        Parsed and validated metadata, or None if file doesn't exist or is invalid

    Raises:
        ValidationError: If YAML structure is invalid (caught and logged)
    """
    metadata_path = checkpoint_path.with_suffix(".metadata.yaml")

    if not metadata_path.exists():
        return None

    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            LOGGER.warning("Metadata file is not a dict: %s", metadata_path)
            return None

        return CheckpointMetadataV1.model_validate(data)

    except ValidationError as exc:
        LOGGER.warning("Invalid metadata file %s: %s", metadata_path, exc)
        return None

    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Failed to load metadata from %s: %s", metadata_path, exc)
        return None


def load_metadata_batch(
    checkpoint_paths: list[Path],
) -> dict[Path, CheckpointMetadataV1 | None]:
    """Load metadata for multiple checkpoints in batch.

    Args:
        checkpoint_paths: List of checkpoint paths

    Returns:
        Dict mapping checkpoint paths to metadata (or None if unavailable)
    """
    return {path: load_metadata(path) for path in checkpoint_paths}


def save_metadata(
    metadata: CheckpointMetadataV1,
    checkpoint_path: Path,
) -> Path:
    """Save metadata to .metadata.yaml file adjacent to checkpoint.

    Args:
        metadata: Metadata to save
        checkpoint_path: Path to .ckpt file

    Returns:
        Path to saved metadata file

    Raises:
        OSError: If file write fails
    """
    metadata_path = checkpoint_path.with_suffix(".metadata.yaml")

    # Convert to dict and write as YAML
    data = metadata.model_dump(mode="python", exclude_none=True)

    with metadata_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    LOGGER.info("Saved metadata to %s", metadata_path)
    return metadata_path
