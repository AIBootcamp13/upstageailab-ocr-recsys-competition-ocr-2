"""Config file resolution for checkpoint catalog.

This module handles locating and loading Hydra configuration files associated
with checkpoints.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)

_CONFIG_SUFFIXES = (".config.json", ".config.yaml", ".config.yml", ".resolved.config.json")


def resolve_config_path(
    checkpoint_path: Path,
    config_filenames: tuple[str, ...] = ("config.yaml", "hparams.yaml"),
) -> Path | None:
    """Resolve config file path for a checkpoint.

    Search order:
    1. Sidecar configs: {checkpoint}.config.yaml, {checkpoint}.resolved.config.json
    2. .hydra/config.yaml in experiment directory (parent.parent/.hydra/)
    3. Fallback to specified config_filenames in parent directories

    Args:
        checkpoint_path: Path to checkpoint file
        config_filenames: Tuple of config filenames to search for

    Returns:
        Resolved config path, or None if not found
    """
    # Check for sidecar configs
    for suffix in _CONFIG_SUFFIXES:
        sidecar_config = checkpoint_path.with_suffix(suffix)
        if sidecar_config.exists():
            return sidecar_config

    # Check .hydra directory (two levels up from checkpoint)
    hydra_candidates = [
        checkpoint_path.parent.parent / ".hydra" / "config.yaml",
        checkpoint_path.parent.parent.parent / ".hydra" / "config.yaml",
    ]

    for candidate in hydra_candidates:
        if candidate.exists():
            return candidate

    # Search parent directories for config filenames
    search_dirs = [
        checkpoint_path.parent,
        checkpoint_path.parent.parent,
    ]

    for directory in search_dirs:
        if not directory.exists():
            continue
        for filename in config_filenames:
            candidate = directory / filename
            if candidate.exists():
                return candidate

    return None


def load_config(config_path: Path | None) -> dict[str, Any] | None:
    """Load Hydra config from YAML or JSON file.

    Args:
        config_path: Path to config file (or None)

    Returns:
        Parsed config dict, or None if path is None or invalid
    """
    if config_path is None or not config_path.exists():
        return None

    try:
        with config_path.open("r", encoding="utf-8") as f:
            if config_path.suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(f)
            elif config_path.suffix == ".json":
                data = json.load(f)
            else:
                LOGGER.warning("Unknown config file format: %s", config_path)
                return None

        if not isinstance(data, dict):
            LOGGER.warning("Config file is not a dict: %s", config_path)
            return None

        return data

    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Failed to load config from %s: %s", config_path, exc)
        return None
