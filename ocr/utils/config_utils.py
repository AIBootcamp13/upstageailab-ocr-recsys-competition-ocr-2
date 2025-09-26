"""Hydra configuration schemas and validation utilities for modular OCR architectures."""

from pathlib import Path
from typing import Any

import yaml

from ocr.models.core import registry

CONFIG_ROOT = Path(__file__).parent.parent.parent / "configs"
SCHEMA_DIR = CONFIG_ROOT / "schemas"


def validate_model_config(config: dict[str, Any]) -> list[str]:
    """Validate model configuration for modular architecture compatibility.

    Args:
        config: Model configuration dictionary

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    # Check required keys
    for key in ["encoder", "decoder", "head", "loss"]:
        if key not in config:
            errors.append(f"Missing required key: {key}")
        elif config[key] not in registry.list_encoders() + registry.list_decoders() + registry.list_heads() + registry.list_losses():
            errors.append(f"Unknown component name for {key}: {config[key]}")
    # Check optimizer and scheduler
    if "optimizer" not in config:
        errors.append("Missing optimizer configuration")
    return errors


def load_config_schema(schema_name: str) -> dict[str, Any]:
    """Load a configuration schema YAML file for validation or UI rendering."""
    schema_path = SCHEMA_DIR / f"{schema_name}.yaml"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    with open(schema_path) as f:
        return yaml.safe_load(f)


def get_default_config() -> dict[str, Any]:
    """Load the default configuration for training/inference."""
    default_path = CONFIG_ROOT / "defaults.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")
    with open(default_path) as f:
        return yaml.safe_load(f)
