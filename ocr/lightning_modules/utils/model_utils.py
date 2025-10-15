from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def load_state_dict_with_fallback(
    model: torch.nn.Module, state_dict: Mapping[str, Any], strict: bool = True, remove_prefix: str = "model."
) -> tuple[list[str], list[str]]:
    """Load state dict with fallback handling for different checkpoint formats.

    This function handles loading checkpoints that may have different key prefixes
    or structures, providing fallback mechanisms for common issues.

    Args:
        model: The model to load state into
        state_dict: The state dictionary to load
        strict: Whether to strictly enforce key matching
        remove_prefix: Prefix to remove from state dict keys if present

    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    # Try loading with original keys first
    try:
        result = model.load_state_dict(state_dict, strict=strict)
        return result.missing_keys, result.unexpected_keys
    except RuntimeError as e:
        if "Missing key(s)" not in str(e) and "Unexpected key(s)" not in str(e):
            raise

    # Fallback: try removing prefix if present
    if remove_prefix:
        modified_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(remove_prefix):
                new_key = key[len(remove_prefix) :]
                modified_state_dict[new_key] = value
            else:
                modified_state_dict[key] = value

        try:
            result = model.load_state_dict(modified_state_dict, strict=strict)
            return result.missing_keys, result.unexpected_keys
        except RuntimeError:
            pass

    # Final fallback: load with strict=False
    result = model.load_state_dict(state_dict, strict=False)
    return result.missing_keys, result.unexpected_keys
