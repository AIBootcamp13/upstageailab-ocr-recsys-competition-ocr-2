from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def load_state_dict_with_fallback(
    model: torch.nn.Module, state_dict: Mapping[str, Any], strict: bool = True, remove_prefix: str = "model.", _recursion_depth: int = 0
) -> tuple[list[str], list[str]]:
    """Load state dict with fallback handling for different checkpoint formats.

    This function handles loading checkpoints that may have different key prefixes
    or structures, providing fallback mechanisms for common issues.

    Args:
        model: The model to load state into
        state_dict: The state dictionary to load
        strict: Whether to strictly enforce key matching
        remove_prefix: Prefix to remove from state dict keys if present
        _recursion_depth: Internal parameter to prevent infinite recursion

    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    # Prevent infinite recursion
    if _recursion_depth > 1:
        raise RuntimeError("Maximum recursion depth exceeded in load_state_dict_with_fallback")

    # Try loading with original keys first
    try:
        result = model.load_state_dict(state_dict, strict=strict)
        return result.missing_keys, result.unexpected_keys
    except RuntimeError as e:
        error_str = str(e)
        # Only retry for missing/unexpected key errors, not other RuntimeErrors
        if "Missing key(s)" not in error_str and "Unexpected key(s)" not in error_str:
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
        except RuntimeError as e:
            error_str = str(e)
            # Only retry for missing/unexpected key errors
            if "Missing key(s)" not in error_str and "Unexpected key(s)" not in error_str:
                raise

    # Fallback: handle compiled model checkpoints with _orig_mod prefix
    modified_state_dict = {}
    for key, value in state_dict.items():
        # Remove _orig_mod prefix that gets added by torch.compile
        if "._orig_mod." in key:
            new_key = key.replace("._orig_mod.", ".")
            modified_state_dict[new_key] = value
        else:
            modified_state_dict[key] = value

    if modified_state_dict != dict(state_dict):  # Only try if we actually modified keys
        try:
            result = model.load_state_dict(modified_state_dict, strict=strict)
            return result.missing_keys, result.unexpected_keys
        except RuntimeError as e:
            error_str = str(e)
            # Only retry for missing/unexpected key errors
            if "Missing key(s)" not in error_str and "Unexpected key(s)" not in error_str:
                raise

    # Final fallback: load with strict=False
    try:
        result = model.load_state_dict(state_dict, strict=False)
        return result.missing_keys, result.unexpected_keys
    except RuntimeError as e:
        # If even strict=False fails with a non-key-related error, re-raise it
        error_str = str(e)
        if "Missing key(s)" not in error_str and "Unexpected key(s)" not in error_str:
            raise
        # If it's still a key-related error even with strict=False, return empty lists
        return [], []
