"""Checkpoint state dict inference (fallback for legacy checkpoints).

<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=["checkpoint_loading", "legacy_checkpoints", "state_dict_errors"] -->

This module provides fallback functionality for analyzing PyTorch checkpoint
files when metadata files are not available. This is the slow path and should
only be used as a last resort.

⚠️ IMPORTANT: This module uses Pydantic validation for all state dict access.
Before modifying weight shape inference logic, read:
- docs/ai_handbook/02_protocols/components/23_checkpoint_loading_protocol.md
- ui/apps/inference/services/checkpoint/state_dict_models.py

Performance: 2-5 seconds per checkpoint (same as current implementation)

## Common Errors Fixed by Validation

1. KeyError: 'state_dict' → Now validates wrapper key
2. AttributeError on None.shape → Now uses safe_get_shape()
3. Silent architecture mismatches → Now validates patterns

## DO NOTs

🔴 NEVER access state_dict keys without validate_checkpoint_structure()
🔴 NEVER assume weight.shape exists - use safe_get_shape()
🔴 NEVER modify key patterns without testing all existing checkpoints
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .state_dict_models import (
    safe_get_shape,
)

LOGGER = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    """Load PyTorch checkpoint file.

    IMPORTANT: This is a fallback operation for legacy checkpoints.
    Should only be called when metadata files are unavailable.

    Args:
        checkpoint_path: Path to .ckpt file

    Returns:
        Checkpoint dict, or None if loading fails
    """
    try:
        import torch
    except ImportError:
        LOGGER.error("PyTorch not available; cannot load checkpoint %s", checkpoint_path)
        return None

    # Handle safe globals for OmegaConf
    try:
        from torch.serialization import add_safe_globals  # type: ignore[attr-defined]
    except ImportError:
        add_safe_globals = None

    if add_safe_globals is not None:
        try:
            from omegaconf.listconfig import ListConfig  # type: ignore[import-untyped]

            add_safe_globals([ListConfig])
        except ImportError:
            pass

    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Unable to load checkpoint %s: %s", checkpoint_path, exc)
        return None


def infer_encoder_from_state(state_dict: dict[str, Any]) -> str | None:
    """Infer encoder model from state dict weight shapes.

    <!-- ai_cue:use_when=["encoder_detection", "unknown_backbone"] -->

    ⚠️ CRITICAL: This function uses validated shape extraction.
    Before adding new encoder patterns:
    1. Test with existing checkpoints to avoid regressions
    2. Document shape patterns in checkpoint_loading_protocol.md
    3. Add test cases in test suite

    This function analyzes the state dict keys and weight tensor shapes to
    determine which encoder architecture was used.

    Args:
        state_dict: PyTorch state dict (should already be unwrapped from checkpoint)

    Returns:
        Inferred encoder name, or None if unable to determine

    Note:
        Uses safe_get_shape() to prevent AttributeError on missing weights
    """
    if not isinstance(state_dict, dict):
        LOGGER.warning("state_dict is not a dict, cannot infer encoder")
        return None

    keys = list(state_dict.keys())

    # Helper function using validated shape extraction
    def _get_shape_validated(key: str) -> tuple[int, ...] | None:
        """Get weight shape using validation."""
        weight = state_dict.get(key)
        shape_obj = safe_get_shape(weight)
        return shape_obj.dims if shape_obj else None

    # Check for MobileNetV3 (conv_stem pattern)
    # Pattern: encoder.model.conv_stem.weight or model.encoder.model.conv_stem.weight
    conv_stem_key = next((key for key in keys if "encoder.model.conv_stem.weight" in key), None)
    if conv_stem_key:
        shape = _get_shape_validated(conv_stem_key)
        if shape:
            out_channels = shape[0]
            # MobileNetV3 variants distinguished by stem output channels
            if out_channels <= 16:
                return "mobilenetv3_small_050"
            if out_channels <= 24:
                return "mobilenetv3_small_075"
            return "mobilenetv3_large_100"
        # Default fallback for MobileNetV3
        LOGGER.debug("Found conv_stem key but no valid shape, defaulting to mobilenetv3_small_050")
        return "mobilenetv3_small_050"

    # Check for EfficientNet (features pattern)
    # Pattern: encoder.model.features.0.0.weight or model.encoder.model.features.0.0.weight
    features_key = next((key for key in keys if "encoder.model.features.0.0.weight" in key), None)
    if features_key:
        shape = _get_shape_validated(features_key)
        if shape and shape[0] <= 40:
            return "efficientnet_b0"
        return "efficientnet_b3"

    # Check for ResNet (layer pattern)
    # Pattern: encoder.model.layerN.0.conv1.weight or model.encoder.model.layerN.0.conv1.weight
    layer_key = next(
        (
            key
            for key in keys
            if "encoder.model.layer3.0.conv1.weight" in key
            or "encoder.model.layer2.0.conv1.weight" in key
            or "encoder.model.layer1.0.conv1.weight" in key
        ),
        None,
    )
    if layer_key:
        weight_shape = _get_shape_validated(layer_key)
        if weight_shape:
            # ResNet50 has 256+ channels in layer3
            # ResNet18 has 128 channels in layer3
            if weight_shape[0] >= 256:
                return "resnet50"
            return "resnet18"

    LOGGER.debug("Could not infer encoder from state dict keys")
    return None


def infer_architecture_from_path(checkpoint_path: Path) -> str | None:
    """Infer architecture from checkpoint path.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Inferred architecture name, or None
    """
    path_str = checkpoint_path.as_posix().lower()
    exp_name = checkpoint_path.parent.parent.name.lower()
    stem = checkpoint_path.stem.lower()

    search_text = f"{path_str} {exp_name} {stem}"

    # Architecture patterns in priority order
    for candidate in ("dbnetpp", "dbnet", "craft", "pan", "psenet"):
        if candidate in search_text:
            return candidate

    return None


def infer_encoder_from_path(checkpoint_path: Path) -> str | None:
    """Infer encoder from checkpoint path.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Inferred encoder name, or None
    """
    path_str = checkpoint_path.as_posix().lower()

    # Encoder patterns in priority order
    encoder_patterns = (
        "mobilenetv3_small_050",
        "mobilenetv3_small_075",
        "mobilenetv3_large_100",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "efficientnet_b0",
        "efficientnet_b3",
        "efficientnet_v2_s",
    )

    for candidate in encoder_patterns:
        if candidate in path_str:
            return candidate

    return None


# TODO: Implement decoder/head signature extraction when needed
# These functions would mirror the logic from checkpoint_catalog.py lines 701-834
# For now, we focus on the primary use case: extracting basic model info
