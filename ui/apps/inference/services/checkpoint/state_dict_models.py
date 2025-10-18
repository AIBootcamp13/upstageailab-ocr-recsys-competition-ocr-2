"""Pydantic models for PyTorch checkpoint state_dict validation.

<!-- ai_cue:priority=critical -->
<!-- ai_cue:use_when=["checkpoint_loading", "state_dict_analysis", "debugging_load_errors"] -->

⚠️ CRITICAL: State Dict Structure Validation
============================================

This module provides Pydantic models for validating PyTorch checkpoint structure.
These models prevent the most common checkpoint loading errors:

1. **Missing state_dict key**: Checkpoint might use 'model_state_dict' or be a raw state dict
2. **Unknown prefix patterns**: Models may use 'model.' prefix or no prefix
3. **Shape inference errors**: Weight shapes can be tensors, arrays, or None
4. **Key pattern changes**: Decoder/head architectures have varied key naming

## Common Confusion Points (AI Cues)

### 🔴 DO NOT modify these signatures without updating:
- `inference_engine.py` (legacy inference)
- `checkpoint_catalog.py` (old catalog)
- Load model functions in training pipeline
- UI inference components

### 🟡 When debugging "load_state_dict" errors:
1. Check if state_dict has 'model.' prefix (use has_model_prefix())
2. Verify decoder/head key patterns match expected architecture
3. Use validated models instead of raw dict access
4. Check weight shapes with safe_get_shape() utility

### 🟢 Safe patterns to follow:
- Always validate state_dict with StateDict model first
- Use typed accessors (get_decoder_weight, get_head_weight)
- Log validation errors before falling back
- Never assume key existence - use .get() with defaults

## Architecture

State dict structure varies by model:

```
# Standard structure
{
  "state_dict": {
    "model.encoder.model.layer1.0.weight": Tensor(...),
    "model.decoder.lateral_convs.0.0.weight": Tensor(...),
    "model.head.binarize.0.weight": Tensor(...),
  },
  "epoch": 10,
  "global_step": 5000,
  ...
}

# Alternative structures
{
  "model_state_dict": {...},  # May use 'model_state_dict' instead
}

# Raw state dict (no wrapper)
{
  "encoder.model.layer1.0.weight": Tensor(...),  # No 'model.' prefix
  ...
}
```

## Performance Notes

- Validation is fast (~1-2ms per checkpoint)
- Use cached validators for repeated checks
- Weight shape extraction defers to torch/numpy inspection
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

LOGGER = logging.getLogger(__name__)


class WeightShape(BaseModel):
    """Validated weight tensor shape.

    <!-- ai_cue:use_when=["shape_extraction", "debugging_tensor_shapes"] -->

    ⚠️ Common Issues:
    - Weights might be None (checkpoint corruption)
    - Shape might not have .shape attribute (raw lists)
    - Tensor might be on different device (causes errors on access)

    Always use safe_get_shape() utility instead of direct .shape access.
    """

    dims: tuple[int, ...] = Field(
        ...,
        description="Tensor dimensions (e.g., [out_channels, in_channels, kernel_h, kernel_w])",
    )

    out_channels: int | None = Field(
        default=None,
        description="Output channels (first dimension)",
    )

    in_channels: int | None = Field(
        default=None,
        description="Input channels (second dimension)",
    )

    @model_validator(mode="after")
    def extract_channels(self) -> WeightShape:
        """Extract channel dimensions from shape tuple."""
        if len(self.dims) >= 1:
            self.out_channels = self.dims[0]
        if len(self.dims) >= 2:
            self.in_channels = self.dims[1]
        return self


class DecoderKeyPattern(BaseModel):
    """Decoder weight key patterns for architecture detection.

    <!-- ai_cue:use_when=["decoder_detection", "unknown_architecture"] -->

    🔴 DO NOT MODIFY these patterns without:
    1. Checking all existing checkpoints still load
    2. Updating inference_engine.py fallback logic
    3. Adding migration path for old checkpoints
    4. Documenting in CHANGELOG.md

    Decoder types:
    - pan_decoder: Uses 'bottom_up' and 'lateral_convs'
    - fpn_decoder: Uses 'fusion' and 'lateral_convs'
    - unet: Uses 'inners' and 'outers'
    """

    decoder_type: Literal["pan_decoder", "fpn_decoder", "unet", "unknown"] = Field(
        ...,
        description="Detected decoder architecture type",
    )

    has_bottom_up: bool = Field(
        default=False,
        description="Has 'bottom_up' layers (indicates PAN)",
    )

    has_fusion: bool = Field(
        default=False,
        description="Has 'fusion' layers (indicates FPN)",
    )

    has_inners: bool = Field(
        default=False,
        description="Has 'inners' layers (indicates UNet)",
    )

    prefix: str = Field(
        default="",
        description="Key prefix ('model.decoder.' or 'decoder.')",
    )

    @model_validator(mode="after")
    def detect_architecture(self) -> DecoderKeyPattern:
        """Detect decoder type from key patterns."""
        if self.has_bottom_up:
            self.decoder_type = "pan_decoder"
        elif self.has_fusion:
            self.decoder_type = "fpn_decoder"
        elif self.has_inners:
            self.decoder_type = "unet"
        else:
            self.decoder_type = "unknown"
        return self


class HeadKeyPattern(BaseModel):
    """Head weight key patterns for architecture detection.

    <!-- ai_cue:use_when=["head_detection", "output_layer_issues"] -->

    Head types:
    - db_head: Uses 'binarize' layer
    - craft_head: Uses 'craft' layer
    """

    head_type: Literal["db_head", "craft_head", "unknown"] = Field(
        ...,
        description="Detected head architecture type",
    )

    has_binarize: bool = Field(
        default=False,
        description="Has 'binarize' layer (indicates DB head)",
    )

    has_craft: bool = Field(
        default=False,
        description="Has 'craft' layer (indicates CRAFT head)",
    )

    prefix: str = Field(
        default="",
        description="Key prefix ('model.head.' or 'head.')",
    )

    @model_validator(mode="after")
    def detect_architecture(self) -> HeadKeyPattern:
        """Detect head type from key patterns."""
        if self.has_binarize:
            self.head_type = "db_head"
        elif self.has_craft:
            self.head_type = "craft_head"
        else:
            self.head_type = "unknown"
        return self


class StateDictStructure(BaseModel):
    """Validated PyTorch checkpoint state_dict structure.

    <!-- ai_cue:priority=critical -->
    <!-- ai_cue:use_when=["checkpoint_loading", "load_state_dict_errors"] -->

    ⚠️ CRITICAL PATTERNS TO FOLLOW:

    1. **Always check has_wrapper first**:
       ```python
       state_dict_obj = StateDictStructure.from_checkpoint(ckpt_data)
       raw_dict = state_dict_obj.get_raw_state_dict()
       ```

    2. **Never assume key existence**:
       ```python
       # ❌ BAD
       weight = state_dict["model.encoder.weight"]

       # ✅ GOOD
       weight = state_dict_obj.get_weight("encoder.weight")
       ```

    3. **Check prefix before constructing keys**:
       ```python
       prefix = "model." if state_dict_obj.has_model_prefix else ""
       key = f"{prefix}encoder.layer1.weight"
       ```

    ## Validation Rules

    - Checkpoint must be a dict
    - State dict must exist (as 'state_dict', 'model_state_dict', or root)
    - All keys must be strings
    - All values should be tensors (but validated separately)
    """

    has_wrapper: bool = Field(
        ...,
        description="Whether state_dict is wrapped in 'state_dict' or 'model_state_dict' key",
    )

    wrapper_key: Literal["state_dict", "model_state_dict", None] = Field(
        default=None,
        description="The wrapper key used, if any",
    )

    has_model_prefix: bool = Field(
        default=False,
        description="Whether state dict keys use 'model.' prefix",
    )

    keys: list[str] = Field(
        default_factory=list,
        description="All state dict keys for validation",
    )

    decoder_pattern: DecoderKeyPattern | None = Field(
        default=None,
        description="Detected decoder architecture patterns",
    )

    head_pattern: HeadKeyPattern | None = Field(
        default=None,
        description="Detected head architecture patterns",
    )

    @field_validator("keys")
    @classmethod
    def validate_keys(cls, v: list[str]) -> list[str]:
        """Ensure all keys are strings."""
        if not all(isinstance(k, str) for k in v):
            raise ValueError("All state dict keys must be strings")
        return v

    @model_validator(mode="after")
    def detect_patterns(self) -> StateDictStructure:
        """Detect architecture patterns from keys."""
        # Detect model prefix
        self.has_model_prefix = any(k.startswith("model.") for k in self.keys)

        # Detect decoder pattern
        prefix = "model.decoder." if self.has_model_prefix else "decoder."

        decoder_pattern = DecoderKeyPattern(
            decoder_type="unknown",
            has_bottom_up=any(k.startswith(f"{prefix}bottom_up") for k in self.keys),
            has_fusion=any(k.startswith(f"{prefix}fusion") for k in self.keys),
            has_inners=any(k.startswith(f"{prefix}inners") for k in self.keys),
            prefix=prefix,
        )
        self.decoder_pattern = decoder_pattern

        # Detect head pattern
        head_prefix = "model.head." if self.has_model_prefix else "head."

        head_pattern = HeadKeyPattern(
            head_type="unknown",
            has_binarize=any(k.startswith(f"{head_prefix}binarize") for k in self.keys),
            has_craft=any(k.startswith(f"{head_prefix}craft") for k in self.keys),
            prefix=head_prefix,
        )
        self.head_pattern = head_pattern

        return self

    @classmethod
    def from_checkpoint(cls, checkpoint_data: dict[str, Any]) -> StateDictStructure:
        """Create from raw checkpoint data.

        <!-- ai_cue:use_when=["checkpoint_loading"] -->

        Args:
            checkpoint_data: Raw checkpoint dictionary from torch.load()

        Returns:
            Validated state dict structure

        Raises:
            ValueError: If checkpoint structure is invalid
        """
        if not isinstance(checkpoint_data, dict):
            raise ValueError(f"Checkpoint must be dict, got {type(checkpoint_data)}")

        # Try standard wrappers first
        state_dict = checkpoint_data.get("state_dict")
        wrapper_key: Literal["state_dict", "model_state_dict", None] = "state_dict"

        if state_dict is None:
            state_dict = checkpoint_data.get("model_state_dict")
            wrapper_key = "model_state_dict"

        # Check if it's a raw state dict (all values are tensors/numbers)
        if state_dict is None:
            # Heuristic: if all values are numeric/tensors, it's a raw state dict
            if all(isinstance(v, int | float) or hasattr(v, "shape") or hasattr(v, "__array__") for v in checkpoint_data.values()):
                state_dict = checkpoint_data
                wrapper_key = None

        if state_dict is None:
            raise ValueError("No state_dict found in checkpoint (tried 'state_dict', 'model_state_dict', raw)")

        if not isinstance(state_dict, dict):
            raise ValueError(f"State dict must be dict, got {type(state_dict)}")

        return cls(
            has_wrapper=wrapper_key is not None,
            wrapper_key=wrapper_key,
            keys=list(state_dict.keys()),
        )

    def get_raw_state_dict(self, checkpoint_data: dict[str, Any]) -> dict[str, Any]:
        """Get the raw state dict from checkpoint data.

        Args:
            checkpoint_data: Raw checkpoint data

        Returns:
            Raw state dict (unwrapped)
        """
        if not self.has_wrapper:
            return checkpoint_data

        if self.wrapper_key:
            return checkpoint_data.get(self.wrapper_key, {})

        return {}


# Utility functions for safe state dict access


def safe_get_shape(weight: Any) -> WeightShape | None:
    """Safely extract shape from weight tensor.

    <!-- ai_cue:use_when=["shape_extraction", "tensor_errors"] -->

    Args:
        weight: Tensor, array, or None

    Returns:
        Validated shape, or None if extraction fails
    """
    if weight is None:
        return None

    try:
        # Try torch tensor
        if hasattr(weight, "shape"):
            dims = tuple(int(d) for d in weight.shape)
            return WeightShape(dims=dims)

        # Try numpy array
        if hasattr(weight, "__array__"):
            import numpy as np

            arr = np.asarray(weight)
            dims = tuple(int(d) for d in arr.shape)
            return WeightShape(dims=dims)

    except Exception as exc:
        LOGGER.debug("Failed to extract shape: %s", exc)
        return None

    return None


def validate_checkpoint_structure(checkpoint_data: dict[str, Any]) -> StateDictStructure:
    """Validate checkpoint structure with detailed error messages.

    <!-- ai_cue:use_when=["checkpoint_loading", "debugging_load_errors"] -->

    This is the preferred entry point for checkpoint validation.

    Args:
        checkpoint_data: Raw checkpoint from torch.load()

    Returns:
        Validated structure

    Raises:
        ValueError: With detailed message about what's wrong
    """
    try:
        return StateDictStructure.from_checkpoint(checkpoint_data)
    except ValueError as exc:
        LOGGER.error("Checkpoint validation failed: %s", exc)
        LOGGER.error("Checkpoint keys: %s", list(checkpoint_data.keys())[:10])
        raise
