"""Schema-based UI input validator."""

from __future__ import annotations

from typing import Any

import yaml

from ui.utils.config_parser import ConfigParser


def _eval_condition(expr: str, values: dict[str, Any]) -> bool:
    if not expr:
        return False
    ns = {"true": True, "false": False, "null": None}
    ns.update(values)
    try:
        return bool(eval(expr, {"__builtins__": {}}, ns))  # noqa: S307
    except Exception:
        return False


def _get_optimizer_lr_metadata(optimizer_name: str, optimizer_metadata: dict[str, Any]) -> dict[str, Any]:
    if not optimizer_name:
        return {}
    info = optimizer_metadata.get(optimizer_name, {}) or {}
    lr_meta = info.get("ui_metadata", {}).get("learning_rate", {})
    return lr_meta if isinstance(lr_meta, dict) else {}


def validate_inputs(values: dict[str, Any], schema_path: str) -> list[str]:
    """Validate collected values using rules in schema YAML.

    Returns:
        List of error messages (empty if valid).
    """
    with open(schema_path) as f:
        schema = yaml.safe_load(f) or {}

    config_parser = ConfigParser()
    architecture_metadata = config_parser.get_architecture_metadata()
    optimizer_metadata = config_parser.get_optimizer_metadata()

    errors: list[str] = []
    elements = schema.get("ui_elements", [])
    for element in elements:
        key = element.get("key")
        label = element.get("label", key)
        rules = element.get("validation", {}) or {}
        value = values.get(key)

        # required_if first
        req_if = rules.get("required_if")
        if req_if and _eval_condition(req_if, values) and value in (None, ""):
            errors.append(f"'{label}' is required.")

        # required
        if rules.get("required") and value in (None, ""):
            errors.append(f"'{label}' is required.")
            continue

        # Only further checks if value present
        if value in (None, ""):
            continue

        # min_length
        if "min_length" in rules and isinstance(value, str) and len(value) < int(rules["min_length"]):
            errors.append(f"'{label}' must be at least {int(rules['min_length'])} characters long.")

        # min/max for numeric
        if "min" in rules and isinstance(value, int | float) and value < rules["min"]:
            errors.append(f"'{label}' must be >= {rules['min']}.")
        if "max" in rules and isinstance(value, int | float) and value > rules["max"]:
            errors.append(f"'{label}' must be <= {rules['max']}.")

        # range [min, max]
        if "range" in rules and isinstance(value, int | float):
            rmin, rmax = rules["range"][0], rules["range"][1]
            if not (rmin <= value <= rmax):
                errors.append(f"'{label}' must be between {rmin} and {rmax}.")

    # Cross-field validation examples
    # 1) Prevent resume with encoder change (if both present in current values)
    # For our schema, encoder is hidden when resume_training==true. If somehow both exist, block.
    if values.get("resume_training") and values.get("checkpoint_path") and values.get("encoder") not in (None, ""):
        errors.append("Cannot change Encoder when resuming from a checkpoint.")

    selected_optimizer = values.get("optimizer")
    selected_arch = values.get("architecture")

    # Architecture compatibility validation
    if selected_arch:
        arch_info = architecture_metadata.get(selected_arch, {})
        ui_meta = arch_info.get("ui_metadata", {}) if arch_info else {}

        # Get user selections
        selected_backbone = values.get("encoder")
        selected_decoder = values.get("decoder")
        selected_head = values.get("head")
        selected_loss = values.get("loss")

        # Validate encoder/backbone compatibility
        compatible_backbones = ui_meta.get("compatible_backbones") or []
        if compatible_backbones and selected_backbone and selected_backbone not in compatible_backbones:
            errors.append(
                f"Encoder '{selected_backbone}' is not compatible with '{selected_arch}' architecture. "
                f"Compatible encoders: {', '.join(compatible_backbones)}"
            )

        # Validate decoder compatibility
        compatible_decoders = ui_meta.get("compatible_decoders") or []
        if compatible_decoders and selected_decoder and selected_decoder not in compatible_decoders:
            errors.append(
                f"Decoder '{selected_decoder}' is not compatible with '{selected_arch}' architecture. "
                f"Compatible decoders: {', '.join(compatible_decoders)}"
            )

        # Validate head compatibility
        compatible_heads = ui_meta.get("compatible_heads") or []
        if compatible_heads and selected_head and selected_head not in compatible_heads:
            errors.append(
                f"Head '{selected_head}' is not compatible with '{selected_arch}' architecture. "
                f"Compatible heads: {', '.join(compatible_heads)}"
            )

        # Validate loss compatibility
        compatible_losses = ui_meta.get("compatible_losses") or []
        if compatible_losses and selected_loss and selected_loss not in compatible_losses:
            errors.append(
                f"Loss '{selected_loss}' is not compatible with '{selected_arch}' architecture. "
                f"Compatible losses: {', '.join(compatible_losses)}"
            )

        # Validate optimizer recommendation (warning, not error)
        recommended_opts = ui_meta.get("recommended_optimizers") or []
        if recommended_opts and selected_optimizer and selected_optimizer not in recommended_opts:
            errors.append(
                f"Warning: Optimizer '{selected_optimizer}' is not recommended for '{selected_arch}'. "
                f"Recommended optimizers: {', '.join(recommended_opts)}"
            )
        learning_rate = values.get("learning_rate")
        if selected_optimizer and isinstance(learning_rate, int | float):
            lr_meta = _get_optimizer_lr_metadata(selected_optimizer, optimizer_metadata)
            lr_min = lr_meta.get("min")
            lr_max = lr_meta.get("max")
            if lr_min is not None and learning_rate < lr_min:
                errors.append(f"Learning rate must be >= {lr_min} for optimizer '{selected_optimizer}'.")
            if lr_max is not None and learning_rate > lr_max:
                errors.append(f"Learning rate must be <= {lr_max} for optimizer '{selected_optimizer}'.")

    return errors
