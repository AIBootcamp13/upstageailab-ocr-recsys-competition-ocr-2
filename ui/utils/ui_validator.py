"""Schema-based UI input validator."""

from __future__ import annotations

from typing import Any

import yaml


def _eval_condition(expr: str, values: dict[str, Any]) -> bool:
    if not expr:
        return False
    ns = {"true": True, "false": False, "null": None}
    ns.update(values)
    try:
        return bool(eval(expr, {"__builtins__": {}}, ns))  # noqa: S307
    except Exception:
        return False


def validate_inputs(values: dict[str, Any], schema_path: str) -> list[str]:
    """Validate collected values using rules in schema YAML.

    Returns:
        List of error messages (empty if valid).
    """
    with open(schema_path) as f:
        schema = yaml.safe_load(f) or {}

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
        if "min" in rules and isinstance(value, (int, float)) and value < rules["min"]:
            errors.append(f"'{label}' must be >= {rules['min']}.")
        if "max" in rules and isinstance(value, (int, float)) and value > rules["max"]:
            errors.append(f"'{label}' must be <= {rules['max']}.")

        # range [min, max]
        if "range" in rules and isinstance(value, (int, float)):
            rmin, rmax = rules["range"][0], rules["range"][1]
            if not (rmin <= value <= rmax):
                errors.append(f"'{label}' must be between {rmin} and {rmax}.")

    # Cross-field validation examples
    # 1) Prevent resume with encoder change (if both present in current values)
    # For our schema, encoder is hidden when resume_training==true. If somehow both exist, block.
    if values.get("resume_training") and values.get("checkpoint_path") and values.get("encoder") not in (None, ""):
        errors.append("Cannot change Encoder when resuming from a checkpoint.")

    return errors
