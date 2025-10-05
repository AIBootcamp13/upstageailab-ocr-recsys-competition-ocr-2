"""
Dynamic Streamlit UI generator driven by a YAML schema.

This module reads a UI schema file, renders widgets, applies conditional
visibility, and returns collected user inputs along with computed Hydra
overrides and constant overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from ui.utils.config_parser import ConfigParser


def _resolve_metadata_value(metadata: dict[str, Any], path: str | None) -> Any:
    if not path:
        return None
    value: Any = metadata
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


def _stringify_metadata_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list | tuple | set):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return yaml.safe_dump(value, sort_keys=False).strip()
    return str(value)


@st.cache_data(show_spinner=False)
def _load_schema(schema_path: str) -> dict[str, Any]:
    with open(schema_path) as f:
        return yaml.safe_load(f) or {}


@st.cache_data(show_spinner=False)
def _get_options_from_source(source: str) -> list[str]:
    """Resolve dynamic options list by a simple registry backed by ConfigParser."""
    cp = ConfigParser()
    model_source_map = {
        "models.backbones": "backbones",
        "models.encoders": "encoders",
        "models.decoders": "decoders",
        "models.heads": "heads",
        "models.optimizers": "optimizers",
        "models.losses": "losses",
    }
    if source in model_source_map:
        models = cp.get_available_models()
        return models.get(model_source_map[source], [])
    if source == "models.architectures":
        return cp.get_available_architectures()
    if source == "checkpoints":
        return cp.get_available_checkpoints()
    return cp.get_available_datasets() if source == "datasets" else []


def _is_visible(visible_if: str | None, values: dict[str, Any]) -> bool:
    """Very small safe evaluator for boolean expressions like `a == true`.

    Supports: ==, !=, and, or, parentheses; variables are keys in `values`.
    """
    if not visible_if:
        return True

    # Build a safe namespace mapping true/false/null and variables
    ns: dict[str, Any] = {
        "true": True,
        "false": False,
        "null": None,
    }
    ns.update(values)
    expr = visible_if.replace(" and ", " and ").replace(" or ", " or ")
    try:
        # eval with restricted globals; expressions restricted to literals and ns
        return bool(eval(expr, {"__builtins__": {}}, ns))  # noqa: S307
    except Exception:
        # On parse error, default to visible to avoid hiding controls unexpectedly
        return True


def _to_override(k: str, v: Any) -> str | None:
    if v is None or v == "":
        return None

    v_str = str(v).lower() if isinstance(v, bool) else str(v)

    # Quote values containing special characters that confuse Hydra parser
    # Characters that need quoting: , = : { } [ ] space tab newline return ' "
    # According to Hydra docs, these chars need escaping or the value needs to be quoted
    special_chars = ",=:{}[] \t\n\r'\""
    needs_quotes = any(char in v_str for char in special_chars)

    # For interpolations or special chars, quote the VALUE only (not the entire override)
    # Hydra expects: key="value" (where value is quoted)
    # The shell quoting 'key="value"' is handled separately when displaying the command
    contains_interpolation = "${" in v_str

    if needs_quotes or contains_interpolation:
        # Escape any double quotes in the value for proper nesting
        escaped_value = v_str.replace('"', '\\"')
        # Use double quotes around the value: key="value"
        return f'{k}="{escaped_value}"'
    else:
        return f"{k}={v_str}"


@dataclass
class UIGenerateResult:
    values: dict[str, Any]
    overrides: list[str]
    constant_overrides: list[str]


def compute_overrides(schema: dict[str, Any], values: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Compute hydra overrides from schema and collected values (pure function).

    Returns:
        (overrides, constant_overrides)
    """
    elements: list[dict[str, Any]] = schema.get("ui_elements", [])
    constant_overrides: list[str] = schema.get("constant_overrides", [])
    overrides: list[str] = []

    # Check if preprocessing profile is active
    preprocessing_profile = values.get("preprocessing_profile", "none")
    preprocessing_active = preprocessing_profile and preprocessing_profile != "none"

    for element in elements:
        key = element.get("key")
        if not isinstance(key, str) or not key:
            continue
        override_key = element.get("hydra_override")
        if not override_key:
            continue

        value = values.get(key)

        # Skip dataset override if preprocessing is active
        # (preprocessing profiles will set data=preprocessing automatically)
        if key == "dataset" and preprocessing_active:
            continue

        if isinstance(override_key, list):
            for k in override_key:
                if ov := _to_override(k, value):
                    overrides.append(ov)
        elif ov := _to_override(override_key, value):
            overrides.append(ov)

    return overrides, constant_overrides


def generate_ui_from_schema(schema_path: str) -> UIGenerateResult:
    """
    Render Streamlit widgets from a YAML schema and compute hydra overrides.

    Returns:
        UIGenerateResult with values, overrides, and constant_overrides.
    """
    schema = _load_schema(schema_path)
    elements: list[dict[str, Any]] = schema.get("ui_elements", [])
    constant_overrides: list[str] = schema.get("constant_overrides", [])

    config_parser = ConfigParser()
    architecture_metadata = config_parser.get_architecture_metadata()
    optimizer_metadata = config_parser.get_optimizer_metadata()

    values: dict[str, Any] = {}
    schema_prefix = Path(schema_path).stem.replace("-", "_")

    # First pass: render or compute defaults so visibility can reference earlier values
    for element in elements:
        etype = element.get("type")
        key = element.get("key")
        if not isinstance(key, str) or not key:
            st.warning("Skipping UI element with missing or invalid 'key'.")
            continue
        label_val = element.get("label")
        label = label_val if isinstance(label_val, str) and label_val else key
        visible_if = element.get("visible_if")

        widget_key = f"{schema_prefix}__{key}"

        # Resolve base options
        options = element.get("options")
        options_source = element.get("options_source")
        if options is None and options_source:
            options = _get_options_from_source(options_source)

        default = element.get("default")

        # Architecture-aware metadata
        arch_key = element.get("architecture_key", "architecture")
        selected_arch = values.get(arch_key) or element.get("architecture_fallback")
        architecture_info = architecture_metadata.get(selected_arch or "", {})
        ui_meta = architecture_info.get("ui_metadata", {}) if architecture_info else {}

        if options_metadata_key := element.get("options_metadata_key"):
            meta_options = _resolve_metadata_value(ui_meta, options_metadata_key)
            if isinstance(meta_options, dict):
                options = list(meta_options.keys())
            elif isinstance(meta_options, list | tuple | set):
                options = list(meta_options)
            elif meta_options is not None:
                options = [str(meta_options)]

        meta_filter_key = element.get("filter_by_architecture_key")
        if meta_filter_key and options:
            allowed = _resolve_metadata_value(ui_meta, meta_filter_key)
            if isinstance(allowed, str):
                allowed = [allowed]
            if allowed and (filtered := [opt for opt in options if opt in allowed]):
                options = filtered

        if meta_default_key := element.get("metadata_default_key"):
            meta_default = _resolve_metadata_value(ui_meta, meta_default_key)
            if meta_default is not None:
                default = meta_default

        # Optimizer-aware metadata
        optimizer_key = element.get("optimizer_key", "optimizer")
        selected_optimizer = values.get(optimizer_key) or element.get("optimizer_fallback")
        optimizer_info = optimizer_metadata.get(selected_optimizer or "", {})
        optimizer_ui_meta = optimizer_info.get("ui_metadata", {}) if optimizer_info else {}
        optimizer_metadata_key = element.get("optimizer_metadata_key")

        # Visibility check using current values dict
        if not _is_visible(visible_if, values):
            values[key] = None
            continue

        # Build help text
        help_segments: list[str] = []
        if base_help := element.get("help"):
            help_segments.append(str(base_help))

        if meta_help_key := element.get("metadata_help_key"):
            if meta_help_val := _resolve_metadata_value(ui_meta, meta_help_key):
                help_segments.append(_stringify_metadata_value(meta_help_val))

        if optimizer_help_key := element.get("optimizer_help_key"):
            if optimizer_help_val := _resolve_metadata_value(optimizer_ui_meta, optimizer_help_key):
                help_segments.append(_stringify_metadata_value(optimizer_help_val))

        help_text = "\n".join(segment for segment in help_segments if segment) or None

        # Precompute slider parameters for metadata overrides
        min_v = element.get("min_value")
        max_v = element.get("max_value")
        step = element.get("step")
        fmt = element.get("format")
        if optimizer_metadata_key and optimizer_ui_meta:
            meta_lr = _resolve_metadata_value(optimizer_ui_meta, optimizer_metadata_key)
            if isinstance(meta_lr, dict):
                if meta_lr.get("min") is not None:
                    min_v = meta_lr.get("min")
                if meta_lr.get("max") is not None:
                    max_v = meta_lr.get("max")
                if meta_lr.get("default") is not None:
                    default = meta_lr.get("default")
                if meta_lr.get("step") is not None:
                    step = meta_lr.get("step")

        if etype == "text_input":
            values[key] = st.text_input(label, value=default or "", help=help_text, key=widget_key)
        elif etype == "number_input":
            kwargs: dict[str, Any] = {}
            if default is not None:
                kwargs["value"] = default
            if element.get("min_value") is not None:
                kwargs["min_value"] = element.get("min_value")
            if element.get("max_value") is not None:
                kwargs["max_value"] = element.get("max_value")
            if help_text:
                kwargs["help"] = help_text
            values[key] = st.number_input(label, key=widget_key, **kwargs)
        elif etype == "checkbox":
            values[key] = st.checkbox(label, value=bool(default), help=help_text, key=widget_key)
        elif etype == "slider":
            if min_v is None or max_v is None:
                st.warning(f"Missing min/max for slider '{label}'. Skipping.")
                values[key] = default
            else:
                values[key] = st.slider(
                    label,
                    min_value=min_v,
                    max_value=max_v,
                    value=default if default is not None else min_v,
                    step=step,
                    format=fmt,
                    help=help_text,
                    key=widget_key,
                )
        elif etype == "info":
            info_key = element.get("metadata_info_key")
            info_value = _resolve_metadata_value(ui_meta, info_key) if info_key else None
            template = element.get("info_template", "{value}")
            fallback_text = element.get("info_fallback", "No metadata available.")
            if isinstance(info_value, dict):
                try:
                    message = template.format(**info_value)
                except KeyError:
                    message = template.format(value=_stringify_metadata_value(info_value))
            elif info_value is not None:
                message = template.format(value=_stringify_metadata_value(info_value))
            else:
                message = fallback_text
            st.info(message)
            values[key] = info_value
        elif etype == "selectbox":
            # Handle options that may be dicts with 'label' and 'value' keys
            if options and all(isinstance(opt, dict) for opt in options):
                # Options are dicts, extract labels for display and create value mapping
                display_options = [opt.get("label", str(opt)) for opt in options]
                value_map = {opt.get("label", str(opt)): opt.get("value", opt) for opt in options}
                opts = display_options
                dval = str(default) if default is not None else ""
                # Find the display option that corresponds to the default value
                default_display = None
                for opt in options:
                    if opt.get("value") == default:
                        default_display = opt.get("label", str(opt))
                        break
                if default_display:
                    dval = default_display
                index = opts.index(dval) if dval in opts else 0
            else:
                # Options are strings
                opts = options or [""]
                dval = str(default) if default is not None else ""
                index = opts.index(dval) if dval in opts else 0

            # Check if current session state value is valid for filtered options
            current_value = st.session_state.get(widget_key)
            if current_value is not None and current_value not in opts:
                # Clear invalid session state to use the default
                st.session_state.pop(widget_key, None)
                current_value = None

            if current_value is not None and current_value in opts:
                index = opts.index(current_value)

            selected_display = st.selectbox(
                label,
                opts,
                index=index,
                help=help_text,
                key=widget_key,
            )

            # Convert back to value if options were dicts
            if options and all(isinstance(opt, dict) for opt in options):
                values[key] = value_map.get(selected_display, selected_display)
            else:
                values[key] = selected_display
        else:
            st.warning(f"Unsupported UI element type: {etype}")
            values[key] = None

    # Compute overrides from collected values
    overrides, constant_overrides = compute_overrides(schema, values)

    return UIGenerateResult(values=values, overrides=overrides, constant_overrides=constant_overrides)
