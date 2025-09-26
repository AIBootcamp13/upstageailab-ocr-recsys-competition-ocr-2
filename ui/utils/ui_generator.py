"""
Dynamic Streamlit UI generator driven by a YAML schema.

This module reads a UI schema file, renders widgets, applies conditional
visibility, and returns collected user inputs along with computed Hydra
overrides and constant overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import yaml

from ui.utils.config_parser import ConfigParser


@st.cache_data(show_spinner=False)
def _load_schema(schema_path: str) -> Dict[str, Any]:
    with open(schema_path, "r") as f:
        return yaml.safe_load(f) or {}


@st.cache_data(show_spinner=False)
def _get_options_from_source(source: str) -> List[str]:
    """Resolve dynamic options list by a simple registry backed by ConfigParser."""
    cp = ConfigParser()
    if source == "models.backbones":
        models = cp.get_available_models()
        return models.get("backbones", [])
    if source == "checkpoints":
        return cp.get_available_checkpoints()
    if source == "datasets":
        return cp.get_available_datasets()
    # Default: unknown source -> empty list
    return []


def _is_visible(visible_if: Optional[str], values: Dict[str, Any]) -> bool:
    """Very small safe evaluator for boolean expressions like `a == true`.

    Supports: ==, !=, and, or, parentheses; variables are keys in `values`.
    """
    if not visible_if:
        return True

    # Build a safe namespace mapping true/false/null and variables
    ns: Dict[str, Any] = {
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


def _to_override(k: str, v: Any) -> Optional[str]:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return f"{k}={str(v).lower()}"
    return f"{k}={v}"


@dataclass
class UIGenerateResult:
    values: Dict[str, Any]
    overrides: List[str]
    constant_overrides: List[str]


def compute_overrides(schema: Dict[str, Any], values: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Compute hydra overrides from schema and collected values (pure function).

    Returns:
        (overrides, constant_overrides)
    """
    elements: List[Dict[str, Any]] = schema.get("ui_elements", [])
    constant_overrides: List[str] = schema.get("constant_overrides", [])
    overrides: List[str] = []
    for element in elements:
        key = element.get("key")
        if not isinstance(key, str) or not key:
            continue
        override_key = element.get("hydra_override")
        if not override_key:
            continue
        if ov := _to_override(override_key, values.get(key)):
            overrides.append(ov)
    return overrides, constant_overrides


def generate_ui_from_schema(schema_path: str) -> UIGenerateResult:
    """
    Render Streamlit widgets from a YAML schema and compute hydra overrides.

    Returns:
        UIGenerateResult with values, overrides, and constant_overrides.
    """
    schema = _load_schema(schema_path)
    elements: List[Dict[str, Any]] = schema.get("ui_elements", [])
    constant_overrides: List[str] = schema.get("constant_overrides", [])

    values: Dict[str, Any] = {}

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

        # Resolve options if present
        options = element.get("options")
        options_source = element.get("options_source")
        if options is None and options_source:
            options = _get_options_from_source(options_source)

        # Prepare default
        default = element.get("default")

        # Visibility check using current values dict
        if not _is_visible(visible_if, values):
            # Store None for hidden to simplify required_if checks
            values[key] = None
            continue

        # Render widget by type
        if etype == "text_input":
            values[key] = st.text_input(label, value=default or "", help=element.get("help"))
        elif etype == "number_input":
            kwargs: Dict[str, Any] = {}
            if default is not None:
                kwargs["value"] = default
            if element.get("min_value") is not None:
                kwargs["min_value"] = element.get("min_value")
            if element.get("max_value") is not None:
                kwargs["max_value"] = element.get("max_value")
            if help_text := element.get("help"):
                kwargs["help"] = help_text
            values[key] = st.number_input(label, **kwargs)
        elif etype == "checkbox":
            values[key] = st.checkbox(label, value=bool(default), help=element.get("help"))
        elif etype == "slider":
            min_v = element.get("min_value")
            max_v = element.get("max_value")
            step = element.get("step")
            fmt = element.get("format")
            help_text = element.get("help")
            # Streamlit slider requires min/max and a value in range
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
                )
        elif etype == "selectbox":
            opts = options or [""]
            # Compute index for default if exists
            # Coerce default to str if options are strings to avoid type mismatch
            dval = str(default) if default is not None else ""
            index = opts.index(dval) if dval in opts else 0
            values[key] = st.selectbox(
                label,
                opts,
                index=index,
                help=element.get("help"),
            )
        else:
            st.warning(f"Unsupported UI element type: {etype}")
            values[key] = None

    # Compute overrides from collected values
    overrides, constant_overrides = compute_overrides(schema, values)

    return UIGenerateResult(values=values, overrides=overrides, constant_overrides=constant_overrides)
