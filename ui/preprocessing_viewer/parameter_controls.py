"""Dynamic parameter control panels for the Streamlit preprocessing viewer."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import streamlit as st

from .preprocessing.parameter_mapping import (
    ModuleParameterMapping,
    PreprocessingParameterMapping,
    ProcessingStage,
    UIControlType,
    UIParameterDefinition,
)

logger = logging.getLogger(__name__)


class ParameterControls:
    """Render categorized parameter panels with real-time validation feedback."""

    def __init__(self, parameter_mapping: PreprocessingParameterMapping | None = None) -> None:
        self.parameter_mapping = parameter_mapping or PreprocessingParameterMapping()
        self._validation_errors: dict[str, str] = {}
        self._last_valid_config: dict[str, Any] = {}

    def render_parameter_panels(
        self,
        current_config: dict[str, Any],
        on_change_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Render the full parameter control interface and return the updated config."""
        st.subheader("⚙️ Parameter Controls")

        stage_groups = self._group_modules_by_stage()
        stage_order = [stage for stage in ProcessingStage if stage in stage_groups]

        if not stage_order:
            st.info("No parameter mappings registered. Using default configuration.")
            return current_config

        self._validation_errors = {}
        updated_config = current_config.copy()

        stage_tabs = st.tabs([stage.value.title() for stage in stage_order])
        for tab, stage in zip(stage_tabs, stage_order, strict=True):
            with tab:
                updated_config = self._render_stage_panel(
                    stage,
                    stage_groups[stage],
                    updated_config,
                    on_change_callback,
                )

        self._render_validation_feedback()

        if not self._validation_errors:
            self._last_valid_config = updated_config.copy()

        return updated_config

    def reset_to_defaults(self) -> dict[str, Any]:
        """Return the default configuration derived from parameter mappings."""
        defaults: dict[str, Any] = {}
        for mapping in self.parameter_mapping.get_all_mappings().values():
            for param in mapping.parameters:
                defaults[param.key] = param.default
            if mapping.enable_parameter:
                defaults[mapping.enable_parameter] = mapping.enabled_by_default
        self._validation_errors.clear()
        self._last_valid_config = defaults.copy()
        return defaults

    def get_validation_errors(self) -> dict[str, str]:
        return self._validation_errors.copy()

    def is_config_valid(self) -> bool:
        return not self._validation_errors

    # Internal helpers -----------------------------------------------------------------

    def _group_modules_by_stage(self) -> dict[ProcessingStage, list[ModuleParameterMapping]]:
        stage_groups: dict[ProcessingStage, list[ModuleParameterMapping]] = {}
        for mapping in self.parameter_mapping.get_all_mappings().values():
            stage_groups.setdefault(mapping.stage, []).append(mapping)
        for modules in stage_groups.values():
            modules.sort(key=lambda module: module.module_name)
        return stage_groups

    def _render_stage_panel(
        self,
        stage: ProcessingStage,
        modules: list[ModuleParameterMapping],
        current_config: dict[str, Any],
        on_change_callback: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        updated_config = current_config.copy()

        for module in modules:
            expanded = module.enabled_by_default if module.enable_parameter else False
            with st.expander(f"📦 {module.module_name.replace('_', ' ').title()}", expanded=expanded):
                st.markdown(f"*{module.description}*")

                if module.enable_parameter:
                    enable_key = module.enable_parameter
                    enabled = st.checkbox(
                        f"Enable {module.module_name.replace('_', ' ').title()}",
                        value=bool(updated_config.get(enable_key, module.enabled_by_default)),
                        key=f"enable_{module.module_name}",
                        help=f"Enable or disable the {module.module_name.replace('_', ' ')} module.",
                    )
                    updated_config[enable_key] = enabled
                    if not enabled:
                        st.info("Module disabled. Enable to adjust parameters.")
                        continue

                for param in module.parameters:
                    if module.enable_parameter and param.key == module.enable_parameter:
                        continue
                    if not self._should_show_parameter(param, updated_config):
                        continue

                    updated_config = self._render_parameter_control(
                        param,
                        module.module_name,
                        updated_config,
                        on_change_callback,
                    )

        return updated_config

    def _should_show_parameter(self, param: UIParameterDefinition, config: dict[str, Any]) -> bool:
        if param.depends_on and not config.get(param.depends_on, False):
            return False

        if param.show_when:
            for key, allowed_values in param.show_when.items():
                current_value = config.get(key)
                if isinstance(allowed_values, list):
                    if current_value not in allowed_values:
                        return False
                elif current_value != allowed_values:
                    return False
        return True

    def _render_parameter_control(
        self,
        param: UIParameterDefinition,
        module_name: str,
        current_config: dict[str, Any],
        on_change_callback: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        updated_config = current_config.copy()
        current_value = updated_config.get(param.key, param.default)
        control_key = f"param_{module_name}_{param.key}"

        try:
            new_value = self._render_control(param, current_value, control_key)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to render parameter %s: %s", param.key, exc)
            st.error(f"Error rendering parameter: {param.label}")
            return updated_config

        updated_config[param.key] = new_value

        validation_error = self._validate_parameter(param, new_value)
        if validation_error:
            self._validation_errors[param.key] = validation_error
            st.error(f"⚠️ {validation_error}")
        else:
            self._validation_errors.pop(param.key, None)

        if new_value != current_value and on_change_callback:
            on_change_callback(updated_config.copy())

        return updated_config

    def _render_control(self, param: UIParameterDefinition, current_value: Any, control_key: str) -> Any:
        if param.type == UIControlType.CHECKBOX:
            return st.checkbox(param.label, value=bool(current_value), key=control_key, help=param.help)

        if param.type == UIControlType.SLIDER:
            if self._is_int_slider(param, current_value):
                min_value = int(param.min_value) if param.min_value is not None else 0
                max_value = int(param.max_value) if param.max_value is not None else 100
                step = int(param.step) if param.step is not None else 1
                value = int(current_value)
                return st.slider(
                    param.label, min_value=min_value, max_value=max_value, value=value, step=step, key=control_key, help=param.help
                )

            min_value = float(param.min_value) if param.min_value is not None else 0.0
            max_value = float(param.max_value) if param.max_value is not None else 1.0
            step = float(param.step) if param.step is not None else 0.01
            value = float(current_value)
            return st.slider(
                param.label, min_value=min_value, max_value=max_value, value=value, step=step, key=control_key, help=param.help
            )

        if param.type == UIControlType.NUMBER_INPUT:
            if self._is_int_number(param, current_value):
                min_value = int(param.min_value) if param.min_value is not None else None
                max_value = int(param.max_value) if param.max_value is not None else None
                step = int(param.step) if param.step is not None else 1
                value = int(current_value)
                return st.number_input(
                    param.label,
                    min_value=min_value,
                    max_value=max_value,
                    value=value,
                    step=step,
                    key=control_key,
                    help=param.help,
                    format="%d",
                )

            min_value = float(param.min_value) if param.min_value is not None else None
            max_value = float(param.max_value) if param.max_value is not None else None
            step = float(param.step) if param.step is not None else 0.1
            value = float(current_value)
            return st.number_input(
                param.label,
                min_value=min_value,
                max_value=max_value,
                value=value,
                step=step,
                key=control_key,
                help=param.help,
            )

        if param.type == UIControlType.SELECTBOX:
            options = list(param.options or [])
            if current_value is not None and current_value not in options:
                options.insert(0, current_value)
            if not options:
                options = ["—"]
                index = 0
            else:
                index = options.index(current_value) if current_value in options else 0
            return st.selectbox(param.label, options=options, index=index, key=control_key, help=param.help)

        if param.type == UIControlType.RADIO:
            options = list(param.options or [])
            if current_value is not None and current_value not in options:
                options.insert(0, current_value)
            if not options:
                options = ["—"]
            index = options.index(current_value) if current_value in options else 0
            return st.radio(param.label, options=options, index=index, key=control_key, help=param.help)

        if param.type == UIControlType.TEXT_INPUT:
            return st.text_input(param.label, value=str(current_value or ""), key=control_key, help=param.help)

        if param.type == UIControlType.MULTISELECT:
            options = list(param.options or [])
            current_list = current_value if isinstance(current_value, list) else ([current_value] if current_value else [])
            for value in current_list:
                if value not in options and value is not None:
                    options.append(value)
            return st.multiselect(param.label, options=options, default=current_list, key=control_key, help=param.help)

        st.error(f"Unsupported control type: {param.type}")
        return current_value

    def _validate_parameter(self, param: UIParameterDefinition, value: Any) -> str | None:
        if param.required and (value is None or value == ""):
            return f"{param.label} is required"

        if param.type in {UIControlType.SLIDER, UIControlType.NUMBER_INPUT} and isinstance(value, int | float):
            if param.min_value is not None and value < param.min_value:
                return f"{param.label} must be at least {param.min_value}"
            if param.max_value is not None and value > param.max_value:
                return f"{param.label} must be at most {param.max_value}"

        if param.validation_rules:
            for rule_name, rule_value in param.validation_rules.items():
                if rule_name == "min_length" and isinstance(value, str | list) and len(value) < int(rule_value):
                    return f"{param.label} must have at least {rule_value} items"
                if rule_name == "max_length" and isinstance(value, str | list) and len(value) > int(rule_value):
                    return f"{param.label} must have at most {rule_value} items"
        return None

    def _render_validation_feedback(self) -> None:
        if self._validation_errors:
            with st.expander("⚠️ Validation Issues", expanded=True):
                for key, message in self._validation_errors.items():
                    st.write(f"• **{key}** — {message}")
        else:
            st.caption("✅ All parameters pass validation checks")

    @staticmethod
    def _is_int_slider(param: UIParameterDefinition, current_value: Any) -> bool:
        default = param.default
        if isinstance(default, bool) or isinstance(current_value, bool):
            return False
        if not isinstance(default, int) and not isinstance(current_value, int):
            return False
        for candidate in (param.min_value, param.max_value, param.step):
            if candidate is not None and float(candidate).is_integer() is False:
                return False
        return True

    @staticmethod
    def _is_int_number(param: UIParameterDefinition, current_value: Any) -> bool:
        default = param.default
        if isinstance(default, bool) or isinstance(current_value, bool):
            return False
        if isinstance(default, int) or isinstance(current_value, int):
            if param.step is not None and float(param.step).is_integer() is False:
                return False
            return True
        return False
