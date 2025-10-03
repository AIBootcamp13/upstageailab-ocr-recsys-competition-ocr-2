from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.utils.command import CommandBuilder, CommandValidator
from ui.utils.config_parser import ConfigParser
from ui.utils.ui_generator import generate_ui_from_schema
from ui.utils.ui_validator import validate_inputs

from ..models.command import CommandPageData
from ..services.overrides import build_additional_overrides, maybe_suffix_exp_name
from ..services.recommendations import UseCaseRecommendationService
from ..state import CommandBuilderState, CommandType
from ..utils import rerun_app
from .command_preview import render_command_preview
from .execution import render_execution_panel
from .suggestions import render_use_case_recommendations

PREPROCESSING_STATE_KEY = "command_builder_train__preprocessing_profile"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "command_builder_train.yaml"
SCHEMA_PREFIX = SCHEMA_PATH.stem


def _reset_form_state() -> None:
    keys_to_clear = [key for key in st.session_state if isinstance(key, str) and key.startswith(f"{SCHEMA_PREFIX}__")]
    for key in keys_to_clear:
        st.session_state.pop(key)
    st.session_state.pop(PREPROCESSING_STATE_KEY, None)


def render_training_page(
    state: CommandBuilderState,
    command_builder: CommandBuilder,
    recommendation_service: UseCaseRecommendationService,
    config_parser: ConfigParser,
) -> None:
    page: CommandPageData = state.get_page(CommandType.TRAIN)

    current_architecture = st.session_state.get(f"{SCHEMA_PREFIX}__architecture") or page.generated.values.get("architecture")
    recommendations = recommendation_service.for_architecture(current_architecture)
    render_use_case_recommendations(
        recommendations,
        state,
        schema_prefix=SCHEMA_PREFIX,
        auxiliary_state_keys={"preprocessing_profile": PREPROCESSING_STATE_KEY},
    )

    st.markdown("### 🚀 Training configuration")

    if st.button("Reset form", key="command_builder_train_reset"):
        state.active_use_case = None
        state.reset_command(CommandType.TRAIN)
        _reset_form_state()
        rerun_app()

    schema_result = generate_ui_from_schema(str(SCHEMA_PATH))
    values = dict(schema_result.values)

    state.append_model_suffix = st.checkbox(
        "Append architecture + encoder to experiment name",
        value=state.append_model_suffix,
        help="Helps keep experiment directories unique across architecture/backbone changes.",
        key="command_builder_append_suffix",
    )

    errors = validate_inputs(values, str(SCHEMA_PATH))

    additional_overrides = build_additional_overrides(values, config_parser)
    all_overrides = schema_result.overrides + additional_overrides
    all_overrides = maybe_suffix_exp_name(all_overrides, values, state.append_model_suffix)
    constant_overrides = list(schema_result.constant_overrides)

    page.generated.values = values
    page.generated.overrides = all_overrides
    page.generated.constant_overrides = constant_overrides
    page.generated.errors = errors

    if st.button("Generate command", type="primary", key="command_builder_train_generate"):
        if errors:
            st.error("Resolve validation errors before generating the command.")
        else:
            command_text = command_builder.build_command_from_overrides(
                script="train.py",
                overrides=all_overrides,
                constant_overrides=constant_overrides,
            )
            page.generated.generated = command_text
            page.generated.edited = command_text
            validator = CommandValidator()
            validation_ok, validation_error = validator.validate_command(command_text)
            page.generated.validation_error = None if validation_ok else validation_error
            st.session_state[f"command_builder_{CommandType.TRAIN.value}_command_text"] = command_text
            if validation_ok:
                st.success("Command is valid and ready to run.")
            else:
                st.warning(f"Command generated but validation raised an issue: {validation_error}")

    if errors:
        for message in errors:
            st.error(message)
    elif page.generated.validation_error:
        st.error(page.generated.validation_error)
    elif page.generated.generated:
        st.success("Command is valid")

    edited_command = render_command_preview(CommandType.TRAIN, page)

    render_execution_panel(
        command_builder=command_builder,
        command=edited_command,
        page=page,
        command_type=CommandType.TRAIN,
    )

    state.persist()
