from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.utils.command import CommandBuilder, CommandValidator
from ui.utils.ui_generator import generate_ui_from_schema
from ui.utils.ui_validator import validate_inputs

from ..models.command import CommandPageData
from ..state import CommandBuilderState, CommandType
from .command_preview import render_command_preview
from .execution import render_execution_panel

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "command_builder_test.yaml"


def render_test_page(state: CommandBuilderState, command_builder: CommandBuilder) -> None:
    page: CommandPageData = state.get_page(CommandType.TEST)
    st.markdown("### 🧪 Testing configuration")

    schema_result = generate_ui_from_schema(str(SCHEMA_PATH))
    values = dict(schema_result.values)
    errors = validate_inputs(values, str(SCHEMA_PATH))

    page.generated.values = values
    page.generated.overrides = schema_result.overrides
    page.generated.constant_overrides = list(schema_result.constant_overrides)
    page.generated.errors = errors

    if st.button("Generate command", type="primary", key="command_builder_test_generate"):
        if errors:
            st.error("Resolve validation errors before generating the command.")
        else:
            command_text = command_builder.build_command_from_overrides(
                script="test.py",
                overrides=schema_result.overrides,
                constant_overrides=schema_result.constant_overrides,
            )
            page.generated.generated = command_text
            page.generated.edited = command_text
            validator = CommandValidator()
            validation_ok, validation_error = validator.validate_command(command_text)
            page.generated.validation_error = None if validation_ok else validation_error
            st.session_state[f"command_builder_{CommandType.TEST.value}_command_text"] = command_text
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

    edited_command = render_command_preview(CommandType.TEST, page)
    render_execution_panel(
        command_builder=command_builder,
        command=edited_command,
        page=page,
        command_type=CommandType.TEST,
    )

    state.persist()
