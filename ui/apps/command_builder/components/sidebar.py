from __future__ import annotations

import streamlit as st

from ..state import CommandBuilderState, CommandType


def render_sidebar(state: CommandBuilderState) -> CommandType:
    with st.sidebar:
        st.markdown("## Command Builder")
        options = list(CommandType.ordered())
        try:
            default_index = options.index(state.command_type)
        except ValueError:
            default_index = 0
        selection = st.radio(
            "Select command",
            options,
            index=default_index,
            format_func=lambda value: value.value.capitalize(),
            key="command_builder_command_type",
        )
        state.command_type = selection

        st.divider()
        st.markdown(
            "Need a starting point? Use the contextual recommendations in the main panel to pre-fill the form with "
            "architecture-specific defaults."
        )

    return state.command_type
