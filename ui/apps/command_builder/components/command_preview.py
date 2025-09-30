from __future__ import annotations

import streamlit as st

from ..models.command import CommandPageData
from ..services.formatting import pretty_format_command
from ..state import CommandType


def render_command_preview(command_type: CommandType, page: CommandPageData) -> str:
    st.subheader("Generated Command")
    text_key = f"command_builder_{command_type.value}_command_text"

    default_text = page.generated.edited or page.generated.generated or ""
    if text_key not in st.session_state:
        st.session_state[text_key] = default_text

    edited_command = st.text_area(
        "Command (editable)",
        height=150,
        key=text_key,
        help="Click Generate to refresh from the form. You can edit before executing.",
    )
    page.generated.edited = edited_command

    with st.expander("🔧 Overrides Preview"):
        if page.generated.constant_overrides:
            st.markdown("**Constant overrides:**")
            st.text("\n".join(page.generated.constant_overrides))
        if page.generated.overrides:
            st.markdown("**Dynamic overrides:**")
            st.text("\n".join(page.generated.overrides))
        if page.generated.generated:
            st.markdown("**Pretty command:**")
            st.code(pretty_format_command(page.generated.generated), language="bash")

    return edited_command
