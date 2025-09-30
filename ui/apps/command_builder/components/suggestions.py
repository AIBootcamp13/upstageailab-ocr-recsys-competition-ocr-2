from __future__ import annotations

import streamlit as st

from ..models.recommendation import UseCaseRecommendation
from ..state import CommandBuilderState
from ..utils import rerun_app


def render_use_case_recommendations(
    recommendations: list[UseCaseRecommendation],
    state: CommandBuilderState,
    schema_prefix: str,
    auxiliary_state_keys: dict[str, str] | None = None,
) -> None:
    if not recommendations:
        return

    st.markdown("### 🎯 Suggested setups")
    auxiliary_state_keys = auxiliary_state_keys or {}

    cols = st.columns(len(recommendations)) if len(recommendations) <= 3 else None
    iterator = enumerate(recommendations)
    for idx, rec in iterator:
        container = cols[idx] if cols else st.container()
        with container:
            is_active = state.active_use_case == rec.id
            badge = "✅" if is_active else ""
            st.markdown(f"**{badge} {rec.title}**")
            if rec.description:
                st.caption(rec.description)

            if st.button("Apply", key=f"apply_use_case_{rec.id}"):
                for key, value in rec.iter_assignments():
                    if session_key := auxiliary_state_keys.get(key):
                        st.session_state[session_key] = value
                    else:
                        st.session_state[f"{schema_prefix}__{key}"] = value
                state.active_use_case = rec.id
                rerun_app()

            if is_active and rec.parameters:
                formatted = "\n".join(f"• **{name}** → `{val}`" for name, val in rec.parameters.items())
                st.markdown(formatted)
