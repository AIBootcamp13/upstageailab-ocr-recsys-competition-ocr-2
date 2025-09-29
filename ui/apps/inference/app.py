from __future__ import annotations

"""Streamlit app orchestration layer.

Refer to ``docs/ai_handbook/02_protocols/11_streamlit_maintenance_protocol.md``
and ``docs/ai_handbook/02_protocols/12_streamlit_refactoring_protocol.md``
before changing layout or control flow. Configuration lives in
``configs/ui/inference.yaml`` with additional metadata in ``ui_meta/`` and
schemas in ``configs/schemas/``—keep those sources authoritative and avoid
guessing widget behaviour.
"""

from typing import Literal, cast

import streamlit as st

from .components import results as results_component
from .components import sidebar as sidebar_component
from .models.checkpoint import CheckpointMetadata
from .services.checkpoint_catalog import CatalogOptions, build_catalog
from .services.config_loader import load_ui_config
from .services.inference_runner import InferenceService
from .services.schema_validator import load_schema
from .state import InferenceState


@st.cache_data(show_spinner=False)
def _load_catalog(options: CatalogOptions) -> list[CheckpointMetadata]:
    schema = load_schema()
    return build_catalog(options, schema)


def run() -> None:
    config = load_ui_config()
    layout: Literal["centered", "wide"] = cast(
        Literal["centered", "wide"],
        config.app.layout if config.app.layout in {"centered", "wide"} else "wide",
    )
    sidebar_state: Literal["auto", "expanded", "collapsed"] = cast(
        Literal["auto", "expanded", "collapsed"],
        config.app.initial_sidebar_state if config.app.initial_sidebar_state in {"auto", "expanded", "collapsed"} else "auto",
    )

    st.set_page_config(
        page_title=config.app.title,
        page_icon=config.app.page_icon,
        layout=layout,
        initial_sidebar_state=sidebar_state,
    )
    st.title(config.app.title)
    st.markdown(config.app.subtitle)

    state = InferenceState.from_session()
    options = CatalogOptions.from_paths(config.paths)
    catalog = _load_catalog(options)
    inference_service = InferenceService()

    request = sidebar_component.render_controls(state, config, catalog)

    if request is not None:
        inference_service.run(state, request, state.hyperparams)

    results_component.render_results(state, config)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
