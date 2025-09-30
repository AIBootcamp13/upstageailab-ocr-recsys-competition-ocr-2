from __future__ import annotations

import streamlit as st


def rerun_app() -> None:
    """Trigger a Streamlit rerun using the available API surface."""

    rerun_callable = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_callable is None:
        raise RuntimeError("Streamlit rerun API is unavailable in this version.")

    rerun_callable()  # type: ignore[misc]
