from __future__ import annotations

import streamlit as st

from ui.utils.command import CommandBuilder
from ui.utils.config_parser import ConfigParser

from .components import render_predict_page, render_sidebar, render_test_page, render_training_page
from .services.recommendations import UseCaseRecommendationService
from .state import CommandBuilderState, CommandType


def run() -> None:
    st.set_page_config(page_title="OCR Command Builder", page_icon="🔍", layout="wide")
    st.title("🔍 OCR Training Command Builder")
    st.caption("Build and execute training, testing, and prediction commands with metadata-aware defaults.")

    state = CommandBuilderState.from_session()
    command_builder = CommandBuilder()
    config_parser = ConfigParser()
    recommendation_service = UseCaseRecommendationService(config_parser)

    command_type = render_sidebar(state)

    if command_type == CommandType.TRAIN:
        render_training_page(state, command_builder, recommendation_service, config_parser)
    elif command_type == CommandType.TEST:
        render_test_page(state, command_builder)
    else:
        render_predict_page(state, command_builder)

    state.persist()


def main() -> None:
    run()


if __name__ == "__main__":
    main()
