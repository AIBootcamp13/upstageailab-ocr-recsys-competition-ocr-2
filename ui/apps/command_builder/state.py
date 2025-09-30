from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import streamlit as st

from .models.command import CommandPageData

SESSION_KEY = "command_builder_state"


class CommandType(str, Enum):
    TRAIN = "train"
    TEST = "test"
    PREDICT = "predict"

    @classmethod
    def ordered(cls) -> tuple[CommandType, ...]:
        return (cls.TRAIN, cls.TEST, cls.PREDICT)


@dataclass(slots=True)
class CommandBuilderState:
    command_type: CommandType = CommandType.TRAIN
    append_model_suffix: bool = True
    active_use_case: str | None = None
    pages: dict[CommandType, CommandPageData] = field(
        default_factory=lambda: {command_type: CommandPageData() for command_type in CommandType.ordered()}
    )

    @classmethod
    def from_session(cls) -> CommandBuilderState:
        state = st.session_state.get(SESSION_KEY)
        if isinstance(state, cls):
            return state
        state = cls()
        st.session_state[SESSION_KEY] = state
        return state

    def persist(self) -> None:
        st.session_state[SESSION_KEY] = self

    def get_page(self, command_type: CommandType) -> CommandPageData:
        if command_type not in self.pages:
            self.pages[command_type] = CommandPageData()
        return self.pages[command_type]

    def reset_command(self, command_type: CommandType) -> None:
        self.get_page(command_type).generated.clear()
