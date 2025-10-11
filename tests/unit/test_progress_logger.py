"""Unit tests for the progress logger helpers."""

from __future__ import annotations

from types import SimpleNamespace

from ocr.utils import logging as logging_utils


def test_get_rich_console_handles_missing_dependency(monkeypatch):
    """Returns None gracefully when Rich Console cannot be imported."""
    monkeypatch.setattr(logging_utils, "RICH_AVAILABLE", False)
    monkeypatch.setattr(logging_utils, "Console", None)
    assert logging_utils.get_rich_console() is None


def test_get_rich_console_returns_instance(monkeypatch):
    """Returns a console instance when Rich is available."""
    fake_console = SimpleNamespace(captured=True)

    def fake_constructor(**_kwargs):
        return fake_console

    monkeypatch.setattr(logging_utils, "RICH_AVAILABLE", True)
    monkeypatch.setattr(logging_utils, "Console", fake_constructor)

    console = logging_utils.get_rich_console(color_system="standard")
    assert console is fake_console
