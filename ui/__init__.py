"""UI package for OCR project Streamlit applications."""

import sys
from importlib import import_module


def _alias_module(alias: str, target: str) -> None:
    module = import_module(target)
    sys.modules[alias] = module


# _alias_module("ui.visualization", "ui.visualization")

__all__: list[str] = []
