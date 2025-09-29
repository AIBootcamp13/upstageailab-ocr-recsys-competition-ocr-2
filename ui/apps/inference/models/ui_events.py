from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from streamlit.runtime.uploaded_file_manager import UploadedFile


@dataclass(slots=True)
class InferenceRequest:
    files: Sequence[UploadedFile]
    model_path: str
