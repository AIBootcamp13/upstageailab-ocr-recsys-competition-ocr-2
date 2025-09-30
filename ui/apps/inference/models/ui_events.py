from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from streamlit.runtime.uploaded_file_manager import UploadedFile

from .config import PreprocessingConfig


@dataclass(slots=True)
class InferenceRequest:
    files: Sequence[UploadedFile]
    model_path: str
    use_preprocessing: bool = False
    preprocessing_config: PreprocessingConfig | None = None
