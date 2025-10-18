from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field
from streamlit.runtime.uploaded_file_manager import UploadedFile

from .config import PreprocessingConfig


class InferenceRequest(BaseModel):
    """Data contract for inference requests from the UI."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    files: Sequence[UploadedFile] = Field(..., description="Uploaded files to process")
    model_path: str = Field(..., description="Path to the model checkpoint")
    config_path: str | None = Field(None, description="Path to the model config file")
    use_preprocessing: bool = Field(default=False, description="Whether to use preprocessing")
    preprocessing_config: PreprocessingConfig | None = Field(default=None, description="Preprocessing configuration")
