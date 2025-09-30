from __future__ import annotations

"""Session state helpers for Streamlit apps.

Hyperparameter defaults and keys must align with ``configs/ui/inference.yaml``
and the guidance in ``docs/ai_handbook/02_protocols``. Update those sources
first, then reflect changes here to avoid divergent behaviour across apps.
"""

from dataclasses import dataclass, field, replace
from typing import Any

import streamlit as st

from .models.config import PreprocessingConfig, SliderConfig

SESSION_KEYS = {
    "inference_results": list,
    "selected_images": set,
    "processed_images": dict,
    "selected_model": lambda: None,
    "hyperparams": dict,
    "previous_uploaded_files": set,
    "preprocessing_enabled": lambda: False,
    "preprocessing_default_initialized": lambda: False,
    "preprocessing_overrides": dict,
}


@dataclass(slots=True)
class InferenceState:
    inference_results: list[dict[str, Any]] = field(default_factory=list)
    selected_images: set[str] = field(default_factory=set)
    processed_images: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    selected_model: str | None = None
    hyperparams: dict[str, float] = field(default_factory=dict)
    preprocessing_enabled: bool = False
    preprocessing_overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_session(cls) -> InferenceState:
        ensure_session_defaults()
        state = st.session_state
        processed_images: dict[str, dict[str, set[str]]] = {}
        for model_key, buckets in state.processed_images.items():
            if isinstance(buckets, dict):
                processed_images[model_key] = {mode: set(values) for mode, values in buckets.items()}
            elif isinstance(buckets, set | list | tuple):
                processed_images[model_key] = {"default": set(buckets)}
            else:
                processed_images[model_key] = {}
        preprocessing_enabled = state.preprocessing_enabled if isinstance(state.preprocessing_enabled, bool) else False
        return cls(
            inference_results=list(state.inference_results),
            selected_images=set(state.selected_images),
            processed_images=processed_images,
            selected_model=state.selected_model,
            hyperparams=dict(state.hyperparams or {}),
            preprocessing_enabled=preprocessing_enabled,
            preprocessing_overrides=dict(state.preprocessing_overrides or {}),
        )

    def persist(self) -> None:
        st.session_state.inference_results = list(self.inference_results)
        st.session_state.selected_images = set(self.selected_images)
        st.session_state.processed_images = {
            model: {mode: set(values) for mode, values in buckets.items()} for model, buckets in self.processed_images.items()
        }
        st.session_state.selected_model = self.selected_model
        st.session_state.hyperparams = dict(self.hyperparams)
        st.session_state.preprocessing_enabled = bool(self.preprocessing_enabled)
        st.session_state.preprocessing_overrides = dict(self.preprocessing_overrides)

    def update_hyperparameter(self, key: str, value: float) -> None:
        self.hyperparams[key] = value

    def ensure_processed_bucket(self, model_path: str, mode_key: str) -> None:
        model_buckets = self.processed_images.setdefault(model_path, {})
        model_buckets.setdefault(mode_key, set())

    def reset_for_model(self, model_path: str | None) -> None:
        if self.selected_model != model_path:
            self.inference_results.clear()
            if self.selected_model is not None:
                self.processed_images.pop(self.selected_model, None)
            self.selected_model = model_path

    def update_preprocessing_override(self, key: str, value: Any) -> None:
        self.preprocessing_overrides[key] = value

    def build_preprocessing_config(self, base: PreprocessingConfig) -> PreprocessingConfig:
        if not self.preprocessing_overrides:
            return base
        return replace(base, **self.preprocessing_overrides)


def ensure_session_defaults() -> None:
    for key, factory in SESSION_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = factory()


def init_hyperparameters(sliders: dict[str, SliderConfig]) -> None:
    ensure_session_defaults()
    if not st.session_state.hyperparams:
        st.session_state.hyperparams = {key: slider.default for key, slider in sliders.items()}


def init_preprocessing(config: PreprocessingConfig) -> None:
    ensure_session_defaults()
    if not st.session_state.preprocessing_default_initialized:
        st.session_state.preprocessing_enabled = bool(config.default_enabled)
        st.session_state.preprocessing_default_initialized = True
