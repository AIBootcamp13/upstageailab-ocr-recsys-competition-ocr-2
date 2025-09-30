from __future__ import annotations

"""Sidebar controls for the OCR inference Streamlit app.

Widget labels, defaults, and copy must come from ``configs/ui/inference.yaml``
and matching assets inside ``ui_meta/``. Review the Streamlit maintenance and
refactor protocols (``docs/ai_handbook/02_protocols/11_streamlit_maintenance_protocol.md``
and ``.../12_streamlit_refactoring_protocol.md``) before adjusting behaviour so
that configs and schemas stay authoritative.
"""

from collections.abc import Sequence

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from ..models.checkpoint import CheckpointMetadata
from ..models.config import SliderConfig, UIConfig
from ..models.ui_events import InferenceRequest
from ..state import InferenceState, init_hyperparameters, init_preprocessing


def render_controls(
    state: InferenceState,
    config: UIConfig,
    checkpoints: Sequence[CheckpointMetadata],
) -> InferenceRequest | None:
    init_hyperparameters(config.hyperparameters)
    init_preprocessing(config.preprocessing)

    selected_metadata = _render_model_selector(state, config, checkpoints)
    _render_model_status(selected_metadata, config)
    _render_hyperparameter_sliders(state, config)
    _render_preprocessing_controls(state, config)
    inference_request = _render_upload_section(state, selected_metadata, config)
    _render_clear_results(state)
    state.persist()

    return inference_request


def _render_model_selector(
    state: InferenceState,
    config: UIConfig,
    checkpoints: Sequence[CheckpointMetadata],
) -> CheckpointMetadata | None:
    st.subheader("Model Selection")

    options, mapping = _build_display_mapping(checkpoints, config)

    selected_label = st.selectbox(
        "Select Trained Model",
        options,
        index=options.index(state.selected_model) if state.selected_model in options else 0,
        help="Choose a trained OCR model for inference. Models are organized using metadata derived from checkpoints.",
    )

    metadata = mapping.get(selected_label)
    selected_model_path = metadata.checkpoint_path if metadata else selected_label
    state.reset_for_model(str(selected_model_path) if metadata else selected_label)

    return metadata


def _build_display_mapping(
    checkpoints: Sequence[CheckpointMetadata],
    config: UIConfig,
) -> tuple[list[str], dict[str, CheckpointMetadata | None]]:
    if not checkpoints:
        return [config.model_selector.demo_label], {config.model_selector.demo_label: None}

    options: list[str] = []
    mapping: dict[str, CheckpointMetadata | None] = {}

    for meta in checkpoints:
        label = meta.to_display_option()
        options.append(label)
        mapping[label] = meta

    return options, mapping


def _render_model_status(metadata: CheckpointMetadata | None, config: UIConfig) -> None:
    if metadata is None:
        st.warning(config.model_selector.empty_message)
        return

    if metadata.issues:
        for issue in metadata.issues:
            st.error(issue)
        st.stop()

    st.success(config.model_selector.success_message)
    with st.expander("Model metadata", expanded=False):
        st.json(metadata.to_dict())


def _render_hyperparameter_sliders(state: InferenceState, config: UIConfig) -> None:
    st.subheader("Inference Parameters")
    columns = st.columns(2)
    slider_items = list(config.hyperparameters.items())
    for index, (key, slider_cfg) in enumerate(slider_items):
        column = columns[index % len(columns)]
        with column:
            default_value = state.hyperparams.get(key, slider_cfg.default)
            value = _slider(slider_cfg, default_value)
            state.update_hyperparameter(key, value)
    state.persist()


def _render_preprocessing_controls(state: InferenceState, config: UIConfig) -> None:
    st.subheader("Preprocessing")
    enabled = st.checkbox(
        config.preprocessing.enable_label,
        value=state.preprocessing_enabled,
        help=config.preprocessing.enable_help,
    )
    state.preprocessing_enabled = bool(enabled)

    if state.preprocessing_enabled:
        try:
            from ocr.datasets.preprocessing import DOCTR_AVAILABLE

            if not DOCTR_AVAILABLE:
                st.warning(
                    "python-doctr is not installed. docTR preprocessing will fall back to OpenCV-only steps.",
                    icon="⚠️",
                )
        except Exception:
            st.warning(
                "Unable to verify python-doctr availability. Ensure it is installed for docTR preprocessing.",
                icon="⚠️",
            )


def _slider(slider_cfg: SliderConfig, default_value: float | int) -> float:
    kwargs = {
        "min_value": int(slider_cfg.min) if slider_cfg.is_integer_domain() else float(slider_cfg.min),
        "max_value": int(slider_cfg.max) if slider_cfg.is_integer_domain() else float(slider_cfg.max),
        "value": int(default_value) if slider_cfg.is_integer_domain() else float(default_value),
        "step": int(slider_cfg.step) if slider_cfg.is_integer_domain() else float(slider_cfg.step),
        "help": slider_cfg.help,
    }
    value = st.slider(slider_cfg.label, **kwargs)
    return float(value)


def _render_upload_section(
    state: InferenceState,
    metadata: CheckpointMetadata | None,
    config: UIConfig,
) -> InferenceRequest | None:
    st.subheader("Image Upload")

    uploaded_raw = st.file_uploader(
        "Upload Images",
        type=config.upload.enabled_file_types,
        accept_multiple_files=config.upload.multi_file_selection,
        help="Upload one or more images for OCR inference.",
    )

    if isinstance(uploaded_raw, UploadedFile):
        uploaded_files: list[UploadedFile] = [uploaded_raw]
    else:
        uploaded_files = list(uploaded_raw or [])

    if not uploaded_files:
        st.info("📤 Upload an image to get started.")
        return None

    if not metadata:
        st.info("📤 Models unavailable. Uploaded images will be kept in memory.")
        return None

    if len(uploaded_files) == 1 and config.upload.immediate_inference_for_single:
        file = uploaded_files[0]
        st.success("✅ 1 image uploaded and ready for inference")
        if st.button("🚀 Run Inference", width="stretch"):
            return InferenceRequest(
                files=[file],
                model_path=str(metadata.checkpoint_path),
                use_preprocessing=state.preprocessing_enabled,
                preprocessing_config=config.preprocessing,
            )
        return None

    _update_selected_images(state, uploaded_files)
    selected_files: list[UploadedFile] = [file for file in uploaded_files if file.name in state.selected_images]

    _render_selection_checkboxes(state, uploaded_files)

    if selected_files:
        st.success(f"✅ {len(selected_files)} of {len(uploaded_files)} images selected for inference")
        if st.button("🚀 Run Inference", width="stretch"):
            return InferenceRequest(
                files=selected_files,
                model_path=str(metadata.checkpoint_path),
                use_preprocessing=state.preprocessing_enabled,
                preprocessing_config=config.preprocessing,
            )
    else:
        st.warning("⚠️ No images selected for inference")

    return None


def _update_selected_images(state: InferenceState, uploaded_files: Sequence[UploadedFile]) -> None:
    filenames = {file.name for file in uploaded_files}
    previous = st.session_state.get("previous_uploaded_files", set())
    if previous != filenames:
        state.selected_images = set(filenames)
        st.session_state.previous_uploaded_files = filenames


def _render_selection_checkboxes(state: InferenceState, uploaded_files: Sequence[UploadedFile]) -> None:
    st.subheader("Select Images for Inference")
    st.markdown("Choose which images to run inference on:")
    for file in uploaded_files:
        key = f"select_{file.name}"
        is_selected = file.name in state.selected_images
        if st.checkbox(f"📄 {file.name}", value=is_selected, key=key):
            state.selected_images.add(file.name)
        else:
            state.selected_images.discard(file.name)


def _render_clear_results(state: InferenceState) -> None:
    if not state.inference_results:
        return
    st.divider()
    if st.button("🗑️ Clear Results", width="stretch"):
        state.inference_results.clear()
        state.processed_images.clear()
        state.persist()
    st.rerun()
