from __future__ import annotations

"""Sidebar controls for the OCR inference Streamlit app.

Widget labels, defaults, and copy must come from ``configs/ui/inference.yaml``
and matching assets inside ``ui_meta/``. Review the Streamlit maintenance and
refactor protocols (``docs/ai_handbook/02_protocols/11_streamlit_maintenance_protocol.md``
and ``.../12_streamlit_refactoring_protocol.md``) before adjusting behaviour so
that configs and schemas stay authoritative.
"""

from collections.abc import Sequence
from pathlib import Path

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from ..models.batch_request import (
    BatchHyperparameters,
    BatchOutputConfig,
    BatchPredictionRequest,
)
from ..models.checkpoint import CheckpointInfo, CheckpointMetadata
from ..models.config import SliderConfig, UIConfig
from ..models.ui_events import InferenceRequest
from ..state import InferenceState, clear_session_state, init_hyperparameters, init_preprocessing


def render_controls(
    state: InferenceState,
    config: UIConfig,
    checkpoints: Sequence[CheckpointInfo],
) -> InferenceRequest | BatchPredictionRequest | None:
    init_hyperparameters(config.hyperparameters)
    init_preprocessing(config.preprocessing)

    # Mode selector at the top
    _render_mode_selector(state)
    _render_session_controls(state)

    selected_metadata = _render_model_selector(state, config, checkpoints)
    _render_model_status(selected_metadata, config)
    _render_hyperparameter_sliders(state, config)
    _render_preprocessing_controls(state, config)

    # Render either single image upload or batch mode controls
    if state.batch_mode:
        batch_request = _render_batch_mode_controls(state, selected_metadata, config)
        _render_clear_results(state)
        state.persist()
        return batch_request
    else:
        inference_request = _render_upload_section(state, selected_metadata, config)
        _render_clear_results(state)
        state.persist()
        return inference_request


def _render_mode_selector(state: InferenceState) -> None:
    """Render mode toggle between Single Image and Batch Prediction."""
    st.subheader("Processing Mode")
    mode = st.radio(
        "Select mode",
        options=["Single Image", "Batch Prediction"],
        index=1 if state.batch_mode else 0,
        help="Choose between processing individual images or batch processing a directory",
        horizontal=True,
    )
    state.batch_mode = mode == "Batch Prediction"
    st.divider()


def _render_session_controls(state: InferenceState) -> None:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Results", key="session_clear_results", use_container_width=True):
            _clear_inference_results(state)
            st.rerun()
    with col2:
        if st.button("♻️ Reset Session", key="session_reset", use_container_width=True):
            clear_session_state()
            st.rerun()
    st.divider()


def _clear_inference_results(state: InferenceState) -> None:
    state.inference_results.clear()
    state.selected_images.clear()
    state.processed_images.clear()
    state.batch_output_files.clear()
    state.persist()


def _clear_image_list(state: InferenceState) -> None:
    """Clear only the uploaded image list while keeping inference results."""
    state.selected_images.clear()
    state.processed_images.clear()
    # Clear the previous_uploaded_files from session state to reset file uploader
    if "previous_uploaded_files" in st.session_state:
        st.session_state.previous_uploaded_files = set()
    state.persist()


def _render_batch_mode_controls(
    state: InferenceState,
    metadata: CheckpointMetadata | None,
    config: UIConfig,
) -> BatchPredictionRequest | None:
    """Render batch prediction mode controls."""
    st.subheader("Batch Prediction")

    if metadata is None:
        st.warning("⚠️ No model selected. Please select a model first.")
        return None

    # Input directory
    input_dir = st.text_input(
        "Input Directory",
        value=state.batch_input_dir,
        help="Path to directory containing images to process",
        placeholder="/path/to/images",
    )
    state.batch_input_dir = input_dir

    # Validate input directory in real-time
    input_dir_valid = False
    if input_dir:
        input_path = Path(input_dir)
        if not input_path.exists():
            st.error(f"❌ Directory does not exist: {input_dir}")
        elif not input_path.is_dir():
            st.error(f"❌ Path is not a directory: {input_dir}")
        else:
            # Try to count images
            try:
                supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
                image_files: list[Path] = []
                for ext in supported_exts:
                    image_files.extend(input_path.glob(f"*{ext}"))
                    image_files.extend(input_path.glob(f"*{ext.upper()}"))
                image_count = len(set(image_files))
                if image_count == 0:
                    st.warning("⚠️ No image files found in directory")
                else:
                    st.success(f"✅ Found {image_count} images in directory")
                    input_dir_valid = True
            except Exception as e:
                st.error(f"❌ Error reading directory: {e}")

    # Output configuration
    st.subheader("Output Configuration")

    output_dir = st.text_input(
        "Output Directory",
        value=state.batch_output_dir,
        help="Directory where submission files will be saved",
    )
    state.batch_output_dir = output_dir

    filename_prefix = st.text_input(
        "Filename Prefix",
        value=state.batch_filename_prefix,
        help="Prefix for output files (timestamp will be appended)",
    )
    state.batch_filename_prefix = filename_prefix

    col1, col2 = st.columns(2)
    with col1:
        save_json = st.checkbox(
            "Save JSON",
            value=state.batch_save_json,
            help="Generate JSON submission file",
        )
        state.batch_save_json = save_json

    with col2:
        save_csv = st.checkbox(
            "Save CSV",
            value=state.batch_save_csv,
            help="Generate CSV submission file",
        )
        state.batch_save_csv = save_csv

    # Confidence score option
    include_confidence = st.checkbox(
        "Include confidence scores",
        value=state.batch_include_confidence,
        help="Add average confidence score as third column (optional per competition format)",
    )
    state.batch_include_confidence = include_confidence

    # Validate at least one output format is selected
    if not save_json and not save_csv:
        st.error("❌ At least one output format (JSON or CSV) must be selected")

    # Run batch prediction button
    st.divider()
    can_run = input_dir_valid and (save_json or save_csv) and filename_prefix.strip()

    if not can_run:
        st.button("🚀 Run Batch Prediction", disabled=True, help="Please complete all required fields")
        return None

    if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
        try:
            # Build BatchPredictionRequest with validation
            request = BatchPredictionRequest(
                input_dir=input_dir,
                model_path=str(metadata.checkpoint_path),
                config_path=str(metadata.config_path) if metadata.config_path else None,
                use_preprocessing=state.preprocessing_enabled,
                output_config=BatchOutputConfig(
                    output_dir=output_dir,
                    filename_prefix=filename_prefix,
                    save_json=save_json,
                    save_csv=save_csv,
                    include_confidence=include_confidence,
                ),
                hyperparameters=BatchHyperparameters(
                    binarization_thresh=state.hyperparams.get("binarization_thresh", 0.3),
                    box_thresh=state.hyperparams.get("box_thresh", 0.7),
                    max_candidates=int(state.hyperparams.get("max_candidates", 1000)),
                    min_detection_size=int(state.hyperparams.get("min_detection_size", 3)),
                ),
            )
            return request
        except Exception as e:
            st.error(f"❌ Validation error: {e}")
            return None

    return None


def _render_model_selector(
    state: InferenceState,
    config: UIConfig,
    checkpoints: Sequence[CheckpointInfo],
) -> CheckpointMetadata | None:
    st.subheader("Model Selection")

    options, mapping = _build_display_mapping(checkpoints, config)

    if state.selected_model_label in options:
        default_index = options.index(state.selected_model_label)
    else:
        default_index = 0

    selected_label = st.selectbox(
        "Select Trained Model",
        options,
        index=default_index,
        help="Choose a trained OCR model for inference. Models are organized using metadata derived from checkpoints.",
    )

    info = mapping.get(selected_label)
    if info is None:
        state.reset_for_model(None, selected_label)
        state.persist()
        return None

    # Load full metadata for the selected checkpoint
    with st.spinner("Loading model metadata..."):
        from ..services.schema_validator import load_schema

        schema = load_schema()
        metadata = info.load_full_metadata(schema)

    selected_model_path = metadata.checkpoint_path
    state.reset_for_model(str(selected_model_path), selected_label)
    state.persist()

    return metadata


def _build_display_mapping(
    checkpoints: Sequence[CheckpointInfo],
    config: UIConfig,
) -> tuple[list[str], dict[str, CheckpointInfo | None]]:
    if not checkpoints:
        return [config.model_selector.demo_label], {config.model_selector.demo_label: None}

    options: list[str] = []
    mapping: dict[str, CheckpointInfo | None] = {}

    for info in checkpoints:
        label = info.to_display_option()
        options.append(label)
        mapping[label] = info

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

        overrides = state.preprocessing_overrides
        base = config.preprocessing
        with st.expander("docTR options", expanded=False):
            st.caption("Tune docTR preprocessing before running inference. These values apply per session.")

            enable_document_detection = st.checkbox(
                "Enable document detection",
                value=overrides.get("enable_document_detection", base.enable_document_detection),
                help="Detect document boundaries and apply geometric corrections.",
            )
            state.update_preprocessing_override("enable_document_detection", enable_document_detection)

            # Document detection options (shown when document detection is enabled)
            if enable_document_detection:
                angle_threshold = st.slider(
                    "Orientation threshold (degrees)",
                    min_value=0.1,
                    max_value=15.0,
                    step=0.1,
                    value=float(overrides.get("orientation_angle_threshold", base.orientation_angle_threshold)),
                )
                state.update_preprocessing_override("orientation_angle_threshold", float(angle_threshold))

                expand_canvas = st.checkbox(
                    "Allow canvas expansion while rotating",
                    value=overrides.get("orientation_expand_canvas", base.orientation_expand_canvas),
                )
                preserve_shape = st.checkbox(
                    "Preserve original shape after rotation",
                    value=overrides.get("orientation_preserve_original_shape", base.orientation_preserve_original_shape),
                )
                state.update_preprocessing_override("orientation_expand_canvas", expand_canvas)
                state.update_preprocessing_override("orientation_preserve_original_shape", preserve_shape)

                use_doctr_geometry = st.checkbox(
                    "Use docTR rcrop geometry",
                    value=overrides.get("use_doctr_geometry", base.use_doctr_geometry),
                    help="Prefer docTR's perspective correction before falling back to OpenCV.",
                )
                state.update_preprocessing_override("use_doctr_geometry", use_doctr_geometry)

                padding_cleanup = st.checkbox(
                    "Remove padding after warp",
                    value=overrides.get("enable_padding_cleanup", base.enable_padding_cleanup),
                )
                state.update_preprocessing_override("enable_padding_cleanup", padding_cleanup)

                document_detection_min_area = st.slider(
                    "Minimum document area (ratio)",
                    min_value=0.05,
                    max_value=0.6,
                    step=0.01,
                    value=float(
                        overrides.get(
                            "document_detection_min_area_ratio",
                            base.document_detection_min_area_ratio,
                        )
                    ),
                    help="Ignore contours smaller than this fraction of the image when hunting for page boundaries.",
                )
                state.update_preprocessing_override("document_detection_min_area_ratio", float(document_detection_min_area))

                detection_use_adaptive = st.checkbox(
                    "Use adaptive threshold fallback",
                    value=overrides.get("document_detection_use_adaptive", base.document_detection_use_adaptive),
                    help="Apply adaptive thresholding when the primary edge detector misses the page.",
                )
                detection_use_box = st.checkbox(
                    "Use bounding-box fallback",
                    value=overrides.get("document_detection_use_fallback_box", base.document_detection_use_fallback_box),
                    help="Fallback to the largest content bounding box if no contour is found.",
                )
                state.update_preprocessing_override("document_detection_use_adaptive", detection_use_adaptive)
                state.update_preprocessing_override("document_detection_use_fallback_box", detection_use_box)

                camscanner = st.checkbox(
                    "Use CamScanner-style detection",
                    value=overrides.get("document_detection_use_camscanner", base.document_detection_use_camscanner),
                    help="Use advanced CamScanner-style LSD line detection for precise document boundary detection. More accurate than basic edge detection.",
                )
                state.update_preprocessing_override("document_detection_use_camscanner", camscanner)

                enable_orientation = st.checkbox(
                    "Enable orientation correction",
                    value=overrides.get("enable_orientation_correction", base.enable_orientation_correction) and not camscanner,
                    disabled=camscanner,
                    help="Rotate pages using docTR's angle estimate before rectifying corners. Disabled when CamScanner is used as it provides better orientation detection."
                    if camscanner
                    else "Rotate pages using docTR's angle estimate before rectifying corners.",
                )
                state.update_preprocessing_override("enable_orientation_correction", enable_orientation)

            enhancement_enabled = st.checkbox(
                "Enable photometric enhancement",
                value=overrides.get("enable_enhancement", base.enable_enhancement),
            )
            state.update_preprocessing_override("enable_enhancement", enhancement_enabled)

            enhancement_method = st.selectbox(
                "Enhancement method",
                options=["conservative", "office_lens"],
                index=["conservative", "office_lens"].index(overrides.get("enhancement_method", base.enhancement_method)),
            )
            state.update_preprocessing_override("enhancement_method", enhancement_method)

            final_resize = st.checkbox(
                "Resize output to target canvas",
                value=overrides.get("enable_final_resize", base.enable_final_resize),
                help="When disabled, docTR returns the rectified page at its native resolution instead of padding to 640×640.",
            )
            state.update_preprocessing_override("enable_final_resize", final_resize)


def _slider(slider_cfg: SliderConfig, default_value: float | int) -> float:
    kwargs = {
        "min_value": int(slider_cfg.min) if slider_cfg.is_integer_domain() else float(slider_cfg.min),
        "max_value": int(slider_cfg.max) if slider_cfg.is_integer_domain() else float(slider_cfg.max),
        "value": int(default_value) if slider_cfg.is_integer_domain() else float(default_value),
        "step": int(slider_cfg.step) if slider_cfg.is_integer_domain() else float(slider_cfg.step),
        "help": slider_cfg.help,
    }
    value = st.slider(slider_cfg.label, **kwargs)  # type: ignore[call-overload]
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
    )  # type: ignore[call-overload]

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
        if st.button("🚀 Run Inference", type="primary", use_container_width=True):
            return InferenceRequest(
                files=[file],
                model_path=str(metadata.checkpoint_path),
                config_path=str(metadata.config_path) if metadata.config_path else None,
                use_preprocessing=state.preprocessing_enabled,
                preprocessing_config=state.build_preprocessing_config(config.preprocessing),
            )
        return None

    _update_selected_images(state, uploaded_files)
    selected_files: list[UploadedFile] = [file for file in uploaded_files if file.name in state.selected_images]

    _render_selection_checkboxes(state, uploaded_files)

    if selected_files:
        st.success(f"✅ {len(selected_files)} of {len(uploaded_files)} images selected for inference")
        if st.button("🚀 Run Inference", type="primary", use_container_width=True):
            return InferenceRequest(
                files=selected_files,
                model_path=str(metadata.checkpoint_path),
                config_path=str(metadata.config_path) if metadata.config_path else None,
                use_preprocessing=state.preprocessing_enabled,
                preprocessing_config=state.build_preprocessing_config(config.preprocessing),
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
    if not (state.inference_results or state.processed_images or state.selected_images):
        return
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Clear Results", use_container_width=True):
            _clear_inference_results(state)
            st.rerun()
    with col2:
        if st.button("📋 Clear Image List", use_container_width=True):
            _clear_image_list(state)
            st.rerun()
    with col3:
        if st.button("♻️ Reset Session", use_container_width=True):
            clear_session_state()
            st.rerun()
