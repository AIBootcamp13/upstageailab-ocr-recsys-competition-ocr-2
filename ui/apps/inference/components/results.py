from __future__ import annotations

"""Results rendering utilities for the Streamlit UI.

Before tweaking layout or copy, review the configs in
``configs/ui/inference.yaml`` and the assets stored under
``ui_meta/inference/``. Align changes with the Streamlit maintenance and
refactor protocols documented in ``docs/ai_handbook/02_protocols``.
"""

# AI_DOCS[
#   bundle: streamlit-maintenance
#   priority: high
#   path: docs/ai_handbook/02_protocols/11_streamlit_maintenance_protocol.md#4-maintenance-checklist
#   path: docs/ai_handbook/02_protocols/05_modular_refactor.md#5-refactoring-checklist
#   path: docs/ai_handbook/02_protocols/02_command_registry.md#5-streamlit-ui-launchers
# ]

from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from ..models.config import UIConfig
from ..models.data_contracts import InferenceResult, Predictions, PreprocessingInfo
from ..state import InferenceState


def render_results(state: InferenceState, config: UIConfig) -> None:
    st.header("📊 Inference Results")

    if not state.inference_results:
        st.info("No inference results yet. Upload images and run inference to see results.")
        return

    # Check if we have batch prediction output files to display
    _render_batch_output_section(state)

    if config.results.show_summary:
        _render_summary(state)
        st.divider()

    # New table view for multiple results
    _render_results_table(state, config)
    st.divider()

    # Keep detailed expandable view
    st.subheader("📋 Detailed Results")
    for index, result in enumerate(state.inference_results):
        title = f"Image {index + 1}: {result.filename}"
        expanded = config.results.expand_first_result and index == 0
        with st.expander(title, expanded=expanded):
            _render_single_result(result, config)


def _render_batch_output_section(state: InferenceState) -> None:
    """Render batch prediction output files with download buttons."""
    from pathlib import Path

    if not state.batch_output_files:
        return

    st.subheader("📦 Batch Prediction Outputs")

    # Display download buttons for each output file
    for format_name, file_path in state.batch_output_files.items():
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            st.warning(f"⚠️ Output file not found: {file_path}")
            continue

        # Read file content for download button
        try:
            with open(file_path_obj, "rb") as f:
                file_content = f.read()

            # Determine MIME type
            mime_type = "application/json" if format_name.lower() == "json" else "text/csv"

            col1, col2 = st.columns([3, 1])
            with col1:
                st.success(f"✅ **{format_name.upper()}**: `{file_path}`")
            with col2:
                st.download_button(
                    label=f"⬇️ Download {format_name.upper()}",
                    data=file_content,
                    file_name=file_path_obj.name,
                    mime=mime_type,
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"❌ Error reading {format_name.upper()} file: {e}")

    st.divider()


def _render_summary(state: InferenceState) -> None:
    successful = [r for r in state.inference_results if r.success]
    total_images = len(state.inference_results)
    successes = len(successful)
    failures = total_images - successes
    confidences = [conf for r in successful for conf in r.predictions.confidences]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Images", total_images)
    col2.metric("Successful", successes)
    col3.metric("Failed", failures)
    col4.metric("Avg. Confidence", f"{avg_confidence:.2%}")


def _render_results_table(state: InferenceState, config: UIConfig) -> None:
    st.subheader("📋 Results Overview")

    if not state.inference_results:
        return

    # Create table data
    table_data = []

    for result in state.inference_results:
        filename = result.filename
        success = result.success

        if success:
            predictions = result.predictions
            confidences = predictions.confidences
            num_detections = len(confidences)
            avg_confidence = sum(confidences) / num_detections if num_detections else 0

            table_data.append(
                {
                    "Filename": filename,
                    "Status": "✅ Success",
                    "Detections": num_detections,
                    "Avg Confidence": f"{avg_confidence:.1%}",
                }
            )
        else:
            error = result.error or "Unknown error"
            table_data.append(
                {
                    "Filename": filename,
                    "Status": f"❌ Failed: {error[:50]}...",
                    "Detections": 0,
                    "Avg Confidence": "N/A",
                }
            )

    # Display as a clean table
    import pandas as pd

    df = pd.DataFrame(table_data)
    st.dataframe(df, width="stretch")


def _render_single_result(result: InferenceResult, config: UIConfig) -> None:
    if not result.success:
        st.error(f"❌ Inference failed: {result.error or 'Unknown error'}")
        return

    predictions = result.predictions
    confidences = predictions.confidences
    num_detections = len(confidences)
    avg_confidence = sum(confidences) / num_detections if num_detections else 0

    col1, col2 = st.columns(2)
    col1.metric("Detections", str(num_detections))
    col2.metric("Avg. Confidence", f"{avg_confidence:.2%}")

    if result.image is not None and predictions:
        _display_image_with_predictions(result.image, predictions, config)

    if result.preprocessing:
        _render_preprocessing_section(result.preprocessing, config)

    if config.results.show_raw_predictions and predictions:
        with st.expander("🔧 Raw Prediction Data"):
            st.json(predictions.model_dump())


def _display_image_with_predictions(image_array: np.ndarray, predictions: Predictions, config: UIConfig) -> None:
    try:
        pil_image = Image.fromarray(image_array)
        draw = ImageDraw.Draw(pil_image, "RGBA")

        if predictions.polygons:
            polygons = predictions.polygons.split("|")
            texts = predictions.texts
            confidences = predictions.confidences

            for index, polygon_str in enumerate(polygons):
                coords = [int(value) for value in polygon_str.split(",") if value]
                if len(coords) < 8 or len(coords) % 2 != 0:
                    continue
                points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]

                draw.polygon(points, outline=(255, 0, 0, 255), fill=(255, 0, 0, 30))

                if points:
                    min_x = min(point[0] for point in points)
                    min_y = min(point[1] for point in points)
                    label_text = texts[index] if index < len(texts) else f"Det_{index + 1}"
                    confidence = confidences[index] if index < len(confidences) else 0
                    label = f"{label_text} ({confidence:.1%})"
                    text_pos = (min_x, max(min_y - 20, 0))

                    try:
                        bbox = draw.textbbox(text_pos, label)
                        draw.rectangle(bbox, fill=(255, 0, 0, 180))
                        draw.text(text_pos, label, fill=(255, 255, 255, 255))
                    except AttributeError:
                        draw.text(text_pos, label, fill=(255, 0, 0, 255))

        width_setting = "stretch" if config.results.image_width == "stretch" else "content"
        st.image(
            pil_image,
            caption="OCR Predictions",
            width=width_setting,  # type: ignore[arg-type]
        )
    except Exception as exc:  # noqa: BLE001
        st.error("Could not render predictions on the image. Displaying original image instead.")
        width_setting = "stretch" if config.results.image_width == "stretch" else "content"
        st.image(
            image_array,
            caption="Original Image",
            width=width_setting,  # type: ignore[arg-type]
        )
        raise exc from exc


def _render_preprocessing_section(preprocessing: PreprocessingInfo, config: UIConfig) -> None:
    if not preprocessing.enabled and not preprocessing.processed:
        if preprocessing.error:
            st.warning(f"docTR preprocessing unavailable: {preprocessing.error}")
        return

    st.markdown("#### 🧪 docTR Preprocessing")

    if preprocessing.enabled and not preprocessing.doctr_available:
        st.warning("docTR geometry helpers are unavailable. Showing OpenCV-only preprocessing output.")

    original_image = preprocessing.original
    processed_image = preprocessing.processed
    metadata = preprocessing.metadata or {}

    col_raw, col_processed = st.columns(2)

    if original_image is not None:
        overlay = _draw_document_overlay(original_image, metadata, config.preprocessing.show_corner_overlay)
        col_raw.image(overlay, caption="Original Upload", width="stretch")
    else:
        col_raw.info("Original image unavailable.")

    if processed_image is not None:
        col_processed.image(processed_image, caption="After docTR Preprocessing", width="stretch")
    else:
        col_processed.info("No preprocessed output available.")

    if config.preprocessing.show_metadata and metadata:
        prepared = _prepare_metadata(metadata)
        with st.expander("📋 Preprocessing Metadata", expanded=False):
            st.json(prepared)

            # Show intermediate images for debugging
            _display_intermediate_images(metadata, config)


def _draw_document_overlay(image_array: np.ndarray, metadata: dict[str, Any], show_overlay: bool) -> Image.Image:
    base_image = Image.fromarray(image_array)
    if not show_overlay:
        return base_image

    corners = metadata.get("document_corners")
    if isinstance(corners, np.ndarray):
        corners_array = corners
    elif isinstance(corners, list):
        corners_array = np.asarray(corners)
    else:
        corners_array = None

    if corners_array is None or corners_array.size < 8:
        return base_image

    draw = ImageDraw.Draw(base_image, "RGBA")
    points = [(float(x), float(y)) for x, y in corners_array]
    draw.polygon(points, outline=(0, 255, 0, 255), fill=(0, 255, 0, 40))

    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    draw.ellipse(
        [
            (center_x - 3, center_y - 3),
            (center_x + 3, center_y + 3),
        ],
        fill=(0, 128, 0, 255),
    )

    return base_image


def _prepare_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, np.floating | np.integer):
            return float(value)
        elif isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        elif isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return {key: convert(val) for key, val in metadata.items()}


def _display_intermediate_images(metadata: dict[str, Any], config: UIConfig) -> None:
    """Display intermediate preprocessing images for debugging."""
    intermediate_images = [
        ("image_after_document_detection", "After Document Detection"),
        ("image_after_orientation_correction", "After Orientation Correction"),
        ("image_after_perspective_correction", "After Perspective Correction"),
        ("image_after_enhancement", "After Enhancement"),
    ]

    displayed_any = False
    for image_key, caption in intermediate_images:
        if image_key in metadata and metadata[image_key] is not None:
            if not displayed_any:
                st.markdown("#### 🔍 Intermediate Processing Steps")
                displayed_any = True

            col1, col2 = st.columns(2)
            with col1:
                st.image(metadata[image_key], caption=caption, width="stretch")

            # Show processing steps up to this point
            step_mapping = {
                "image_after_document_detection": ["document_detection"],
                "image_after_orientation_correction": ["document_detection", "orientation_correction"],
                "image_after_perspective_correction": ["document_detection", "orientation_correction", "perspective_correction"],
                "image_after_enhancement": ["document_detection", "orientation_correction", "perspective_correction", "image_enhancement"],
            }

            if image_key in step_mapping:
                processing_steps = metadata.get("processing_steps", [])
                completed_steps = [step for step in step_mapping[image_key] if step in processing_steps]
                with col2:
                    st.markdown(f"**Completed steps:** {', '.join(completed_steps)}")

                    # Add specific metadata for perspective correction
                    if image_key == "image_after_perspective_correction":
                        if "perspective_matrix" in metadata:
                            st.markdown("**Perspective matrix applied** ✅")
                        if "document_corners" in metadata:
                            corners = metadata["document_corners"]
                            if hasattr(corners, "shape"):
                                st.markdown(f"**Document corners detected:** {corners.shape[0]} corners")

    if displayed_any:
        st.markdown("---")
        st.markdown(
            "💡 **Debug Tip:** Check the 'After Perspective Correction' image. If it looks distorted or only shows part of the document, that's likely causing the OCR predictions to be clustered at the bottom."
        )
