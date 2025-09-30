from __future__ import annotations

"""Results rendering utilities for the Streamlit UI.

Before tweaking layout or copy, review the configs in
``configs/ui/inference.yaml`` and the assets stored under
``ui_meta/inference/``. Align changes with the Streamlit maintenance and
refactor protocols documented in ``docs/ai_handbook/02_protocols``.
"""

from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from ..models.config import UIConfig
from ..state import InferenceState


def render_results(state: InferenceState, config: UIConfig) -> None:
    st.header("📊 Inference Results")

    if not state.inference_results:
        st.info("No inference results yet. Upload images and run inference to see results.")
        return

    if config.results.show_summary:
        _render_summary(state)
        st.divider()

    for index, result in enumerate(state.inference_results):
        title = f"Image {index + 1}: {result.get('filename', 'unknown')}"
        expanded = config.results.expand_first_result and index == 0
        with st.expander(title, expanded=expanded):
            _render_single_result(result, config)


def _render_summary(state: InferenceState) -> None:
    successful = [r for r in state.inference_results if r.get("success")]
    total_images = len(state.inference_results)
    successes = len(successful)
    failures = total_images - successes
    confidences = [conf for r in successful for conf in r.get("predictions", {}).get("confidences", [])]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Images", total_images)
    col2.metric("Successful", successes)
    col3.metric("Failed", failures)
    col4.metric("Avg. Confidence", f"{avg_confidence:.2%}")


def _render_single_result(result: dict[str, Any], config: UIConfig) -> None:
    if not result.get("success"):
        st.error(f"❌ Inference failed: {result.get('error', 'Unknown error')}")
        return

    predictions = result.get("predictions", {})
    confidences = predictions.get("confidences", [])
    num_detections = len(confidences)
    avg_confidence = sum(confidences) / num_detections if num_detections else 0

    col1, col2 = st.columns(2)
    col1.metric("Detections", str(num_detections))
    col2.metric("Avg. Confidence", f"{avg_confidence:.2%}")

    if "image" in result and predictions:
        _display_image_with_predictions(result["image"], predictions, config)

    if preprocessing := result.get("preprocessing"):
        _render_preprocessing_section(preprocessing, config)

    if config.results.show_raw_predictions and predictions:
        with st.expander("🔧 Raw Prediction Data"):
            st.json(predictions)


def _display_image_with_predictions(image_array: np.ndarray, predictions: dict[str, Any], config: UIConfig) -> None:
    try:
        pil_image = Image.fromarray(image_array)
        draw = ImageDraw.Draw(pil_image, "RGBA")

        if polygons_text := predictions.get("polygons", ""):
            polygons = polygons_text.split("|")
            texts = predictions.get("texts", [])
            confidences = predictions.get("confidences", [])

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
            width=width_setting,
        )
    except Exception as exc:  # noqa: BLE001
        st.error("Could not render predictions on the image. Displaying original image instead.")
        width_setting = "stretch" if config.results.image_width == "stretch" else "content"
        st.image(
            image_array,
            caption="Original Image",
            width=width_setting,
        )
        raise exc from exc


def _render_preprocessing_section(preprocessing: dict[str, Any], config: UIConfig) -> None:
    if not preprocessing.get("enabled") and not preprocessing.get("processed"):
        if error := preprocessing.get("error"):
            st.warning(f"docTR preprocessing unavailable: {error}")
        return

    st.markdown("#### 🧪 docTR Preprocessing")

    doctr_available = preprocessing.get("doctr_available", False)
    if preprocessing.get("enabled") and not doctr_available:
        st.warning("docTR geometry helpers are unavailable. Showing OpenCV-only preprocessing output.")

    original_image = preprocessing.get("original")
    processed_image = preprocessing.get("processed")
    metadata = preprocessing.get("metadata") or {}

    col_raw, col_processed = st.columns(2)

    if original_image is not None:
        overlay = _draw_document_overlay(original_image, metadata, config.preprocessing.show_corner_overlay)
        col_raw.image(overlay, caption="Original Upload", use_container_width=True)
    else:
        col_raw.info("Original image unavailable.")

    if processed_image is not None:
        col_processed.image(processed_image, caption="After docTR Preprocessing", use_container_width=True)
    else:
        col_processed.info("No preprocessed output available.")

    if config.preprocessing.show_metadata and metadata:
        prepared = _prepare_metadata(metadata)
        with st.expander("📋 Preprocessing Metadata", expanded=False):
            st.json(prepared)


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
