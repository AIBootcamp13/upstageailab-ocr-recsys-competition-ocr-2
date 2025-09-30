from __future__ import annotations

"""Image viewer utilities."""

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from .helpers import draw_predictions_on_image, parse_polygon_string


def display_image_viewer(df: pd.DataFrame, image_dir: str) -> None:
    """Display image viewer with predictions and pagination."""
    st.subheader("🖼️ Image Viewer")

    if "image_viewer_page" not in st.session_state:
        st.session_state.image_viewer_page = 0

    total_images = len(df)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button(
            "⬅️ Previous Image",
            key="viewer_prev",
            disabled=st.session_state.image_viewer_page == 0,
        ):
            st.session_state.image_viewer_page -= 1

    with col2:
        current_idx, current_row = _current_viewer_row(df)
        st.markdown(f"**Image {current_idx + 1} of {total_images}**")
        st.markdown(f"**{current_row['filename']}**")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Predictions", current_row["prediction_count"])
        with col_b:
            st.metric("Total Area", f"{current_row['total_area']:.0f}")
        with col_c:
            st.metric("Confidence", f"{current_row['avg_confidence']:.2f}")

    with col3:
        if st.button(
            "Next Image ➡️",
            key="viewer_next",
            disabled=st.session_state.image_viewer_page >= total_images - 1,
        ):
            st.session_state.image_viewer_page += 1

    selected_image = current_row["filename"]
    display_image_with_predictions(df, selected_image, image_dir)


def display_image_with_predictions(df: pd.DataFrame, image_name: str, image_dir: str) -> None:
    """Display a single image with its predictions."""
    image_path = Path(image_dir) / image_name
    if not image_path.exists():
        st.error(f"Image not found: {image_path}")
        return

    try:
        image = Image.open(image_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error loading image: {exc}")
        return

    row = df[df["filename"] == image_name].iloc[0]
    polygons_str = str(row.get("polygons", ""))

    if polygons := parse_polygon_string(polygons_str):
        annotated = draw_predictions_on_image(image, polygons_str, (255, 0, 0))
        st.image(annotated, caption=f"Predictions on {image_name}")
        st.info(f"Found {len(polygons)} valid text regions in this image (out of {len(polygons)} total)")
        _render_enlarge_toggle(
            image_name,
            annotated,
            polygons_count=len(polygons),
            total_polygons=len(polygons),
            confidence=row.get("avg_confidence", 0.8),
        )
    else:
        st.image(image, caption=image_name)
        st.info("No predictions found for this image")
        _render_enlarge_toggle(
            image_name,
            image,
            polygons_count=0,
            total_polygons=0,
            confidence=row.get("avg_confidence", 0.8),
        )


def display_image_grid(df: pd.DataFrame, image_dir: str, sort_metric: str, max_images: int = 10, start_idx: int = 0) -> None:
    """Display a grid of images with their metrics."""
    end_idx = min(start_idx + max_images, len(df))
    st.markdown(f"### Images {start_idx + 1}-{end_idx} of {len(df)} ({sort_metric})")

    images_to_show = df.iloc[start_idx:end_idx]
    cols = st.columns(5)

    for i, (_, row) in enumerate(images_to_show.iterrows()):
        col_idx = i % 5
        with cols[col_idx]:
            image_path = Path(image_dir) / row["filename"]
            if not image_path.exists():
                st.error(f"Image not found: {row['filename']}")
                continue

            try:
                image = Image.open(image_path)
                image.thumbnail((200, 200))
                st.image(image, caption=f"{row['filename'][:15]}...")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error loading {row['filename']}: {exc}")
                continue

            if sort_metric in row and sort_metric != "filename":
                value = row[sort_metric]
                if isinstance(value, int | float):
                    st.metric(sort_metric, f"{value:.2f}")
                else:
                    st.metric(sort_metric, str(value))


def _render_enlarge_toggle(
    image_name: str,
    image: Image.Image,
    polygons_count: int,
    total_polygons: int,
    confidence: float,
) -> None:
    key = f"enlarge_{image_name}"
    if st.button("🔍 Click to Enlarge", key=key):
        st.session_state[key] = not st.session_state.get(key, False)

    if st.session_state.get(key, False):
        st.markdown("### 🔍 Enlarged View")
        col1, col2 = st.columns([4, 1])

        with col1:
            st.image(image, caption=f"Enlarged: {image_name}", use_column_width=True)

        with col2:
            st.markdown("**Image Details:**")
            st.write(f"**Filename:** {image_name}")
            st.write(f"**Valid Polygons:** {polygons_count}")
            st.write(f"**Total Polygons:** {total_polygons}")
            st.write(f"**Confidence:** {confidence:.3f}")

            if st.button("❌ Close Enlarged View", key=f"close_{image_name}"):
                st.session_state[key] = False


def _current_viewer_row(df: pd.DataFrame) -> tuple[int, pd.Series]:
    index = st.session_state.image_viewer_page
    index = max(0, min(index, len(df) - 1))
    st.session_state.image_viewer_page = index
    return index, df.iloc[index]
