from __future__ import annotations

"""Image viewer utilities."""

# AI_DOCS[
#   bundle: streamlit-maintenance
#   priority: medium
#   path: docs/ai_handbook/02_protocols/11_streamlit_maintenance_protocol.md#5-maintenance-playbook
#   path: docs/ai_handbook/02_protocols/05_modular_refactor.md#2-when-to-use-this-protocol
# ]

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from ocr.utils.orientation import normalize_pil_image, remap_polygons

from .helpers import validate_polygons


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
        pil_image = Image.open(image_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error loading image: {exc}")
        return

    raw_width, raw_height = pil_image.size
    normalized_image, orientation = normalize_pil_image(pil_image)

    canonical_width, canonical_height = normalized_image.size

    if normalized_image.mode != "RGB":
        image = normalized_image.convert("RGB")
    else:
        image = normalized_image.copy()

    if normalized_image is not pil_image and normalized_image is not image:
        normalized_image.close()
    pil_image.close()

    row = df[df["filename"] == image_name].iloc[0]
    polygons_str = str(row.get("polygons", ""))

    polygons, validation_stats = validate_polygons(polygons_str)

    if orientation != 1 and polygons:
        np_polygons = [np.array(coords, dtype=np.float32).reshape(1, -1, 2) for coords in polygons]
        remapped = remap_polygons(np_polygons, canonical_width, canonical_height, orientation)
        polygons = [poly.reshape(-1).tolist() for poly in remapped]

    if polygons:
        # Draw predictions on the rotated image
        annotated = _draw_predictions_from_list(image, polygons, (255, 0, 0))
        st.image(annotated, caption=f"Predictions on {image_name}")

        # Display validation summary
        if validation_stats["total"] > validation_stats["valid"]:
            st.info(f"Found {validation_stats['valid']} valid text regions in this image (out of {validation_stats['total']} total)")
            _render_polygon_validation_report(validation_stats)
        else:
            st.info(f"Found {validation_stats['valid']} valid text regions in this image")

        _render_enlarge_toggle(
            image_name,
            annotated,
            polygons_count=validation_stats["valid"],
            total_polygons=validation_stats["total"],
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
            st.image(image, caption=f"Enlarged: {image_name}", width="stretch")

        with col2:
            st.markdown("**Image Details:**")
            st.write(f"**Filename:** {image_name}")
            st.write(f"**Valid Polygons:** {polygons_count}")
            st.write(f"**Total Polygons:** {total_polygons}")
            st.write(f"**Confidence:** {confidence:.3f}")

            if st.button("❌ Close Enlarged View", key=f"close_{image_name}"):
                st.session_state[key] = False


def _render_polygon_validation_report(validation_stats: dict[str, int]) -> None:
    """Render detailed polygon validation report."""
    with st.expander("🔍 Validation Details"):
        st.write("**Polygon Validation Summary:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Polygons", validation_stats["total"])
            st.metric("Valid Polygons", validation_stats["valid"])
        with col2:
            filtered = validation_stats["total"] - validation_stats["valid"]
            st.metric("Filtered Out", filtered)
            if filtered > 0:
                st.metric("Success Rate", f"{validation_stats['valid'] / validation_stats['total']:.1%}")

        if validation_stats["too_small"] > 0:
            st.write(f"• **Too small** (<4 points): {validation_stats['too_small']} polygons")
        if validation_stats["odd_coords"] > 0:
            st.write(f"• **Odd coordinates**: {validation_stats['odd_coords']} polygons")
        if validation_stats["parse_errors"] > 0:
            st.write(f"• **Parse errors**: {validation_stats['parse_errors']} polygons")


def _current_viewer_row(df: pd.DataFrame) -> tuple[int, pd.Series]:
    index = st.session_state.image_viewer_page
    index = max(0, min(index, len(df) - 1))
    st.session_state.image_viewer_page = index
    return index, df.iloc[index]


def _draw_predictions_from_list(image: Image.Image, polygons: list[list[float]], color: tuple[int, int, int]) -> Image.Image:
    """Draw polygon predictions from a list on a copy of the image."""
    if not polygons:
        return image

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    for index, coords in enumerate(polygons):
        points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        try:
            draw.polygon(points, outline=color + (255,), fill=color + (50,), width=2)
        except TypeError:
            draw.polygon(points, outline=color + (255,), fill=color + (50,))
        if points:
            draw.text((points[0][0], points[0][1] - 10), f"T{index + 1}", fill=color + (255,))

    return overlay
