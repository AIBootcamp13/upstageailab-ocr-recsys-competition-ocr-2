from __future__ import annotations

"""Model comparison components."""

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from ui.data_utils import (
    calculate_image_differences,
    calculate_model_metrics,
    calculate_total_area,
    find_common_images,
)

from .helpers import draw_predictions_on_image


def display_model_comparison_stats(df_a: pd.DataFrame, df_b: pd.DataFrame) -> None:
    """Display statistical comparison between two models."""
    st.subheader("📊 Model Comparison Statistics")

    metrics_a = calculate_model_metrics(df_a)
    metrics_b = calculate_model_metrics(df_b)

    comparison_data = {
        "Metric": [
            "Total Predictions",
            "Avg Predictions/Image",
            "Images with Predictions",
            "Empty Predictions",
        ],
        "Model A": [
            str(metrics_a.total_predictions),
            f"{metrics_a.avg_predictions:.1f}",
            str(metrics_a.images_with_predictions),
            str(metrics_a.empty_predictions),
        ],
        "Model B": [
            str(metrics_b.total_predictions),
            f"{metrics_b.avg_predictions:.1f}",
            str(metrics_b.images_with_predictions),
            str(metrics_b.empty_predictions),
        ],
        "Difference": [
            str(metrics_b.total_predictions - metrics_a.total_predictions),
            f"{metrics_b.avg_predictions - metrics_a.avg_predictions:.1f}",
            str(metrics_b.images_with_predictions - metrics_a.images_with_predictions),
            str(metrics_b.empty_predictions - metrics_a.empty_predictions),
        ],
    }
    st.table(pd.DataFrame(comparison_data))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delta1 = metrics_b.total_predictions - metrics_a.total_predictions
        st.metric("Total Predictions", metrics_b.total_predictions, delta=delta1)
    with col2:
        delta2 = metrics_b.avg_predictions - metrics_a.avg_predictions
        st.metric("Avg per Image", f"{metrics_b.avg_predictions:.1f}", delta=f"{delta2:.1f}")
    with col3:
        delta3 = metrics_b.images_with_predictions - metrics_a.images_with_predictions
        st.metric("Images w/ Preds", metrics_b.images_with_predictions, delta=delta3)
    with col4:
        delta4 = int(metrics_a.empty_predictions - metrics_b.empty_predictions)
        st.metric("Empty Preds", metrics_b.empty_predictions, delta=-delta4)


def display_visual_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame, image_dir: str, gt_df: pd.DataFrame | None = None) -> None:
    """Display visual comparison of predictions on the same images."""
    st.subheader("🖼️ Visual Comparison")

    common_images = find_common_images(df_a, df_b)
    if not common_images:
        st.warning("No common images found between the two models.")
        return

    if "visual_comparison_page" not in st.session_state:
        st.session_state.visual_comparison_page = 0

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅️ Previous", disabled=st.session_state.visual_comparison_page == 0):
            st.session_state.visual_comparison_page -= 1

    with col2:
        current_image = common_images[st.session_state.visual_comparison_page]
        st.markdown(f"**Image {st.session_state.visual_comparison_page + 1} of {len(common_images)}**")
        st.markdown(f"**{current_image}**")

    with col3:
        if st.button(
            "Next ➡️",
            disabled=st.session_state.visual_comparison_page >= len(common_images) - 1,
        ):
            st.session_state.visual_comparison_page += 1

    display_side_by_side_comparison(df_a, df_b, current_image, image_dir, gt_df)


def display_side_by_side_comparison(
    df_a: pd.DataFrame, df_b: pd.DataFrame, image_name: str, image_dir: str, gt_df: pd.DataFrame | None = None
) -> None:
    """Display side-by-side comparison of two models on the same image."""
    image_path = Path(image_dir) / image_name
    if not image_path.exists():
        st.error(f"Image not found: {image_path}")
        return

    try:
        original_image = Image.open(image_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error creating comparison: {exc}")
        return

    row_a = df_a[df_a["filename"] == image_name].iloc[0]
    row_b = df_b[df_b["filename"] == image_name].iloc[0]
    gt_row = None
    if gt_df is not None:
        gt_matches = gt_df[gt_df["filename"] == image_name]
        if not gt_matches.empty:
            gt_row = gt_matches.iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original Image**")
        if gt_row is not None:
            img_gt = draw_predictions_on_image(original_image.copy(), str(gt_row.get("polygons", "")), (0, 255, 0), rotate_ccw90=True)
            st.image(img_gt, caption="Original with GT")
        else:
            st.image(original_image.rotate(-90, expand=True), caption="Original")

    with col2:
        st.markdown("**Model A Predictions**")
        img_a = draw_predictions_on_image(original_image.copy(), str(row_a.get("polygons", "")), (255, 0, 0), rotate_ccw90=True)
        pred_count_a = len(str(row_a.get("polygons", "")).split("|")) if pd.notna(row_a.get("polygons")) else 0
        conf_a = row_a.get("avg_confidence", 0.8)
        st.image(img_a, caption=f"Model A ({pred_count_a} predictions, conf: {conf_a:.2f})")

    with col3:
        st.markdown("**Model B Predictions**")
        img_b = draw_predictions_on_image(
            original_image.copy(), str(row_b.get("polygons", "")), (0, 0, 255), rotate_ccw90=True
        )  # Changed to blue
        pred_count_b = len(str(row_b.get("polygons", "")).split("|")) if pd.notna(row_b.get("polygons")) else 0
        conf_b = row_b.get("avg_confidence", 0.8)
        st.image(img_b, caption=f"Model B ({pred_count_b} predictions, conf: {conf_b:.2f})")

    st.markdown("### Comparison Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Prediction Count", pred_count_b, delta=pred_count_b - pred_count_a)

    with col2:
        area_a = calculate_total_area(str(row_a.get("polygons", "")))
        area_b = calculate_total_area(str(row_b.get("polygons", "")))
        st.metric("Total Area", f"{area_b:.0f}", delta=f"{area_b - area_a:.0f}")

    with col3:
        st.metric("Confidence", f"{conf_b:.2f}", delta=f"{conf_b - conf_a:.2f}")


def display_model_differences(df_a: pd.DataFrame, df_b: pd.DataFrame) -> None:
    """Calculates and displays the differences between two models' predictions."""
    st.subheader("Prediction Differences")

    if "pred_page" not in st.session_state:
        st.session_state.pred_page = 0
    if "area_page" not in st.session_state:
        st.session_state.area_page = 0
    if "conf_page" not in st.session_state:
        st.session_state.conf_page = 0

    try:
        diff_df = calculate_image_differences(df_a, df_b)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error calculating prediction differences: {exc}")
        return

    if diff_df.empty:
        st.warning("No common images found to compare between the two models.")
        return

    items_per_page = 20

    st.markdown("#### Top Differences in Prediction Count")
    _render_difference_table(
        diff_df,
        column="abs_pred_diff",
        value_columns=["filename", "pred_a", "pred_b", "pred_diff"],
        page_key="pred_page",
        order_key="pred_order",
        default_order="Largest",
        items_per_page=items_per_page,
    )

    st.markdown("#### Top Differences in Total Prediction Area")
    _render_difference_table(
        diff_df,
        column="abs_area_diff",
        value_columns=["filename", "area_a", "area_b", "area_diff"],
        page_key="area_page",
        order_key="area_order",
        default_order="Largest",
        items_per_page=items_per_page,
    )

    st.markdown("#### Top Differences in Prediction Confidence")
    _render_difference_table(
        diff_df,
        column="abs_conf_diff",
        value_columns=["filename", "conf_a", "conf_b", "conf_diff"],
        page_key="conf_page",
        order_key="conf_order",
        default_order="Largest",
        items_per_page=items_per_page,
    )


def _render_difference_table(
    df: pd.DataFrame,
    *,
    column: str,
    value_columns: list[str],
    page_key: str,
    order_key: str,
    default_order: str,
    items_per_page: int,
) -> None:
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        order = st.radio("Order", ["Largest", "Smallest"], key=order_key, horizontal=True, index=0 if default_order == "Largest" else 1)
    with col2:
        if st.button("⬅️ Previous", key=f"{page_key}_prev", disabled=st.session_state[page_key] == 0):
            st.session_state[page_key] -= 1
    with col3:
        total_pages = (len(df) + items_per_page - 1) // items_per_page
        if st.button(
            "Next ➡️",
            key=f"{page_key}_next",
            disabled=st.session_state[page_key] >= total_pages - 1,
        ):
            st.session_state[page_key] += 1

    ascending = order == "Smallest"
    start_idx = st.session_state[page_key] * items_per_page
    end_idx = min(start_idx + items_per_page, len(df))
    sorted_df = df.sort_values(column, ascending=ascending)
    st.dataframe(sorted_df.iloc[start_idx:end_idx][value_columns])
