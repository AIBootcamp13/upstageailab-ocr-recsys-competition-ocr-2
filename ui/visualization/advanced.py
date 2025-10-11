from __future__ import annotations

"""Advanced analysis tools for the visualization suite."""

from pathlib import Path

import pandas as pd
import streamlit as st

from ui.data_utils import apply_sorting_filtering, calculate_prediction_metrics, prepare_export_data

from .overview import display_statistical_summary
from .viewer import display_image_grid


def display_advanced_analysis(df: pd.DataFrame, image_dir: str | Path) -> None:
    """Render advanced analysis tools including sorting, filtering, and exports."""
    st.subheader("🔬 Advanced Analysis Tools")

    metrics_df = calculate_prediction_metrics(df)

    st.markdown("### Performance Sorting & Filtering")
    col1, col2, col3 = st.columns(3)

    with col1:
        sort_by = st.selectbox(
            "Sort by",
            [
                "prediction_count",
                "total_area",
                "avg_confidence",
                "aspect_ratio",
            ],
            help="Sort images by different calculated metrics.",
        )
    with col2:
        sort_order = st.selectbox("Order", ["descending", "ascending"])
    with col3:
        filter_metric = st.selectbox(
            "Filter by",
            [
                "all",
                "high_confidence",
                "low_confidence",
                "many_predictions",
                "few_predictions",
                "custom_threshold",
            ],
            help="Filter images by predefined categories or custom threshold.",
        )

    # Custom threshold filtering
    filtered_df = None
    if filter_metric == "custom_threshold":
        st.markdown("#### 🎯 Custom Threshold Filtering")
        threshold_col1, threshold_col2, threshold_col3 = st.columns(3)

        with threshold_col1:
            threshold_metric = st.selectbox(
                "Metric",
                ["avg_confidence", "prediction_count", "total_area", "aspect_ratio"],
                help="Select which metric to filter by",
            )

        with threshold_col2:
            threshold_operator = st.selectbox(
                "Operator",
                ["<", "<=", ">", ">=", "=="],
                help="Comparison operator",
            )

        with threshold_col3:
            # Dynamic default based on metric
            if threshold_metric == "avg_confidence":
                default_threshold = 0.30
                step = 0.01
                format_str = "%.2f"
            elif threshold_metric == "prediction_count":
                default_threshold = 10
                step = 1
                format_str = "%d"
            elif threshold_metric == "total_area":
                default_threshold = 1000
                step = 100
                format_str = "%d"
            else:  # aspect_ratio
                default_threshold = 1.0
                step = 0.1
                format_str = "%.2f"

            threshold_value = st.number_input(
                "Threshold",
                min_value=0.0,
                value=float(default_threshold),
                step=float(step),
                format=format_str,
                help=f"Filter images where {threshold_metric} {threshold_operator} threshold",
            )

        # Apply custom filter
        if threshold_operator == "<":
            filtered_df = metrics_df[metrics_df[threshold_metric] < threshold_value]
        elif threshold_operator == "<=":
            filtered_df = metrics_df[metrics_df[threshold_metric] <= threshold_value]
        elif threshold_operator == ">":
            filtered_df = metrics_df[metrics_df[threshold_metric] > threshold_value]
        elif threshold_operator == ">=":
            filtered_df = metrics_df[metrics_df[threshold_metric] >= threshold_value]
        elif threshold_operator == "==":
            filtered_df = metrics_df[metrics_df[threshold_metric] == threshold_value]
        else:
            filtered_df = metrics_df

        num_filtered = len(filtered_df) if filtered_df is not None else 0
        st.info(f"✅ Found **{num_filtered}** images where `{threshold_metric}` {threshold_operator} `{threshold_value}`")

        # Add download options for filtered results
        if filtered_df is not None and len(filtered_df) > 0:
            st.markdown("#### 📥 Download Options")
            download_col1, download_col2 = st.columns(2)

            with download_col1:
                # Simple list download (filename + metric)
                st.download_button(
                    label=f"� Download List ({len(filtered_df)} images)",
                    data=filtered_df[["filename", threshold_metric]].to_csv(index=False),
                    file_name=f"filtered_{threshold_metric}_{threshold_operator.replace('<', 'lt').replace('>', 'gt').replace('=', 'eq')}_{threshold_value}.csv",
                    mime="text/csv",
                    help="Download simple list with filename and metric value",
                )

            with download_col2:
                # Full data download (with polygons)
                # Check if 'polygons' column exists in the original dataframe
                if "polygons" in df.columns:
                    # filtered_df already contains polygons from the original dataframe
                    st.download_button(
                        label=f"🔷 Download with Polygons ({len(filtered_df)} images)",
                        data=filtered_df.to_csv(index=False),
                        file_name=f"filtered_with_polygons_{threshold_metric}_{threshold_operator.replace('<', 'lt').replace('>', 'gt').replace('=', 'eq')}_{threshold_value}.csv",
                        mime="text/csv",
                        help="Download full data including polygon coordinates for analysis",
                    )
                else:
                    st.warning("⚠️ Polygon data not available in the predictions file")

    # Apply sorting and filtering
    if filtered_df is not None and filter_metric == "custom_threshold":
        sorted_df = filtered_df.sort_values(by=sort_by, ascending=(sort_order == "ascending"))
    else:
        sorted_df = apply_sorting_filtering(metrics_df, sort_by, sort_order, filter_metric)

    if "advanced_page" not in st.session_state:
        st.session_state.advanced_page = 0

    images_per_page = st.slider("Images per page", 5, 25, 10, key="advanced_images_per_page")
    total_images = len(sorted_df)
    total_pages = (total_images + images_per_page - 1) // images_per_page if total_images else 1

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button(
            "⬅️ Previous",
            key="advanced_prev",
            disabled=st.session_state.advanced_page == 0,
        ):
            st.session_state.advanced_page -= 1

    with col2:
        current_page = st.session_state.advanced_page
        current_page = max(0, min(current_page, total_pages - 1))
        st.session_state.advanced_page = current_page
        start_idx = current_page * images_per_page
        end_idx = min(start_idx + images_per_page, total_images)
        if total_images:
            st.markdown(f"**Showing {start_idx + 1}-{end_idx} of {total_images} images**")
        else:
            st.markdown("**No images available for the selected filters.**")

    with col3:
        if st.button(
            "Next ➡️",
            key="advanced_next",
            disabled=st.session_state.advanced_page >= total_pages - 1,
        ):
            st.session_state.advanced_page += 1

    if total_images:
        display_image_grid(sorted_df, str(image_dir), sort_by, images_per_page, start_idx)

    st.markdown("---")
    display_statistical_summary(metrics_df)

    st.markdown("---")
    export_results(metrics_df, sorted_df)


def export_results(original_df: pd.DataFrame, sorted_df: pd.DataFrame) -> None:
    """Provide download buttons for exporting analysis results."""
    st.markdown("### Export Results")
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📥 Download Sorted Results (CSV)",
            data=sorted_df.to_csv(index=False),
            file_name="sorted_predictions.csv",
            mime="text/csv",
        )

    with col2:
        df_export, summary_stats = prepare_export_data(original_df)
        summary_csv = pd.DataFrame([summary_stats]).to_csv(index=False)
        st.download_button(
            label="📊 Download Summary Stats (CSV)",
            data=summary_csv,
            file_name="summary_statistics.csv",
            mime="text/csv",
        )
