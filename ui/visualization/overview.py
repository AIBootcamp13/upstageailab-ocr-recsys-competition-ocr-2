from __future__ import annotations

"""Dataset overview and statistical summaries."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from .helpers import parse_polygon_string


def display_dataset_overview(df: pd.DataFrame) -> None:
    """Display basic dataset overview with key metrics."""
    st.subheader("📊 Dataset Overview")

    total_images = len(df)
    total_polygons = sum(len(parse_polygon_string(str(row.get("polygons", "")))) for _, row in df.iterrows())
    avg_polygons = total_polygons / total_images if total_images else 0
    empty_predictions = sum(not str(row.get("polygons", "")).strip() for _, row in df.iterrows())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", total_images)
    with col2:
        st.metric("Total Predictions", total_polygons)
    with col3:
        st.metric("Avg Predictions/Image", f"{avg_polygons:.1f}")
    with col4:
        st.metric("Empty Predictions", empty_predictions)

    st.markdown("### Prediction Statistics")
    # Calculate confidence statistics
    if "avg_confidence" in df.columns:
        min_conf = df["avg_confidence"].min()
        avg_conf = df["avg_confidence"].mean()
        max_conf = df["avg_confidence"].max()

        # Check if all confidence scores are 1.0 (indicating synthetic/default values)
        if avg_conf == 1.0 and min_conf == 1.0 and max_conf == 1.0:
            conf_stats = ["N/A (confidence scores not calculated)"]
        else:
            conf_stats = [
                f"Lowest Confidence: {min_conf:.3f}",
                f"Average Confidence: {avg_conf:.3f}",
                f"Highest Confidence: {max_conf:.3f}",
            ]
    else:
        conf_stats = ["Confidence data not available"]

    stats_data = {
        "Metric": [
            "Total Polygons",
            "Average per Image",
            "Images with Predictions",
            "Empty Predictions",
        ]
        + (["Confidence Scores"] if "avg_confidence" in df.columns else []),
        "Value": [
            str(total_polygons),
            f"{avg_polygons:.1f}",
            str(total_images - empty_predictions),
            str(empty_predictions),
        ]
        + ([" | ".join(conf_stats)] if "avg_confidence" in df.columns else []),
    }
    st.table(pd.DataFrame(stats_data))


def display_prediction_analysis(df: pd.DataFrame) -> None:
    """Display detailed prediction analysis."""
    st.subheader("🎯 Prediction Analysis")

    areas: list[float] = []
    aspect_ratios: list[float] = []

    for _, row in df.iterrows():
        polygons = parse_polygon_string(str(row.get("polygons", "")))
        for polygon in polygons:
            xs = polygon[::2]
            ys = polygon[1::2]
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            if width > 0 and height > 0:
                areas.append(width * height)
                aspect_ratios.append(width / height)

    if areas:
        stats_data = {
            "Metric": [
                "Count",
                "Mean Area",
                "Median Area",
                "Mean Aspect Ratio",
                "Median Aspect Ratio",
            ],
            "Value": [
                str(len(areas)),
                f"{np.mean(areas):.0f}",
                f"{np.median(areas):.0f}",
                f"{np.mean(aspect_ratios):.2f}",
                f"{np.median(aspect_ratios):.2f}",
            ],
        }
        st.table(pd.DataFrame(stats_data))
    else:
        st.info("No prediction data available for analysis.")


def display_statistical_summary(df: pd.DataFrame) -> None:
    """Display comprehensive statistical summary."""
    st.subheader("📈 Statistical Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Prediction Statistics")
        pred_counts = df["prediction_count"]
        st.metric("Total Predictions", pred_counts.sum())
        st.metric("Avg per Image", f"{pred_counts.mean():.1f}")
        st.metric("Max per Image", pred_counts.max())

    with col2:
        st.markdown("#### Area Statistics")
        areas = df["total_area"]
        st.metric("Total Area", f"{areas.sum():.0f}")
        st.metric("Avg Area", f"{areas.mean():.0f}")
        st.metric("Max Area", f"{areas.max():.0f}")


def display_statistical_analysis(df: pd.DataFrame) -> None:
    """Display comprehensive statistical analysis with visualizations."""
    st.subheader("📈 Statistical Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Prediction Count Distribution")
        _render_histogram(df, "prediction_count", "Distribution of Predictions per Image")

    with col2:
        st.markdown("#### Total Area Distribution")
        _render_histogram(df, "total_area", "Distribution of Total Prediction Area")

    st.markdown("#### Prediction Count vs Total Area")
    fig = px.scatter(
        df,
        x="prediction_count",
        y="total_area",
        title="Relationship between Prediction Count and Total Area",
        labels={"prediction_count": "Number of Predictions", "total_area": "Total Area"},
    )
    st.plotly_chart(fig)

    st.markdown("#### Detailed Statistics")
    stats_df = pd.DataFrame(
        {
            "Metric": ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
            "Prediction Count": df["prediction_count"].describe(),
            "Total Area": df["total_area"].describe(),
        }
    )
    st.table(stats_df)

    col1, col2 = st.columns(2)

    with col1:
        top_pred = df.nlargest(5, "prediction_count")[["filename", "prediction_count"]]
        st.markdown("**Images with Most Predictions:**")
        st.table(top_pred)

    with col2:
        top_area = df.nlargest(5, "total_area")[["filename", "total_area"]]
        st.markdown("**Images with Largest Prediction Area:**")
        st.table(top_area)


def render_low_confidence_analysis(df: pd.DataFrame) -> None:
    """Render specialized analysis for low confidence predictions."""
    st.markdown("### ⚠️ Low Confidence Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Low Confidence Images", len(df))
    with col2:
        st.metric("Average Confidence", f"{df['avg_confidence'].mean():.3f}")
    with col3:
        st.metric("Avg Predictions", f"{df['prediction_count'].mean():.1f}")

    st.markdown("#### Confidence Distribution")
    _render_histogram(
        df,
        "avg_confidence",
        "Low Confidence Distribution",
        width="stretch",
    )

    st.markdown("#### Images with Lowest Confidence")
    lowest_conf = df.nsmallest(10, "avg_confidence")[["filename", "avg_confidence", "prediction_count"]]
    st.dataframe(lowest_conf)

    st.markdown("#### Correlation Analysis")
    corr_data = df[["avg_confidence", "prediction_count", "total_area"]].corr()
    fig = px.imshow(corr_data, text_auto=True, title="Correlation between Confidence and Other Metrics")
    st.plotly_chart(fig, width="stretch")


def _render_histogram(df: pd.DataFrame, column: str, title: str, width: str | None = None) -> None:
    fig = px.histogram(df, x=column, nbins=20, title=title)
    st.plotly_chart(fig, width=width)
