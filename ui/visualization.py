"""
Visualization components for OCR evaluation viewer.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw

# Import data utilities
try:
    from .data_utils import (
        apply_sorting_filtering,
        calculate_image_differences,
        calculate_prediction_metrics,
        prepare_export_data,
    )
except ImportError:
    # Fallback for direct execution
    from data_utils import (
        apply_sorting_filtering,
        calculate_image_differences,
        calculate_prediction_metrics,
        prepare_export_data,
    )


def display_dataset_overview(df: pd.DataFrame):
    """Display basic dataset overview with key metrics."""
    st.subheader("📊 Dataset Overview")

    # Calculate basic statistics
    total_images = len(df)
    total_polygons = sum(
        (len(row["polygons"].split("|")) if pd.notna(row["polygons"]) and row["polygons"].strip() else 0) for _, row in df.iterrows()
    )
    avg_polygons = total_polygons / total_images if total_images > 0 else 0
    empty_predictions = sum(1 for _, row in df.iterrows() if pd.isna(row["polygons"]) or not row["polygons"].strip())

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Images", total_images)

    with col2:
        st.metric("Total Predictions", total_polygons)

    with col3:
        st.metric("Avg Predictions/Image", f"{avg_polygons:.1f}")

    with col4:
        st.metric("Empty Predictions", empty_predictions)

    # Statistics table
    st.markdown("### Prediction Statistics")
    stats_data = {
        "Metric": [
            "Total Polygons",
            "Average per Image",
            "Images with Predictions",
            "Empty Predictions",
        ],
        "Value": [
            str(total_polygons),
            f"{avg_polygons:.1f}",
            str(total_images - empty_predictions),
            str(empty_predictions),
        ],
    }
    st.table(pd.DataFrame(stats_data))


def display_prediction_analysis(df: pd.DataFrame):
    """Display detailed prediction analysis."""
    st.subheader("🎯 Prediction Analysis")

    # Calculate areas for analysis
    areas = []
    aspect_ratios = []

    for _, row in df.iterrows():
        polygons_str = str(row["polygons"]) if pd.notna(row["polygons"]) else ""
        if polygons_str.strip():
            polygons = polygons_str.split("|")
            for polygon in polygons:
                coords = [float(x) for x in polygon.split()]
                if len(coords) >= 8:
                    x_coords = coords[::2]
                    y_coords = coords[1::2]
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)
                    if width > 0 and height > 0:
                        areas.append(width * height)
                        aspect_ratios.append(width / height)

    if areas:
        # Display statistics
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


def display_image_viewer(df: pd.DataFrame, image_dir: str):
    """Display image viewer with predictions and pagination."""
    st.subheader("🖼️ Image Viewer")

    # Initialize session state for pagination
    if "image_viewer_page" not in st.session_state:
        st.session_state.image_viewer_page = 0

    # Pagination controls
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
        current_idx = st.session_state.image_viewer_page
        current_row = df.iloc[current_idx]
        st.markdown(f"**Image {current_idx + 1} of {total_images}**")
        st.markdown(f"**{current_row['filename']}**")

        # Display confidence and other metrics
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

    # Display the selected image
    selected_image = current_row["filename"]
    display_image_with_predictions(df, selected_image, image_dir)


def display_image_with_predictions(df: pd.DataFrame, image_name: str, image_dir: str):
    """Display a single image with its predictions."""
    image_path = Path(image_dir) / image_name

    if not image_path.exists():
        st.error(f"Image not found: {image_path}")
        return

    try:
        image = Image.open(image_path)

        # Get predictions for this image
        row = df[df["filename"] == image_name].iloc[0]
        polygons_str = str(row["polygons"]) if pd.notna(row["polygons"]) else ""

        if polygons_str.strip():
            # Draw predictions on image
            draw = ImageDraw.Draw(image, "RGBA")
            polygons = polygons_str.split("|")
            valid_polygons = 0

            for i, polygon in enumerate(polygons):
                try:
                    # Parse coordinates with validation
                    coords = []
                    for x in polygon.split():
                        try:
                            coords.append(float(x))
                        except ValueError:
                            # Skip invalid coordinate values
                            break

                    if len(coords) >= 8 and len(coords) % 2 == 0:
                        # Convert to (x,y) tuples
                        points = [(coords[j], coords[j + 1]) for j in range(0, len(coords), 2)]

                        # Draw polygon with compatibility check for width parameter
                        try:
                            draw.polygon(
                                points,
                                outline=(255, 0, 0, 255),
                                fill=(255, 0, 0, 50),
                                width=2,
                            )
                        except TypeError:
                            # Fallback for older Pillow versions that don't support width parameter
                            draw.polygon(points, outline=(255, 0, 0, 255), fill=(255, 0, 0, 50))

                        # Draw label
                        if points:
                            draw.text(
                                (points[0][0], points[0][1] - 10),
                                f"T{i + 1}",
                                fill=(255, 0, 0, 255),
                            )

                        valid_polygons += 1

                except Exception:
                    # Skip malformed polygons and continue with others
                    continue

            st.image(image, caption=f"Predictions on {image_name}")
            st.info(f"Found {valid_polygons} valid text regions in this image (out of {len(polygons)} total)")

            # Click to enlarge functionality
            if st.button("🔍 Click to Enlarge", key=f"enlarge_{image_name}"):
                st.session_state[f"enlarge_{image_name}"] = not st.session_state.get(f"enlarge_{image_name}", False)

            if st.session_state.get(f"enlarge_{image_name}", False):
                st.markdown("### 🔍 Enlarged View")
                col1, col2 = st.columns([4, 1])

                with col1:
                    # Display larger image
                    st.image(
                        image,
                        caption=f"Enlarged: {image_name}",
                        use_container_width=True,
                    )

                with col2:
                    st.markdown("**Image Details:**")
                    st.write(f"**Filename:** {image_name}")
                    st.write(f"**Valid Polygons:** {valid_polygons}")
                    st.write(f"**Total Polygons:** {len(polygons)}")

                    # Show confidence score
                    row = df[df["filename"] == image_name].iloc[0]
                    confidence = row.get("avg_confidence", 0.8)
                    st.write(f"**Confidence:** {confidence:.3f}")

                    if st.button("❌ Close Enlarged View", key=f"close_{image_name}"):
                        st.session_state[f"enlarge_{image_name}"] = False

        else:
            st.image(image, caption=image_name)
            st.info("No predictions found for this image")

            # Click to enlarge for images without predictions too
            if st.button("🔍 Click to Enlarge", key=f"enlarge_{image_name}"):
                st.session_state[f"enlarge_{image_name}"] = not st.session_state.get(f"enlarge_{image_name}", False)

            if st.session_state.get(f"enlarge_{image_name}", False):
                st.markdown("### 🔍 Enlarged View")
                col1, col2 = st.columns([4, 1])

                with col1:
                    st.image(
                        image,
                        caption=f"Enlarged: {image_name}",
                        use_container_width=True,
                    )

                with col2:
                    st.markdown("**Image Details:**")
                    st.write(f"**Filename:** {image_name}")
                    st.write("**No predictions found**")

                    if st.button("❌ Close Enlarged View", key=f"close_{image_name}"):
                        st.session_state[f"enlarge_{image_name}"] = False

    except Exception as e:
        st.error(f"Error loading image: {str(e)}")


def display_image_grid(
    df: pd.DataFrame,
    image_dir: str,
    sort_metric: str,
    max_images: int = 10,
    start_idx: int = 0,
):
    """Display a grid of images with their metrics."""
    end_idx = min(start_idx + max_images, len(df))

    st.markdown(f"### Images {start_idx + 1}-{end_idx} of {len(df)} ({sort_metric})")

    images_to_show = df.iloc[start_idx:end_idx]
    cols = st.columns(5)  # 5 images per row

    for i, (_, row) in enumerate(images_to_show.iterrows()):
        col_idx = i % 5
        with cols[col_idx]:
            image_path = Path(image_dir) / row["filename"]
            if image_path.exists():
                try:
                    image = Image.open(image_path)
                    # Resize for grid display
                    image.thumbnail((200, 200))
                    st.image(image, caption=f"{row['filename'][:15]}...")

                    # Show metric value
                    if sort_metric in row and sort_metric != "filename":
                        try:
                            value = row[sort_metric]
                            if isinstance(value, int | float):
                                st.metric(sort_metric, f"{value:.2f}")
                            else:
                                st.metric(sort_metric, str(value))
                        except Exception:
                            st.metric(sort_metric, str(row[sort_metric]))
                except Exception as e:
                    st.error(f"Error loading {row['filename']}: {str(e)}")
            else:
                st.error(f"Image not found: {row['filename']}")


def display_statistical_summary(df: pd.DataFrame):
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


def display_statistical_analysis(df: pd.DataFrame):
    """Display comprehensive statistical analysis with visualizations."""
    st.subheader("📈 Statistical Analysis")

    # Distribution plots
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Prediction Count Distribution")
        fig = px.histogram(
            df,
            x="prediction_count",
            nbins=20,
            title="Distribution of Predictions per Image",
        )
        st.plotly_chart(
            fig,
        )

    with col2:
        st.markdown("#### Total Area Distribution")
        fig = px.histogram(df, x="total_area", nbins=20, title="Distribution of Total Prediction Area")
        st.plotly_chart(
            fig,
        )

    # Scatter plot: prediction count vs area
    st.markdown("#### Prediction Count vs Total Area")
    fig = px.scatter(
        df,
        x="prediction_count",
        y="total_area",
        title="Relationship between Prediction Count and Total Area",
        labels={
            "prediction_count": "Number of Predictions",
            "total_area": "Total Area",
        },
    )
    st.plotly_chart(
        fig,
    )

    # Summary statistics table
    st.markdown("#### Detailed Statistics")
    stats_df = pd.DataFrame(
        {
            "Metric": ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
            "Prediction Count": df["prediction_count"].describe(),
            "Total Area": df["total_area"].describe(),
        }
    )
    st.table(stats_df)

    # Outlier analysis
    st.markdown("#### Outlier Analysis")
    col1, col2 = st.columns(2)

    with col1:
        # Images with most predictions
        top_pred = df.nlargest(5, "prediction_count")[["filename", "prediction_count"]]
        st.markdown("**Images with Most Predictions:**")
        st.table(top_pred)

    with col2:
        # Images with largest area
        top_area = df.nlargest(5, "total_area")[["filename", "total_area"]]
        st.markdown("**Images with Largest Prediction Area:**")
        st.table(top_area)


def display_model_comparison_stats(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """Display statistical comparison between two models."""
    from .data_utils import calculate_model_metrics

    st.subheader("📊 Model Comparison Statistics")

    # Calculate metrics for both models
    metrics_a = calculate_model_metrics(df_a)
    metrics_b = calculate_model_metrics(df_b)

    # Display comparison table
    comparison_data = {
        "Metric": [
            "Total Predictions",
            "Avg Predictions/Image",
            "Images with Predictions",
            "Empty Predictions",
        ],
        "Model A": [
            str(metrics_a["total_predictions"]),
            f"{metrics_a['avg_predictions']:.1f}",
            str(metrics_a["images_with_predictions"]),
            str(metrics_a["empty_predictions"]),
        ],
        "Model B": [
            str(metrics_b["total_predictions"]),
            f"{metrics_b['avg_predictions']:.1f}",
            str(metrics_b["images_with_predictions"]),
            str(metrics_b["empty_predictions"]),
        ],
        "Difference": [
            str(metrics_b["total_predictions"] - metrics_a["total_predictions"]),
            f"{metrics_b['avg_predictions'] - metrics_a['avg_predictions']:.1f}",
            str(metrics_b["images_with_predictions"] - metrics_a["images_with_predictions"]),
            str(metrics_b["empty_predictions"] - metrics_a["empty_predictions"]),
        ],
    }

    st.table(pd.DataFrame(comparison_data))

    # Performance indicators
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta = metrics_b["total_predictions"] - metrics_a["total_predictions"]
        st.metric("Total Predictions", metrics_b["total_predictions"], delta=delta)

    with col2:
        delta = metrics_b["avg_predictions"] - metrics_a["avg_predictions"]
        st.metric("Avg per Image", f"{metrics_b['avg_predictions']:.1f}", delta=f"{delta:.1f}")

    with col3:
        delta = metrics_b["images_with_predictions"] - metrics_a["images_with_predictions"]
        st.metric("Images w/ Preds", metrics_b["images_with_predictions"], delta=delta)

    with col4:
        delta = metrics_a["empty_predictions"] - metrics_b["empty_predictions"]  # Note: inverted for positive meaning
        st.metric("Empty Preds", metrics_b["empty_predictions"], delta=-delta)


def display_visual_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame, image_dir: str):
    """Display visual comparison of predictions on the same images."""
    from .data_utils import find_common_images

    st.subheader("🖼️ Visual Comparison")

    # Find common images
    common_images = find_common_images(df_a, df_b)

    if not common_images:
        st.warning("No common images found between the two models.")
        return

    # Pagination controls
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

    # Display the comparison
    display_side_by_side_comparison(df_a, df_b, current_image, image_dir)


def display_side_by_side_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame, image_name: str, image_dir: str):
    """Display side-by-side comparison of two models on the same image."""
    image_path = Path(image_dir) / image_name

    if not image_path.exists():
        st.error(f"Image not found: {image_path}")
        return

    try:
        # Load original image
        original_image = Image.open(image_path)

        # Get predictions for both models
        row_a = df_a[df_a["filename"] == image_name].iloc[0]
        row_b = df_b[df_b["filename"] == image_name].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Original Image**")
            st.image(
                original_image,
                caption="Original",
            )

        with col2:
            st.markdown("**Model A Predictions**")
            img_a = draw_predictions_on_image(
                original_image.copy(),
                str(row_a["polygons"]) if pd.notna(row_a["polygons"]) else "",
                (255, 0, 0),
            )
            pred_count_a = len(str(row_a["polygons"]).split("|")) if pd.notna(row_a["polygons"]) else 0
            conf_a = row_a.get("avg_confidence", 0.8)
            st.image(
                img_a,
                caption=f"Model A ({pred_count_a} predictions, conf: {conf_a:.2f})",
            )

        with col3:
            st.markdown("**Model B Predictions**")
            img_b = draw_predictions_on_image(
                original_image.copy(),
                str(row_b["polygons"]) if pd.notna(row_b["polygons"]) else "",
                (0, 255, 0),
            )
            pred_count_b = len(str(row_b["polygons"]).split("|")) if pd.notna(row_b["polygons"]) else 0
            conf_b = row_b.get("avg_confidence", 0.8)
            st.image(
                img_b,
                caption=f"Model B ({pred_count_b} predictions, conf: {conf_b:.2f})",
            )

        # Show difference metrics
        st.markdown("### Comparison Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Prediction Count", pred_count_b, delta=pred_count_b - pred_count_a)

        with col2:
            from .data_utils import calculate_total_area

            area_a = calculate_total_area(str(row_a["polygons"]) if pd.notna(row_a["polygons"]) else "")
            area_b = calculate_total_area(str(row_b["polygons"]) if pd.notna(row_b["polygons"]) else "")
            st.metric("Total Area", f"{area_b:.0f}", delta=f"{area_b - area_a:.0f}")

        with col3:
            st.metric("Confidence", f"{conf_b:.2f}", delta=f"{conf_b - conf_a:.2f}")

    except Exception as e:
        st.error(f"Error creating comparison: {str(e)}")


def draw_predictions_on_image(image: Image.Image, polygons_str: str, color: tuple[int, int, int]) -> Image.Image:
    """Draw predictions on an image with specified color."""
    if not polygons_str.strip():
        return image

    draw = ImageDraw.Draw(image, "RGBA")
    polygons = polygons_str.split("|")

    for i, polygon in enumerate(polygons):
        coords = [float(x) for x in polygon.split()]
        if len(coords) >= 8:
            points = [(coords[j], coords[j + 1]) for j in range(0, len(coords), 2)]
            draw.polygon(points, outline=color + (255,), fill=color + (50,), width=2)

    return image


def display_advanced_analysis(df: pd.DataFrame, image_dir: Path):
    """Renders advanced analysis features like sorting, filtering, and exporting."""
    st.subheader("🔬 Advanced Analysis Tools")

    # Initialize session state for pagination
    if "advanced_page" not in st.session_state:
        st.session_state.advanced_page = 0

    try:
        df_metrics = calculate_prediction_metrics(df)
    except Exception as e:
        st.error(f"Error calculating prediction metrics: {e}")
        return

    st.markdown("### Performance Sorting & Filtering")
    col1, col2, col3 = st.columns(3)
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            ["prediction_count", "total_area", "avg_confidence", "aspect_ratio"],
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
            ],
        )

    sorted_df = apply_sorting_filtering(df_metrics, sort_by, sort_order, filter_metric)

    # Pagination controls
    images_per_page = st.slider("Images per page", 5, 25, 10, key="advanced_images_per_page")
    total_images = len(sorted_df)
    total_pages = (total_images + images_per_page - 1) // images_per_page

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
        start_idx = current_page * images_per_page
        end_idx = min(start_idx + images_per_page, total_images)
        st.markdown(f"**Showing {start_idx + 1}-{end_idx} of {total_images} images**")

    with col3:
        if st.button("Next ➡️", key="advanced_next", disabled=current_page >= total_pages - 1):
            st.session_state.advanced_page += 1

    # Display paginated image grid
    display_image_grid(sorted_df, str(image_dir), sort_by, images_per_page, start_idx)

    st.markdown("---")
    display_statistical_summary(df_metrics)

    st.markdown("---")
    export_results(df_metrics, sorted_df)


def export_results(original_df: pd.DataFrame, sorted_df: pd.DataFrame):
    """Provides download buttons for exporting analysis results."""
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


def display_model_differences(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """Calculates and displays the differences between two models' predictions."""
    st.subheader("Prediction Differences")

    # Initialize session state for pagination
    if "pred_page" not in st.session_state:
        st.session_state.pred_page = 0
    if "area_page" not in st.session_state:
        st.session_state.area_page = 0
    if "conf_page" not in st.session_state:
        st.session_state.conf_page = 0

    try:
        diff_df = calculate_image_differences(df_a, df_b)
    except Exception as e:
        st.error(f"Error calculating prediction differences: {e}")
        return

    if diff_df.empty:
        st.warning("No common images found to compare between the two models.")
        return

    # Pagination settings
    items_per_page = 20

    # Prediction Count Differences
    st.markdown("#### Top Differences in Prediction Count")
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        pred_order = st.radio("Order", ["Largest", "Smallest"], key="pred_order", horizontal=True)

    with col2:
        if st.button("⬅️ Previous", key="pred_prev", disabled=st.session_state.pred_page == 0):
            st.session_state.pred_page -= 1

    with col3:
        total_pred_pages = (len(diff_df) + items_per_page - 1) // items_per_page
        if st.button(
            "Next ➡️",
            key="pred_next",
            disabled=st.session_state.pred_page >= total_pred_pages - 1,
        ):
            st.session_state.pred_page += 1

    # Display prediction count differences
    ascending = pred_order == "Smallest"
    sorted_pred = diff_df.sort_values("abs_pred_diff", ascending=ascending)
    start_idx = st.session_state.pred_page * items_per_page
    end_idx = min(start_idx + items_per_page, len(sorted_pred))

    st.dataframe(sorted_pred.iloc[start_idx:end_idx][["filename", "pred_a", "pred_b", "pred_diff"]])

    # Prediction Area Differences
    st.markdown("#### Top Differences in Total Prediction Area")
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        area_order = st.radio("Order", ["Largest", "Smallest"], key="area_order", horizontal=True)

    with col2:
        if st.button("⬅️ Previous", key="area_prev", disabled=st.session_state.area_page == 0):
            st.session_state.area_page -= 1

    with col3:
        total_area_pages = (len(diff_df) + items_per_page - 1) // items_per_page
        if st.button(
            "Next ➡️",
            key="area_next",
            disabled=st.session_state.area_page >= total_area_pages - 1,
        ):
            st.session_state.area_page += 1

    # Display area differences
    ascending = area_order == "Smallest"
    sorted_area = diff_df.sort_values("abs_area_diff", ascending=ascending)
    start_idx = st.session_state.area_page * items_per_page
    end_idx = min(start_idx + items_per_page, len(sorted_area))

    st.dataframe(sorted_area.iloc[start_idx:end_idx][["filename", "area_a", "area_b", "area_diff"]])

    # Prediction Confidence Differences
    st.markdown("#### Top Differences in Prediction Confidence")
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        conf_order = st.radio("Order", ["Largest", "Smallest"], key="conf_order", horizontal=True)

    with col2:
        if st.button("⬅️ Previous", key="conf_prev", disabled=st.session_state.conf_page == 0):
            st.session_state.conf_page -= 1

    with col3:
        total_conf_pages = (len(diff_df) + items_per_page - 1) // items_per_page
        if st.button(
            "Next ➡️",
            key="conf_next",
            disabled=st.session_state.conf_page >= total_conf_pages - 1,
        ):
            st.session_state.conf_page += 1

    # Display confidence differences
    ascending = conf_order == "Smallest"
    sorted_conf = diff_df.sort_values("abs_conf_diff", ascending=ascending)
    start_idx = st.session_state.conf_page * items_per_page
    end_idx = min(start_idx + items_per_page, len(sorted_conf))

    st.dataframe(sorted_conf.iloc[start_idx:end_idx][["filename", "conf_a", "conf_b", "conf_diff"]])


def render_low_confidence_analysis(df: pd.DataFrame):
    """Render specialized analysis for low confidence predictions."""
    st.markdown("### ⚠️ Low Confidence Analysis")

    # Summary statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Low Confidence Images", len(df))

    with col2:
        avg_conf = df["avg_confidence"].mean()
        st.metric("Average Confidence", f"{avg_conf:.3f}")

    with col3:
        avg_preds = df["prediction_count"].mean()
        st.metric("Avg Predictions", f"{avg_preds:.1f}")

    # Confidence distribution
    st.markdown("#### Confidence Distribution")
    fig = px.histogram(df, x="avg_confidence", nbins=20, title="Low Confidence Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # Top issues
    st.markdown("#### Images with Lowest Confidence")
    lowest_conf = df.nsmallest(10, "avg_confidence")[["filename", "avg_confidence", "prediction_count"]]
    st.dataframe(lowest_conf)

    # Correlation analysis
    st.markdown("#### Correlation Analysis")
    corr_data = df[["avg_confidence", "prediction_count", "total_area"]].corr()
    fig = px.imshow(
        corr_data,
        text_auto=True,
        title="Correlation between Confidence and Other Metrics",
    )
    st.plotly_chart(fig, use_container_width=True)
