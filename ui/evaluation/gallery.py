"""
Image gallery view for OCR Evaluation Viewer.
"""

import sys
from pathlib import Path

import streamlit as st

# --- Project Setup ---
# Resolve and add the project root to the system path for module imports.
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import utilities
from ui.data_utils import apply_sorting_filtering, calculate_prediction_metrics, load_predictions_file
from ui.visualization import display_image_grid, display_image_with_predictions, render_low_confidence_analysis


def render_image_gallery():
    """Render the image gallery with pagination and filtering."""
    st.header("🖼️ Image Gallery")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Load Gallery Data")

        # File selection
        predictions_file = st.file_uploader("Upload predictions CSV", type=["csv"], key="gallery_predictions")

        # Image directory selection
        image_dir = st.text_input(
            "Image directory path",
            value="data/datasets/images/test",
            help="Path to the directory containing the images",
        )

        if predictions_file is None:
            st.info("Please upload a predictions CSV file to view the gallery.")
            return

        # Validate inputs
        if not Path(image_dir).exists():
            st.error(f"Image directory not found: {image_dir}")
            return

    with col2:
        try:
            # Load and validate data
            df = load_predictions_file(predictions_file)

            # Validate the loaded DataFrame
            if df.empty:
                st.error("The predictions file is empty. Please check your file and try again.")
                return

            # Check for required columns
            required_columns = ["filename", "polygons"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(
                    f"The predictions file is missing required columns: {', '.join(missing_columns)}. "
                    f"Expected columns: {', '.join(required_columns)}"
                )
                return

            st.success(f"Successfully loaded predictions with {len(df)} rows.")

            # Ensure metrics are calculated
            df = calculate_prediction_metrics(df)

            # Sidebar controls
            st.sidebar.header("🎛️ Gallery Controls")

            # Sorting options
            sort_options = {
                "filename": "Filename (A-Z)",
                "prediction_count": "Prediction Count",
                "total_area": "Total Area",
                "avg_confidence": "Average Confidence",
            }
            sort_by = st.sidebar.selectbox(
                "Sort by",
                list(sort_options.keys()),
                format_func=lambda x: sort_options[x],
            )
            sort_order = st.sidebar.radio("Sort order", ["descending", "ascending"], index=0)

            # Filtering options
            filter_options = {
                "all": "All Images",
                "high_confidence": "High Confidence (>0.8)",
                "low_confidence": "Low Confidence (<0.5)",
                "many_predictions": "Many Predictions",
                "few_predictions": "Few Predictions",
            }
            filter_metric = st.sidebar.selectbox(
                "Filter",
                list(filter_options.keys()),
                format_func=lambda x: filter_options[x],
            )

            # Apply sorting and filtering
            filtered_df = apply_sorting_filtering(df, sort_by, sort_order, filter_metric)

            # Pagination
            images_per_page = st.sidebar.slider("Images per page", 10, 50, 20)
            total_images = len(filtered_df)
            total_pages = (total_images + images_per_page - 1) // images_per_page

            if total_pages > 1:
                page = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1)
            else:
                page = 1

            start_idx = (page - 1) * images_per_page
            end_idx = min(start_idx + images_per_page, total_images)

            # Display summary
            st.markdown(f"**Showing {start_idx + 1}-{end_idx} of {total_images} images**")

            if filter_metric != "all":
                st.markdown(f"**Filter:** {filter_options[filter_metric]}")
            if sort_by != "filename":
                st.markdown(f"**Sorted by:** {sort_options[sort_by]} ({sort_order})")

            # Low confidence analysis section
            if filter_metric == "low_confidence":
                render_low_confidence_analysis(filtered_df)

            # Image grid display
            current_page_df = filtered_df.iloc[start_idx:end_idx]
            display_image_grid(current_page_df, str(image_dir), sort_by, images_per_page)

            # Individual image viewer
            st.markdown("---")
            st.subheader("🔍 Individual Image Viewer")
            if selected_image := st.selectbox(
                "Select an image to view in detail",
                filtered_df["filename"].tolist(),
                key="gallery_image_select",
            ):
                display_image_with_predictions(filtered_df, selected_image, str(image_dir))

        except Exception as e:
            st.error(f"Error loading gallery data: {e}")
            return
