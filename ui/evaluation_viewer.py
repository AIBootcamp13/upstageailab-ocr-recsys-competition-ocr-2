# ui/evaluation_viewer.py
"""
OCR Evaluation Results Viewer

A Streamlit application for analyzing OCR model predictions with advanced
visualization and comparison capabilities.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# --- Project Setup ---
# Resolve and add the project root to the system path for module imports.
# This ensures that modules like `data_utils` and `visualization` can be found.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Modular Imports ---
# Import the modular components for data processing and visualization
try:
    from ui.data_utils import (
        apply_sorting_filtering,
        calculate_image_differences,
        calculate_prediction_metrics,
        load_predictions_file,
        prepare_export_data,
    )
    from ui.visualization import (
        display_dataset_overview,
        display_image_grid,
        display_image_viewer,
        display_image_with_predictions,
        display_model_comparison_stats,
        display_prediction_analysis,
        display_statistical_summary,
        display_visual_comparison,
    )
except (ImportError, ModuleNotFoundError):
    st.error("Could not import `data_utils` or `visualization` modules. Please ensure they are in the correct path.")

    # Create dummy functions to allow the app to run without crashing
    def dummy_func(*args, **kwargs):
        st.warning("A function was called but its module could not be loaded.")
        if "df" in kwargs:
            return kwargs["df"]
        if args and isinstance(args[0], pd.DataFrame):
            return args[0]
        return pd.DataFrame()

    load_predictions_file = calculate_prediction_metrics = apply_sorting_filtering = calculate_image_differences = prepare_export_data = (
        dummy_func
    )
    display_dataset_overview = display_prediction_analysis = display_image_viewer = display_image_grid = display_statistical_summary = (
        display_model_comparison_stats
    ) = display_visual_comparison = dummy_func


# --- Main Application ---
def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="OCR Evaluation Viewer", page_icon="📊", layout="wide")

    st.title("📊 OCR Evaluation Results Viewer")
    st.markdown("Analyze predictions, view metrics, and compare model performance.")

    # Main navigation tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Run Analysis", "⚖️ Model Comparison", "🖼️ Image Gallery"])

    with tab1:
        render_single_run_analysis()

    with tab2:
        render_comparison_view()

    with tab3:
        render_image_gallery()


# --- Single Run Analysis View ---
def render_single_run_analysis():
    """Renders the UI for analyzing a single model's prediction file."""
    st.header("🔍 Single Run Analysis")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Load Evaluation Data")
        predictions_file = None

        # Option 1: File Uploader
        uploaded_file = st.file_uploader(
            "Upload Predictions CSV",
            type=["csv"],
            help="Upload the submission.csv file with predictions",
        )

        # Option 2: Select from the project's 'outputs' directory
        st.markdown("**Or select from outputs:**")

        # Clarify file selection behavior to the user
        if uploaded_file is not None:
            st.info("📤 You have uploaded a file. The uploaded file will override any file selected below.")
        else:
            st.info("📁 You can either upload a file above or select one from the list below.")

        # Disable selectbox if a file is uploaded
        selectbox_disabled = uploaded_file is not None

        outputs_dir = project_root / "outputs"
        if outputs_dir.exists():
            if prediction_files := sorted(
                list(outputs_dir.rglob("**/predictions/submission.csv")),
                reverse=True,
            ):
                file_options = {str(f.relative_to(outputs_dir)): f for f in prediction_files}
                selected_option = st.selectbox(
                    "Select prediction file",
                    [""] + list(file_options.keys()),
                    disabled=selectbox_disabled,
                    help="Select a predictions file from the list. Disabled if a file is uploaded.",
                )
                if selected_option and not selectbox_disabled:
                    predictions_file = file_options[selected_option]

        if uploaded_file:
            predictions_file = uploaded_file

        image_dir_path = st.text_input(
            "Image Directory Path",
            value=str(project_root / "data/datasets/images/test"),
            help="Path to the directory containing the test images.",
        )
        image_dir = Path(image_dir_path)

    with col2:
        if predictions_file is not None:
            try:
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

                st.success(f"Successfully loaded `{getattr(predictions_file, 'name', predictions_file)}` with {len(df)} rows.")

                # Analysis tabs for the single run
                analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs(
                    [
                        "📊 Overview",
                        "🔍 Detailed Analysis",
                        "🖼️ Image Viewer",
                        "🔬 Advanced Tools",
                    ]
                )

                with analysis_tab1:
                    display_dataset_overview(df)
                with analysis_tab2:
                    display_prediction_analysis(df)
                with analysis_tab3:
                    if image_dir.is_dir():
                        display_image_viewer(df, str(image_dir))
                    else:
                        st.warning("The specified image directory does not exist. Please provide a valid path.")
                with analysis_tab4:
                    if image_dir.is_dir():
                        # Check if directory contains image files
                        image_files = [
                            f
                            for f in image_dir.iterdir()
                            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"} and f.is_file()
                        ]
                        if image_files:
                            display_advanced_analysis(df, image_dir)
                        else:
                            st.warning(
                                "The specified image directory does not contain any image files. " "Advanced analysis requires images."
                            )
                    else:
                        st.warning("The specified image directory does not exist. Advanced analysis requires images.")

            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            st.info("Please upload a predictions file or select one from the outputs directory to begin analysis.")


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


# --- Model Comparison View ---
def render_comparison_view():
    """Renders the UI for comparing two model prediction files."""
    st.header("⚖️ Model Comparison")

    model_a_file = None
    model_b_file = None

    outputs_dir = project_root / "outputs"
    prediction_files = []
    file_options = {}

    if outputs_dir.exists():
        prediction_files = sorted(list(outputs_dir.rglob("**/predictions/submission.csv")), reverse=True)
        if prediction_files:
            file_options = {str(f.relative_to(outputs_dir)): f for f in prediction_files}

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model A")
        uploaded_a = st.file_uploader("Upload Model A Predictions", type=["csv"], key="model_a_upload")
        if not uploaded_a and file_options:
            selected_a = st.selectbox(
                "Or select Model A file",
                [""] + list(file_options.keys()),
                key="model_a_select",
            )
            if selected_a:
                model_a_file = file_options[selected_a]
        elif uploaded_a:
            model_a_file = uploaded_a

    with col2:
        st.subheader("Model B")
        uploaded_b = st.file_uploader("Upload Model B Predictions", type=["csv"], key="model_b_upload")
        if not uploaded_b and file_options:
            selected_b = st.selectbox(
                "Or select Model B file",
                [""] + list(file_options.keys()),
                key="model_b_select",
            )
            if selected_b:
                model_b_file = file_options[selected_b]
        elif uploaded_b:
            model_b_file = uploaded_b

    image_dir_path = st.text_input(
        "Image Directory Path",
        value=str(project_root / "data/datasets/images/test"),
        help="Path to the directory containing test images for visual comparison.",
        key="comparison_image_dir",
    )
    image_dir = Path(image_dir_path)

    def validate_predictions_file(file_obj):
        """Validate a predictions file before processing."""
        try:
            # Handle both uploaded files and file paths
            if hasattr(file_obj, "read"):  # Uploaded file object
                df = pd.read_csv(file_obj)
                file_name = getattr(file_obj, "name", "uploaded file")
            else:  # File path
                df = pd.read_csv(file_obj)
                file_name = str(file_obj)

            # Validate the loaded DataFrame
            if df.empty:
                st.error(f"The predictions file '{file_name}' is empty. Please check your file and try again.")
                return None

            # Check for required columns
            required_columns = ["filename", "polygons"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(
                    f"The predictions file '{file_name}' is missing required columns: "
                    f"{', '.join(missing_columns)}. Expected columns: {', '.join(required_columns)}"
                )
                return None

            return df

        except Exception as e:
            file_name = getattr(file_obj, "name", str(file_obj))
            st.error(f"Error reading predictions file '{file_name}': {e}")
            return None

    if model_a_file and model_b_file:
        df_a = validate_predictions_file(model_a_file)
        df_b = validate_predictions_file(model_b_file)

        if df_a is not None and df_b is not None:
            st.success("Successfully loaded both prediction files.")

            comp_tab1, comp_tab2, comp_tab3 = st.tabs(["📊 Statistics", "🔍 Differences", "🖼️ Visual Comparison"])

            with comp_tab1:
                display_model_comparison_stats(df_a, df_b)
            with comp_tab2:
                display_model_differences(df_a, df_b)
            with comp_tab3:
                if image_dir.is_dir():
                    display_visual_comparison(df_a, df_b, str(image_dir))
                else:
                    st.warning("The specified image directory does not exist. Please provide a valid path.")

        else:
            st.error("Failed to load one or both prediction files. Please check the files and try again.")
    else:
        st.info("Please provide two prediction files to compare.")


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


# --- Image Gallery View ---
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
                "low_confidence": "Low Confidence (≤0.8)",
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


# --- Script Entry Point ---
if __name__ == "__main__":
    main()
