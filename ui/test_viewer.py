# ui/test_viewer.py
"""
OCR Evaluation Results Viewer

A Streamlit application for analyzing OCR model predictions with advanced
visualization and comparison capabilities.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


# Import our modular components
# --- Modular Imports ---
# Import the modular components for data processing and visualization
try:
    from ui.data_utils import apply_sorting_filtering, calculate_image_differences, calculate_prediction_metrics, prepare_export_data
    from ui.utils.config_parser import ConfigParser
    from ui.visualization import (
        display_dataset_overview,
        display_image_grid,
        display_image_viewer,
        display_model_comparison_stats,
        display_prediction_analysis,
        display_statistical_summary,
        display_visual_comparison,
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please ensure all required modules are available.")
    st.stop()


def render_single_run_analysis():
    """Render single run analysis interface."""
    st.header("🔍 Single Run Analysis")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Load Evaluation Data")

        # File selection
        predictions_file = st.file_uploader(
            "Upload Predictions CSV",
            type=["csv"],
            help="Upload the submission.csv file with predictions",
        )

        # Or select from outputs directory
        st.markdown("**Or select from outputs:**")
        outputs_dir = Path(project_root) / "outputs"
        if outputs_dir.exists():
            prediction_files = list(outputs_dir.rglob("**/predictions/submission.csv"))
            if prediction_files:
                file_options = [str(f.relative_to(outputs_dir)) for f in prediction_files]
                selected_file = st.selectbox(
                    "Select prediction file",
                    [""] + file_options,
                    help="Choose from existing prediction files",
                )
                if selected_file:
                    predictions_file = outputs_dir / selected_file

        # Image directory
        image_dir = st.text_input(
            "Image Directory",
            value="data/datasets/images/test",
            help="Path to directory containing test images",
        )

    with col2:
        if predictions_file is not None:
            try:
                # Load predictions
                if hasattr(predictions_file, "read"):  # Uploaded file
                    df = pd.read_csv(predictions_file)
                else:  # File path
                    df = pd.read_csv(predictions_file)

                # Add advanced analysis tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Analysis", "🖼️ Images", "📈 Advanced"])

                with tab1:
                    display_dataset_overview(df)

                with tab2:
                    display_prediction_analysis(df)

                with tab3:
                    display_image_viewer(df, image_dir)

                with tab4:
                    display_advanced_analysis(df, image_dir)

            except Exception as e:
                st.error(f"Error loading predictions: {str(e)}")
        else:
            st.info("Please upload a predictions CSV file or select one from outputs.")


def display_advanced_analysis(df: pd.DataFrame, image_dir: str):
    """Display advanced analysis features."""
    st.subheader("🔬 Advanced Analysis")

    # Calculate metrics
    df = calculate_prediction_metrics(df)

    # Performance sorting and filtering
    st.markdown("### Performance Sorting & Filtering")

    col1, col2, col3 = st.columns(3)

    with col1:
        sort_by = st.selectbox(
            "Sort by",
            ["prediction_count", "total_area", "avg_confidence", "aspect_ratio"],
            help="Sort images by different metrics",
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

    # Apply sorting and filtering
    sorted_df = apply_sorting_filtering(df, sort_by, sort_order, filter_metric)

    # Display top performers
    display_image_grid(sorted_df, image_dir, sort_by)

    # Statistical analysis
    display_statistical_summary(df)

    # Export capabilities
    export_results(df, sorted_df)


def export_results(original_df: pd.DataFrame, sorted_df: pd.DataFrame):
    """Provide export capabilities for results."""
    st.markdown("### Export Results")
    col1, col2 = st.columns(2)

    with col1:
        # Export sorted results
        csv_data = sorted_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sorted Results",
            data=csv_data,
            file_name="sorted_predictions.csv",
            mime="text/csv",
        )

    with col2:
        # Export summary statistics
        df_export, summary_stats = prepare_export_data(original_df)
        summary_csv = pd.DataFrame([summary_stats]).to_csv(index=False)
        st.download_button(
            label="📊 Download Summary Stats",
            data=summary_csv,
            file_name="summary_statistics.csv",
            mime="text/csv",
        )


def render_model_comparison():
    """Render model comparison interface."""
    st.header("⚖️ Model Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model A")

        # File selection for Model A
        model_a_file = st.file_uploader("Upload Model A Predictions", type=["csv"], key="model_a")

        # Or select from outputs
        st.markdown("**Or select from outputs:**")
        outputs_dir = Path(project_root) / "outputs"
        if outputs_dir.exists():
            prediction_files = list(outputs_dir.rglob("**/predictions/submission.csv"))
            if prediction_files:
                file_options = [str(f.relative_to(outputs_dir)) for f in prediction_files]
                selected_a = st.selectbox("Select Model A file", [""] + file_options, key="select_a")
                if selected_a:
                    model_a_file = outputs_dir / selected_a

    with col2:
        st.subheader("Model B")

        # File selection for Model B
        model_b_file = st.file_uploader("Upload Model B Predictions", type=["csv"], key="model_b")

        # Or select from outputs
        st.markdown("**Or select from outputs:**")
        if outputs_dir.exists():
            prediction_files = list(outputs_dir.rglob("**/predictions/submission.csv"))
            if prediction_files:
                file_options = [str(f.relative_to(outputs_dir)) for f in prediction_files]
                selected_b = st.selectbox("Select Model B file", [""] + file_options, key="select_b")
                if selected_b:
                    model_b_file = outputs_dir / selected_b

    # Image directory
    image_dir = st.text_input(
        "Image Directory",
        value="data/datasets/images/test",
        help="Path to directory containing test images",
        key="comparison_image_dir",
    )

    if model_a_file is not None and model_b_file is not None:
        try:
            # Load both models' predictions
            if hasattr(model_a_file, "read"):
                df_a = pd.read_csv(model_a_file)
            else:
                df_a = pd.read_csv(model_a_file)

            if hasattr(model_b_file, "read"):
                df_b = pd.read_csv(model_b_file)
            else:
                df_b = pd.read_csv(model_b_file)

            # Comparison tabs
            tab1, tab2, tab3 = st.tabs(["📊 Statistics", "🔍 Differences", "🖼️ Visual Comparison"])

            with tab1:
                display_model_comparison_stats(df_a, df_b)

            with tab2:
                display_model_differences(df_a, df_b)

            with tab3:
                display_visual_comparison(df_a, df_b, image_dir)

        except Exception as e:
            st.error(f"Error loading predictions: {str(e)}")
    else:
        st.info("Please upload or select prediction files for both models to compare.")


def display_model_differences(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """Display differences between model predictions."""
    st.subheader("🔍 Model Differences")

    # Calculate differences
    diff_df = calculate_image_differences(df_a, df_b)

    if diff_df.empty:
        st.warning("No common images found between the two models.")
        return

    # Display top differences
    st.markdown("### Top Differences in Prediction Count")
    diff_df_sorted = diff_df.sort_values("abs_pred_diff", ascending=False)
    st.dataframe(diff_df_sorted.head(20)[["filename", "pred_a", "pred_b", "pred_diff"]])

    st.markdown("### Top Differences in Total Area")
    diff_df_sorted = diff_df.sort_values("abs_area_diff", ascending=False)
    st.dataframe(diff_df_sorted.head(20)[["filename", "area_a", "area_b", "area_diff"]])


def render_comparison_view():
    """Render comparison view interface."""
    st.header("⚖️ Model Comparison")

    st.info("Comparison view functionality coming soon!")
    st.markdown(
        """
    Planned features:
    - Load multiple prediction files
    - Compare metrics across runs
    - Side-by-side performance analysis
    - Statistical significance testing
    """
    )


def render_image_gallery():
    """Render image gallery interface."""
    st.header("🖼️ Image Gallery")

    st.info("Image gallery functionality coming soon!")
    st.markdown(
        """
    Planned features:
    - Browse images with predictions
    - Filter by prediction confidence
    - Sort by various metrics
    - Batch image viewing
    """
    )


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="OCR Evaluation Viewer", page_icon="📊", layout="wide")

    st.title("📊 OCR Evaluation Results Viewer")
    st.markdown("Analyze predictions, view metrics, and compare model performance")

    # Initialize utilities
    try:
        ConfigParser()
    except Exception as e:
        print(f"Warning: Failed to initialize config parser: {e}")

    # Sidebar for navigation
    st.sidebar.header("Navigation")
    view_mode = st.sidebar.selectbox(
        "View Mode",
        ["Single Run Analysis", "Comparison View", "Image Gallery"],
        help="Choose how to analyze the evaluation results",
    )

    if view_mode == "Single Run Analysis":
        render_single_run_analysis()
    elif view_mode == "Comparison View":
        render_model_comparison()  # Use the actual implementation instead of placeholder
    elif view_mode == "Image Gallery":
        render_image_gallery()


if __name__ == "__main__":
    main()
