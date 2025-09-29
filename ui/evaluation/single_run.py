"""
Single run analysis view for OCR Evaluation Viewer.
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
from ui.data_utils import calculate_prediction_metrics, load_predictions_file
from ui.visualization import (
    display_advanced_analysis,
    display_dataset_overview,
    display_image_viewer,
    display_prediction_analysis,
)


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
                outputs_dir.rglob("**/predictions/submission.csv"),
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

                # Calculate derived metrics for analysis
                df = calculate_prediction_metrics(df)
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
                            st.warning("No image files found in the specified directory.")
                    else:
                        st.warning("The specified image directory does not exist. Please provide a valid path.")

            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            st.info("Please upload a predictions file or select one from the outputs directory to begin analysis.")
