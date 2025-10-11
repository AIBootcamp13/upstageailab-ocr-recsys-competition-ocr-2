"""
Model comparison view for OCR Evaluation Viewer.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# --- Project Setup ---
# Resolve and add the project root to the system path for module imports.
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import utilities
from ui.visualization import display_model_comparison_stats, display_model_differences, display_visual_comparison


def render_comparison_view():
    """Renders the UI for comparing two model prediction files."""
    st.header("⚖️ Model Comparison")

    model_a_file = None
    model_b_file = None

    outputs_dir = project_root / "outputs"
    prediction_files = []
    file_options = {}

    if outputs_dir.exists():
        prediction_files = sorted(outputs_dir.rglob("**/predictions/submission.csv"), reverse=True)
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

    gt_file = st.file_uploader("Upload Ground Truth (optional)", type=["csv"], key="gt_upload")

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
        gt_df = validate_predictions_file(gt_file) if gt_file else None

        if df_a is not None and df_b is not None:
            st.success("Successfully loaded both prediction files.")
            if gt_df is not None:
                st.success("Ground truth loaded successfully.")

            comp_tab1, comp_tab2, comp_tab3 = st.tabs(["📊 Statistics", "🔍 Differences", "🖼️ Visual Comparison"])

            with comp_tab1:
                display_model_comparison_stats(df_a, df_b)
            with comp_tab2:
                display_model_differences(df_a, df_b)
            with comp_tab3:
                if image_dir.is_dir():
                    display_visual_comparison(df_a, df_b, str(image_dir), gt_df)
                else:
                    st.warning("The specified image directory does not exist. Please provide a valid path.")

        else:
            st.error("Failed to load one or both prediction files. Please check the files and try again.")
    else:
        st.info("Please provide two prediction files to compare.")
