"""
Main application module for OCR Evaluation Viewer.
"""

import sys
from pathlib import Path

import streamlit as st

# --- Project Setup ---
# Resolve and add the project root to the system path for module imports.
# This ensures that modules like `data_utils` and `visualization` can be found.
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import view modules
from .comparison import render_comparison_view
from .gallery import render_image_gallery
from .single_run import render_single_run_analysis


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
