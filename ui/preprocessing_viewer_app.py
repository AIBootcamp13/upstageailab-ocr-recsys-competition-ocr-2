"""
Streamlit Preprocessing Viewer

Interactive preprocessing pipeline visualization and testing tool.
Allows users to upload images and see the effects of each preprocessing step
in real-time with side-by-side comparisons.
"""

import logging
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import streamlit as st
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in st.session_state.get("sys_path", []):
    import sys

    sys.path.insert(0, str(project_root))
    if "sys_path" not in st.session_state:
        st.session_state.sys_path = []
    st.session_state.sys_path.append(str(project_root))

try:
    from ui.preprocessing_viewer.parameter_controls import ParameterControls
    from ui.preprocessing_viewer.pipeline import PreprocessingViewerPipeline
    from ui.preprocessing_viewer.pipeline_visualizer import PipelineVisualizer
    from ui.preprocessing_viewer.preset_manager import PresetManager, render_preset_management
    from ui.preprocessing_viewer.side_by_side_viewer import SideBySideViewer
except ImportError as e:
    st.error(f"Failed to import preprocessing viewer components: {e}")
    st.info("Make sure all preprocessing viewer modules are available.")
    st.stop()


def main():
    """Main preprocessing viewer application."""
    st.set_page_config(page_title="OCR Preprocessing Viewer", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

    st.title("🔍 OCR Preprocessing Viewer")
    st.markdown("Interactive preprocessing pipeline visualization and testing")

    # Initialize components
    pipeline = PreprocessingViewerPipeline()
    pipeline_visualizer = PipelineVisualizer(pipeline)
    side_by_side_viewer = SideBySideViewer(pipeline)
    preset_manager = PresetManager()
    parameter_controls = ParameterControls()

    if "viewer_config" not in st.session_state:
        base_defaults = preset_manager.get_default_config()
        parameter_defaults = parameter_controls.reset_to_defaults()
        base_defaults.update(parameter_defaults)
        st.session_state.viewer_config = base_defaults
        st.session_state.current_config = base_defaults.copy()
    else:
        if "current_config" not in st.session_state:
            st.session_state.current_config = st.session_state.viewer_config.copy()

    def handle_config_change(updated_config: dict[str, Any]) -> None:
        st.session_state.viewer_config = dict(updated_config)
        st.session_state.current_config = dict(updated_config)

    uploaded_file = None
    with st.sidebar:
        st.header("⚙️ Controls")

        uploaded_file = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png", "bmp"], help="Upload a receipt or document image to preprocess"
        )

        # Make current config available to preset manager helpers
        st.session_state.current_config = st.session_state.viewer_config.copy()

        preset_override = render_preset_management(preset_manager)
        if preset_override:
            st.session_state.viewer_config = dict(preset_override)
            st.session_state.current_config = dict(preset_override)
            st.rerun()

        st.subheader("📤 Export")
        export_yaml = yaml.dump(st.session_state.viewer_config, default_flow_style=False, sort_keys=True)
        st.download_button(
            label="Download Config",
            data=export_yaml,
            file_name="preprocessing_config.yaml",
            mime="text/yaml",
        )

    image: np.ndarray | None = None
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Failed to load image. Please try a different file.")

    tab_params, tab_full, tab_step = st.tabs(
        [
            "🛠️ Parameter Controls",
            "🎬 Full Pipeline",
            "🎯 Step-by-Step Visualizer",
        ]
    )

    with tab_params:
        st.caption("Adjust preprocessing parameters and see validation feedback instantly.")
        if st.button("Reset Parameters to Defaults", key="reset_parameters"):
            defaults = parameter_controls.reset_to_defaults()
            st.session_state.viewer_config = defaults
            st.session_state.current_config = defaults.copy()
            st.success("Parameters reset to defaults.")
            st.rerun()

        parameter_controls.render_parameter_panels(
            st.session_state.viewer_config,
            handle_config_change,
        )
        # Note: Don't update session_state here - parameter_controls uses on_change_callback
        # Updating here causes infinite rerun loop since render_parameter_panels always returns a new dict

        with st.expander("Current Configuration (JSON)", expanded=False):
            st.json(st.session_state.viewer_config)

    with tab_full:
        if image is None:
            st.info("Upload an image using the sidebar to run the preprocessing pipeline.")
        else:
            st.success(f"Image loaded successfully! Shape: {image.shape}")
            with st.spinner("Running preprocessing pipeline..."):
                try:
                    pipeline_results = pipeline.process_with_intermediates(image, st.session_state.viewer_config)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Error processing image: {exc}")
                    st.exception(exc)
                else:
                    st.header("📊 Pipeline Results")
                    available_stages = [stage for stage in pipeline.get_available_stages() if stage in pipeline_results]
                    image_stages = [
                        stage for stage in available_stages if SideBySideViewer.is_displayable_image(pipeline_results.get(stage))
                    ]

                    if image_stages:
                        side_by_side_viewer.render_comparison(pipeline_results, image_stages)
                    else:
                        st.info("No image stages available for side-by-side comparison.")

                    if available_stages:
                        st.header("🎭 Individual Stages")
                        cols = st.columns(min(len(available_stages), 4))
                        for index, stage in enumerate(available_stages):
                            with cols[index % 4]:
                                st.subheader(stage.replace("_", " ").title())
                                result = pipeline_results.get(stage)
                                if SideBySideViewer.is_displayable_image(result):
                                    display_image = SideBySideViewer.prepare_image_for_display(cast(np.ndarray, result))
                                    st.image(
                                        display_image,
                                        caption=pipeline.get_stage_description(stage),
                                        use_column_width=True,
                                    )
                                elif isinstance(result, np.ndarray):
                                    st.write("Array data:")
                                    st.write(result.tolist())
                                else:
                                    st.write(f"Result: {result}")
                    else:
                        st.warning("No pipeline stages available to display.")

    with tab_step:
        if image is None:
            st.info("Upload an image to execute the preprocessing pipeline stage-by-stage.")
        else:
            pipeline_visualizer.initialize_with_image(image)
            stages = pipeline_visualizer.get_pipeline_stages()
            pipeline_visualizer.render_stage_controls(stages, st.session_state.viewer_config)
            pipeline_visualizer.render_stage_results()

    if image is None:
        st.info("👆 Upload an image using the sidebar to get started!")
        st.header("📖 How to Use")
        st.markdown(
            """
        1. **Upload an Image**: Use the sidebar to upload a receipt or document image
        2. **Configure Parameters**: Adjust preprocessing options on the *Parameter Controls* tab
        3. **Run Full Pipeline**: Review automated preprocessing results and compare stages
        4. **Inspect Step-by-Step**: Execute stages individually for deeper analysis
        5. **Manage Presets**: Save or load parameter presets from the sidebar
        6. **Export Config**: Download the active configuration as YAML for experiments

        ### Available Preprocessing Stages
        - **Original**: The uploaded image as-is
        - **Document Detection**: Finds document boundaries
        - **Perspective Correction**: Straightens the document
        - **Noise Elimination**: Removes image noise
        - **Brightness Adjustment**: Optimizes image brightness
        - **Enhancement**: Applies final image enhancements
        - **Final**: The fully preprocessed result
        """
        )


if __name__ == "__main__":
    main()
