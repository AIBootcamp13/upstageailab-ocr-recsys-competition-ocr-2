"""
Advanced side-by-side viewer for Streamlit Preprocessing Viewer.

Allows users to compare any two stages from the preprocessing pipeline
with interactive selection and detailed metadata display.
"""

import cv2
import numpy as np
import streamlit as st


class SideBySideViewer:
    """
    Advanced side-by-side image comparison viewer.

    Provides interactive selection of pipeline stages for comparison,
    with zoom controls, metadata display, and difference visualization.
    """

    def __init__(self, pipeline_orchestrator):
        """
        Initialize the side-by-side viewer.

        Args:
            pipeline_orchestrator: Pipeline orchestrator with stage information
        """
        self.pipeline = pipeline_orchestrator

    def render_comparison(self, pipeline_results: dict[str, np.ndarray | str], available_stages: list[str]) -> None:
        """
        Render the side-by-side comparison interface.

        Args:
            pipeline_results: Results dictionary from pipeline processing
            available_stages: List of available pipeline stages
        """
        st.subheader("⚖️ Pipeline Stage Comparison")

        # Filter stages that actually have results
        valid_stages = [stage for stage in available_stages if stage in pipeline_results]

        if len(valid_stages) < 2:
            st.warning("Need at least 2 pipeline stages to compare. Run preprocessing first.")
            return

        # Stage selection
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Left Image**")
            left_stage = st.selectbox(
                "Select stage for left panel",
                valid_stages,
                index=0 if valid_stages else 0,
                key="left_stage_select",
                help="Choose which pipeline stage to display on the left",
            )

        with col2:
            st.markdown("**Right Image**")
            # Default to final stage for right panel
            default_right = "final" if "final" in valid_stages else valid_stages[-1]
            right_stage = st.selectbox(
                "Select stage for right panel",
                valid_stages,
                index=valid_stages.index(default_right) if default_right in valid_stages else 0,
                key="right_stage_select",
                help="Choose which pipeline stage to display on the right",
            )

        # Display comparison
        if left_stage and right_stage:
            self._display_comparison(pipeline_results, left_stage, right_stage)

    def _display_comparison(self, pipeline_results: dict[str, np.ndarray | str], left_stage: str, right_stage: str) -> None:
        """
        Display the actual side-by-side comparison.

        Args:
            pipeline_results: Pipeline results dictionary
            left_stage: Stage name for left panel
            right_stage: Stage name for right panel
        """
        left_image = pipeline_results.get(left_stage)
        right_image = pipeline_results.get(right_stage)

        if not isinstance(left_image, np.ndarray) or not isinstance(right_image, np.ndarray):
            st.error("Selected stages do not contain valid image data")
            return

        # Convert BGR to RGB for display
        left_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

        # Display options
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            show_metadata = st.checkbox("Show Metadata", value=True, key="show_metadata")

        with col2:
            show_difference = st.checkbox("Show Difference", value=False, key="show_difference")

        with col3:
            zoom_level = st.slider("Zoom Level", min_value=0.1, max_value=2.0, value=1.0, step=0.1, key="zoom_slider")

        # Create display images with zoom
        left_display = self._apply_zoom(left_rgb, zoom_level)
        right_display = self._apply_zoom(right_rgb, zoom_level)

        # Main comparison display
        if show_difference:
            self._display_difference_view(left_display, right_display, left_stage, right_stage)
        else:
            self._display_side_by_side(left_display, right_display, left_stage, right_stage)

        # Metadata display
        if show_metadata:
            self._display_metadata(pipeline_results, left_stage, right_stage)

    def _display_side_by_side(self, left_image: np.ndarray, right_image: np.ndarray, left_stage: str, right_stage: str) -> None:
        """Display images side by side."""
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{left_stage.replace('_', ' ').title()}**")
            st.image(left_image, use_column_width=True, caption=self.pipeline.get_stage_description(left_stage))

        with col2:
            st.markdown(f"**{right_stage.replace('_', ' ').title()}**")
            st.image(right_image, use_column_width=True, caption=self.pipeline.get_stage_description(right_stage))

    def _display_difference_view(self, left_image: np.ndarray, right_image: np.ndarray, left_stage: str, right_stage: str) -> None:
        """Display difference between the two images."""
        # Ensure images are the same size for difference calculation
        if left_image.shape != right_image.shape:
            # Resize images to match
            min_height = min(left_image.shape[0], right_image.shape[0])
            min_width = min(left_image.shape[1], right_image.shape[1])

            left_resized = cv2.resize(left_image, (min_width, min_height))
            right_resized = cv2.resize(right_image, (min_width, min_height))
        else:
            left_resized = left_image
            right_resized = right_image

        # Calculate difference
        diff = cv2.absdiff(left_resized, right_resized)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

        # Enhance difference for visibility
        diff_enhanced = cv2.applyColorMap(diff_gray, cv2.COLORMAP_HOT)

        # Display
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"**{left_stage.replace('_', ' ').title()}**")
            st.image(left_image, use_column_width=True)

        with col2:
            st.markdown("**Difference**")
            st.image(diff_enhanced, use_column_width=True, caption="Hot colormap shows differences (red = high difference)")

        with col3:
            st.markdown(f"**{right_stage.replace('_', ' ').title()}**")
            st.image(right_image, use_column_width=True)

    def _apply_zoom(self, image: np.ndarray, zoom_level: float) -> np.ndarray:
        """Apply zoom to image."""
        if zoom_level == 1.0:
            return image

        height, width = image.shape[:2]
        new_width = int(width * zoom_level)
        new_height = int(height * zoom_level)

        zoomed = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return zoomed

    def _display_metadata(self, pipeline_results: dict[str, np.ndarray | str], left_stage: str, right_stage: str) -> None:
        """Display metadata for the compared stages."""
        st.markdown("---")
        st.subheader("📊 Stage Information")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{left_stage.replace('_', ' ').title()}**")
            left_data = pipeline_results.get(left_stage)
            if isinstance(left_data, np.ndarray):
                st.write(f"Shape: {left_data.shape}")
                st.write(f"Data type: {left_data.dtype}")
                st.write(f"Size: {left_data.size:,} pixels")

                # Color analysis
                if len(left_data.shape) == 3:
                    means = cv2.mean(left_data)
                    st.write(f"Mean RGB: ({means[0]:.1f}, {means[1]:.1f}, {means[2]:.1f})")
            else:
                st.write("No image data available")

        with col2:
            st.markdown(f"**{right_stage.replace('_', ' ').title()}**")
            right_data = pipeline_results.get(right_stage)
            if isinstance(right_data, np.ndarray):
                st.write(f"Shape: {right_data.shape}")
                st.write(f"Data type: {right_data.dtype}")
                st.write(f"Size: {right_data.size:,} pixels")

                # Color analysis
                if len(right_data.shape) == 3:
                    means = cv2.mean(right_data)
                    st.write(f"Mean RGB: ({means[0]:.1f}, {means[1]:.1f}, {means[2]:.1f})")
            else:
                st.write("No image data available")

        # Processing summary
        if "error" in pipeline_results:
            st.error(f"Processing Error: {pipeline_results['error']}")


def render_roi_tool() -> tuple[int, int, int, int] | None:
    """
    Render ROI selection tool.

    Returns:
        ROI coordinates as (x, y, w, h) tuple or None if not selected
    """
    st.subheader("🎯 Region of Interest (ROI) Tool")

    st.markdown("""
    Select a region of interest on the image above to apply preprocessing only to that area.
    This enables faster parameter tuning with near-instantaneous feedback.
    """)

    # ROI coordinate inputs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        roi_x = st.number_input("X", min_value=0, value=0, help="X coordinate of ROI top-left corner")

    with col2:
        roi_y = st.number_input("Y", min_value=0, value=0, help="Y coordinate of ROI top-left corner")

    with col3:
        roi_w = st.number_input("Width", min_value=1, value=100, help="Width of ROI rectangle")

    with col4:
        roi_h = st.number_input("Height", min_value=1, value=100, help="Height of ROI rectangle")

    if st.button("Apply ROI", key="apply_roi"):
        return (roi_x, roi_y, roi_w, roi_h)

    if st.button("Clear ROI", key="clear_roi"):
        return None

    return None
