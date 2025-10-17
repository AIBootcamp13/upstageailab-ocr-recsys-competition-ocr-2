"""
Advanced side-by-side viewer for Streamlit Preprocessing Viewer.

Allows users to compare any two stages from the preprocessing pipeline
with interactive selection and detailed metadata display.
"""

from typing import Any, cast

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

        # Filter stages that have displayable image results
        valid_stages = [
            stage for stage in available_stages if stage in pipeline_results and self.is_displayable_image(pipeline_results.get(stage))
        ]

        if len(valid_stages) < 2:
            st.warning("Need at least 2 pipeline stages to compare. Run preprocessing first.")
            return

        # Layout selection
        layout_options = ["Side by Side (2 images)", "Grid (2x2)", "Horizontal (3 images)", "Vertical (3 images)"]
        selected_layout = st.selectbox(
            "Comparison Layout", layout_options, index=0, key="comparison_layout", help="Choose how to display multiple pipeline stages"
        )

        # Stage selection based on layout
        if selected_layout == "Side by Side (2 images)":
            self._render_side_by_side_selection(valid_stages)
        elif selected_layout == "Grid (2x2)":
            self._render_grid_selection(valid_stages)
        elif selected_layout == "Horizontal (3 images)":
            self._render_horizontal_selection(valid_stages)
        elif selected_layout == "Vertical (3 images)":
            self._render_vertical_selection(valid_stages)

        # Get selected stages based on layout
        selected_stages = self._get_selected_stages_for_layout(selected_layout, valid_stages)

        if selected_stages:
            self._display_comparison_layout(pipeline_results, selected_stages, selected_layout)

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

        if not self.is_displayable_image(left_image) or not self.is_displayable_image(right_image):
            st.error("Selected stages do not contain valid image data")
            return

        # Prepare images for display
        left_rgb = self.prepare_image_for_display(cast(np.ndarray, left_image))
        right_rgb = self.prepare_image_for_display(cast(np.ndarray, right_image))

        # Display options
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            show_metadata = st.checkbox("Show Metadata", value=True, key="show_metadata")

        with col2:
            show_difference = st.checkbox("Show Difference", value=False, key="show_difference")

        with col3:
            # Placeholder for annotations - will be implemented later
            st.checkbox("Show Annotations", value=False, key="show_annotations", disabled=True, help="Annotation overlay coming soon")

        with col4:
            zoom_level = st.slider("Zoom Level", min_value=0.1, max_value=3.0, value=1.0, step=0.1, key="zoom_slider")

        with col4:
            enable_pan = st.checkbox("Enable Pan", value=False, key="enable_pan") if zoom_level > 1.0 else False

        # Pan controls (only show when zoom > 1.0 and pan is enabled)
        pan_x, pan_y = 0.0, 0.0
        if enable_pan and zoom_level > 1.0:
            st.markdown("**Pan Controls**")
            pan_col1, pan_col2 = st.columns(2)
            with pan_col1:
                pan_x = st.slider("Pan X", min_value=-0.5, max_value=0.5, value=0.0, step=0.01, key="pan_x")
            with pan_col2:
                pan_y = st.slider("Pan Y", min_value=-0.5, max_value=0.5, value=0.0, step=0.01, key="pan_y")

            # Quick pan buttons
            btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5)
            with btn_col1:
                if st.button("⬅️", key="pan_left"):
                    pan_x = max(pan_x - 0.1, -0.5)
                    st.session_state.pan_x = pan_x
            with btn_col2:
                if st.button("➡️", key="pan_right"):
                    pan_x = min(pan_x + 0.1, 0.5)
                    st.session_state.pan_x = pan_x
            with btn_col3:
                if st.button("⬆️", key="pan_up"):
                    pan_y = max(pan_y - 0.1, -0.5)
                    st.session_state.pan_y = pan_y
            with btn_col4:
                if st.button("⬇️", key="pan_down"):
                    pan_y = min(pan_y + 0.1, 0.5)
                    st.session_state.pan_y = pan_y
            with btn_col5:
                if st.button("🔄 Reset", key="pan_reset"):
                    pan_x, pan_y = 0.0, 0.0
                    st.session_state.pan_x = 0.0
                    st.session_state.pan_y = 0.0

        # Create display images with zoom and pan
        left_display = self._apply_zoom_and_pan(left_rgb, zoom_level, pan_x, pan_y)
        right_display = self._apply_zoom_and_pan(right_rgb, zoom_level, pan_x, pan_y)

        # Main comparison display
        if show_difference:
            self._display_difference_view(left_display, right_display, left_stage, right_stage)
        else:
            self._display_side_by_side(left_display, right_display, left_stage, right_stage)

        # Metadata display
        if show_metadata:
            self._display_metadata(pipeline_results, left_stage, right_stage)

    def _render_side_by_side_selection(self, valid_stages: list[str]) -> None:
        """Render stage selection for side-by-side layout."""
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Left Image**")
            left_stage = st.selectbox(
                "Select stage for left panel",
                valid_stages,
                index=0,
                key="left_stage_select",
                help="Choose which pipeline stage to display on the left",
            )
            st.session_state.selected_left = left_stage

        with col2:
            st.markdown("**Right Image**")
            default_right = "final" if "final" in valid_stages else valid_stages[-1]
            right_stage = st.selectbox(
                "Select stage for right panel",
                valid_stages,
                index=valid_stages.index(default_right),
                key="right_stage_select",
                help="Choose which pipeline stage to display on the right",
            )
            st.session_state.selected_right = right_stage

    def _render_grid_selection(self, valid_stages: list[str]) -> None:
        """Render stage selection for 2x2 grid layout."""
        st.markdown("**Select 4 stages for 2x2 grid comparison**")
        cols = st.columns(2)
        defaults = ["original", "document_detection", "perspective_correction", "final"]

        for i in range(4):
            with cols[i % 2]:
                stage_name = f"Stage {i + 1}"
                default_idx = min(i, len(valid_stages) - 1)
                if i < len(defaults) and defaults[i] in valid_stages:
                    default_idx = valid_stages.index(defaults[i])

                selected = st.selectbox(
                    f"Select {stage_name}",
                    valid_stages,
                    index=default_idx,
                    key=f"grid_stage_{i}",
                    help=f"Choose pipeline stage for {stage_name}",
                )
                st.session_state[f"grid_stage_{i}"] = selected

    def _render_horizontal_selection(self, valid_stages: list[str]) -> None:
        """Render stage selection for horizontal 3-image layout."""
        st.markdown("**Select 3 stages for horizontal comparison**")
        cols = st.columns(3)
        defaults = ["original", "perspective_correction", "final"]

        for i in range(3):
            with cols[i]:
                stage_name = f"Stage {i + 1}"
                default_idx = min(i, len(valid_stages) - 1)
                if i < len(defaults) and defaults[i] in valid_stages:
                    default_idx = valid_stages.index(defaults[i])

                selected = st.selectbox(
                    f"Select {stage_name}",
                    valid_stages,
                    index=default_idx,
                    key=f"horizontal_stage_{i}",
                    help=f"Choose pipeline stage for {stage_name}",
                )
                st.session_state[f"horizontal_stage_{i}"] = selected

    def _render_vertical_selection(self, valid_stages: list[str]) -> None:
        """Render stage selection for vertical 3-image layout."""
        st.markdown("**Select 3 stages for vertical comparison**")
        defaults = ["original", "perspective_correction", "final"]

        for i in range(3):
            stage_name = f"Stage {i + 1}"
            default_idx = min(i, len(valid_stages) - 1)
            if i < len(defaults) and defaults[i] in valid_stages:
                default_idx = valid_stages.index(defaults[i])

            selected = st.selectbox(
                f"Select {stage_name}",
                valid_stages,
                index=default_idx,
                key=f"vertical_stage_{i}",
                help=f"Choose pipeline stage for {stage_name}",
            )
            st.session_state[f"vertical_stage_{i}"] = selected

    def _get_selected_stages_for_layout(self, layout: str, valid_stages: list[str]) -> list[str]:
        """Get selected stages based on layout type."""
        if layout == "Side by Side (2 images)":
            left = st.session_state.get("selected_left")
            right = st.session_state.get("selected_right")
            return [left, right] if left and right else []

        elif layout == "Grid (2x2)":
            stages = []
            for i in range(4):
                stage = st.session_state.get(f"grid_stage_{i}")
                if stage:
                    stages.append(stage)
            return stages

        elif layout == "Horizontal (3 images)":
            stages = []
            for i in range(3):
                stage = st.session_state.get(f"horizontal_stage_{i}")
                if stage:
                    stages.append(stage)
            return stages

        elif layout == "Vertical (3 images)":
            stages = []
            for i in range(3):
                stage = st.session_state.get(f"vertical_stage_{i}")
                if stage:
                    stages.append(stage)
            return stages

        return []

    def _display_comparison_layout(self, pipeline_results: dict[str, np.ndarray | str], selected_stages: list[str], layout: str) -> None:
        """Display comparison based on selected layout."""
        # Get zoom and pan settings
        zoom_level = st.session_state.get("zoom_slider", 1.0)
        enable_pan = st.session_state.get("enable_pan", False) if zoom_level > 1.0 else False
        pan_x = st.session_state.get("pan_x", 0.0) if enable_pan else 0.0
        pan_y = st.session_state.get("pan_y", 0.0) if enable_pan else 0.0

        # Process images
        processed_images: list[tuple[str, np.ndarray | None]] = []
        for stage in selected_stages:
            image = pipeline_results.get(stage)
            if self.is_displayable_image(image):
                rgb_image = self.prepare_image_for_display(cast(np.ndarray, image))
                processed = self._apply_zoom_and_pan(rgb_image, zoom_level, pan_x, pan_y)
                processed_images.append((stage, processed))
            else:
                processed_images.append((stage, None))

        # Display based on layout
        if layout == "Side by Side (2 images)":
            self._display_side_by_side_layout(processed_images)
        elif layout == "Grid (2x2)":
            self._display_grid_layout(processed_images)
        elif layout == "Horizontal (3 images)":
            self._display_horizontal_layout(processed_images)
        elif layout == "Vertical (3 images)":
            self._display_vertical_layout(processed_images)

        # Metadata display
        if st.session_state.get("show_metadata", True):
            self._display_enhanced_metadata(pipeline_results, selected_stages)

    def _display_side_by_side_layout(self, processed_images: list[tuple[str, np.ndarray | None]]) -> None:
        """Display side-by-side layout."""
        if len(processed_images) >= 2:
            left_stage, left_img = processed_images[0]
            right_stage, right_img = processed_images[1]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{left_stage.replace('_', ' ').title()}**")
                if left_img is not None:
                    st.image(left_img, use_column_width=True, caption=self.pipeline.get_stage_description(left_stage))

            with col2:
                st.markdown(f"**{right_stage.replace('_', ' ').title()}**")
                if right_img is not None:
                    st.image(right_img, use_column_width=True, caption=self.pipeline.get_stage_description(right_stage))

    def _display_grid_layout(self, processed_images: list[tuple[str, np.ndarray | None]]) -> None:
        """Display 2x2 grid layout."""
        if len(processed_images) >= 4:
            cols = st.columns(2)
            for i, (stage, img) in enumerate(processed_images[:4]):
                with cols[i % 2]:
                    st.markdown(f"**{stage.replace('_', ' ').title()}**")
                    if img is not None:
                        st.image(img, use_column_width=True, caption=self.pipeline.get_stage_description(stage))

    def _display_horizontal_layout(self, processed_images: list[tuple[str, np.ndarray | None]]) -> None:
        """Display horizontal 3-image layout."""
        if len(processed_images) >= 3:
            cols = st.columns(3)
            for i, (stage, img) in enumerate(processed_images[:3]):
                with cols[i]:
                    st.markdown(f"**{stage.replace('_', ' ').title()}**")
                    if img is not None:
                        st.image(img, use_column_width=True, caption=self.pipeline.get_stage_description(stage))

    def _display_vertical_layout(self, processed_images: list[tuple[str, np.ndarray | None]]) -> None:
        """Display vertical 3-image layout."""
        for stage, img in processed_images[:3]:
            st.markdown(f"**{stage.replace('_', ' ').title()}**")
            if img is not None:
                st.image(img, use_column_width=True, caption=self.pipeline.get_stage_description(stage))
            st.markdown("---")

    def _display_enhanced_metadata(self, pipeline_results: dict[str, np.ndarray | str], selected_stages: list[str]) -> None:
        """Display enhanced metadata with histograms for selected stages."""
        st.markdown("---")
        st.subheader("📊 Enhanced Stage Information")

        # Create tabs for each stage
        tabs = st.tabs([stage.replace("_", " ").title() for stage in selected_stages])

        for tab, stage in zip(tabs, selected_stages, strict=True):
            with tab:
                self._display_single_stage_metadata(pipeline_results, stage)

    def _display_single_stage_metadata(self, pipeline_results: dict[str, np.ndarray | str], stage: str) -> None:
        """Display detailed metadata and histogram for a single stage."""
        data = pipeline_results.get(stage)

        if isinstance(data, np.ndarray) and self.is_displayable_image(data):
            # Basic info in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Shape", f"{data.shape[0]}×{data.shape[1]}")
            with col2:
                st.metric("Data Type", str(data.dtype))
            with col3:
                st.metric("Size", f"{data.size:,} px")
            with col4:
                memory_mb = data.nbytes / (1024 * 1024)
                st.metric("Memory", f"{memory_mb:.1f} MB")

            # Color analysis for RGB images
            if data.ndim == 3 and data.shape[2] == 3:
                st.markdown("**Color Analysis**")
                analysis_cols = st.columns(4)

                with analysis_cols[0]:
                    means = cv2.mean(data)
                    st.write("**Mean RGB:**")
                    st.write(f"R: {means[2]:.1f}")
                    st.write(f"G: {means[1]:.1f}")
                    st.write(f"B: {means[0]:.1f}")

                with analysis_cols[1]:
                    stds = [np.std(data[:, :, i]) for i in range(3)]
                    st.write("**Std RGB:**")
                    st.write(f"R: {stds[2]:.1f}")
                    st.write(f"G: {stds[1]:.1f}")
                    st.write(f"B: {stds[0]:.1f}")

                with analysis_cols[2]:
                    mins = [np.min(data[:, :, i]) for i in range(3)]
                    maxs = [np.max(data[:, :, i]) for i in range(3)]
                    st.write("**Range RGB:**")
                    st.write(f"R: {mins[2]}-{maxs[2]}")
                    st.write(f"G: {mins[1]}-{maxs[1]}")
                    st.write(f"B: {mins[0]}-{maxs[0]}")

                with analysis_cols[3]:
                    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    brightness = float(gray.mean())
                    contrast = float(gray.std())
                    st.write(f"**Brightness:** {brightness:.1f}")
                    st.write(f"**Contrast:** {contrast:.1f}")

                st.markdown("**Color Histograms**")
                hist_cols = st.columns(3)
                colors = [("Red", 2), ("Green", 1), ("Blue", 0)]

                for col, (label, channel) in zip(hist_cols, colors, strict=True):
                    with col:
                        hist = cv2.calcHist([data], [channel], None, [256], [0, 256]).flatten()
                        max_val = hist.max()
                        if max_val > 0:
                            hist_img = np.zeros((100, 256, 3), dtype=np.uint8)
                            for x in range(256):
                                height = int((hist[x] / max_val) * 100)
                                cv2.line(hist_img, (x, 100), (x, 100 - height), (255, 255, 255), 1)

                            st.image(hist_img, caption=f"{label} Channel", use_column_width=True)
                            st.write(f"Peak: {hist.argmax()} ({max_val:.0f} px)")

            elif data.ndim == 2:
                st.markdown("**Grayscale Analysis**")
                analysis_cols = st.columns(3)

                mean_val = float(np.mean(data))
                std_val = float(np.std(data))

                with analysis_cols[0]:
                    st.write(f"**Mean Intensity:** {mean_val:.1f}")
                    st.write(f"**Std Deviation:** {std_val:.1f}")

                with analysis_cols[1]:
                    min_val = int(np.min(data))
                    max_val = int(np.max(data))
                    st.write(f"**Range:** {min_val}-{max_val}")
                    st.write(f"**Dynamic Range:** {max_val - min_val}")

                with analysis_cols[2]:
                    median_val = float(np.median(data))
                    skewness = float(np.mean(((data - mean_val) / std_val) ** 3)) if std_val > 0 else 0.0
                    st.write(f"**Median:** {median_val:.1f}")
                    st.write(f"**Skewness:** {skewness:.2f}")

                st.markdown("**Intensity Histogram**")
                hist = cv2.calcHist([data], [0], None, [256], [0, 256]).flatten()
                max_val = hist.max()
                if max_val > 0:
                    hist_img = np.zeros((100, 256, 3), dtype=np.uint8)
                    for x in range(256):
                        height = int((hist[x] / max_val) * 100)
                        cv2.line(hist_img, (x, 100), (x, 100 - height), (255, 255, 255), 1)

                    st.image(hist_img, caption="Intensity Distribution", use_column_width=True)
                    st.write(f"**Peak Intensity:** {hist.argmax()}")
                    st.write(f"**Peak Count:** {max_val:.0f} pixels")

        elif isinstance(data, np.ndarray):
            st.write("Array data (non-image):")
            st.write(data.tolist())
        else:
            st.write("No image data available for this stage")

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

    @staticmethod
    def is_displayable_image(data: Any) -> bool:
        """Return True when the provided data resembles an image array."""
        if not isinstance(data, np.ndarray):
            return False

        if data.ndim == 2:
            return data.shape[0] > 3 and data.shape[1] > 3

        if data.ndim == 3 and data.shape[2] in (1, 3, 4):
            return data.shape[0] > 3 and data.shape[1] > 3

        return False

    @staticmethod
    def prepare_image_for_display(image: np.ndarray) -> np.ndarray:
        """Convert pipeline outputs to RGB for Streamlit display."""
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if image.ndim == 3:
            channels = image.shape[2]
            if channels == 1:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            if channels == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if channels == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        raise ValueError(f"Unsupported image shape for display: {image.shape if hasattr(image, 'shape') else type(image)}")

    def _apply_zoom_and_pan(self, image: np.ndarray, zoom_level: float, pan_x: float, pan_y: float) -> np.ndarray:
        """Apply zoom and pan to image."""
        if zoom_level == 1.0:
            return image

        height, width = image.shape[:2]
        new_width = int(width * zoom_level)
        new_height = int(height * zoom_level)

        # Resize image
        zoomed = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Apply pan by cropping the zoomed image
        if pan_x != 0.0 or pan_y != 0.0:
            # Calculate crop region
            crop_width = width
            crop_height = height

            # Calculate offset based on pan values (-0.5 to 0.5 range)
            offset_x = int(pan_x * (new_width - crop_width))
            offset_y = int(pan_y * (new_height - crop_height))

            # Ensure offsets are within bounds
            offset_x = max(0, min(offset_x, new_width - crop_width))
            offset_y = max(0, min(offset_y, new_height - crop_height))

            # Crop the zoomed image
            zoomed = zoomed[offset_y : offset_y + crop_height, offset_x : offset_x + crop_width]

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
