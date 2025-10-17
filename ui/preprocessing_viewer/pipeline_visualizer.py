"""
Step-by-step pipeline visualizer for Streamlit Preprocessing Viewer.

Provides interactive controls to execute preprocessing pipeline stages individually,
with intermediate result caching, progress tracking, and rollback capabilities.
"""

import time
from typing import Any

import cv2
import numpy as np
import streamlit as st


class PipelineVisualizer:
    """
    Interactive step-by-step pipeline visualizer.

    Allows users to execute preprocessing stages individually with full control
    over the pipeline execution flow, intermediate result inspection, and rollback.
    """

    def __init__(self, pipeline_orchestrator):
        """
        Initialize the pipeline visualizer.

        Args:
            pipeline_orchestrator: Pipeline orchestrator instance
        """
        self.pipeline = pipeline_orchestrator
        self._init_session_state()

    def _init_session_state(self):
        """Initialize session state for pipeline visualization."""
        if "pipeline_stage" not in st.session_state:
            st.session_state.pipeline_stage = 0
        if "pipeline_results" not in st.session_state:
            st.session_state.pipeline_results = {}
        if "pipeline_executed_stages" not in st.session_state:
            st.session_state.pipeline_executed_stages = set()
        if "pipeline_current_image" not in st.session_state:
            st.session_state.pipeline_current_image = None
        if "pipeline_stage_times" not in st.session_state:
            st.session_state.pipeline_stage_times = {}

    def reset_pipeline(self):
        """Reset the pipeline execution state."""
        st.session_state.pipeline_stage = 0
        st.session_state.pipeline_results = {}
        st.session_state.pipeline_executed_stages = set()
        st.session_state.pipeline_current_image = None
        st.session_state.pipeline_stage_times = {}

    def get_pipeline_stages(self) -> list[dict[str, Any]]:
        """
        Get the ordered list of pipeline stages with metadata.

        Returns:
            List of stage dictionaries with name, description, and dependencies
        """
        return [
            {
                "name": "original",
                "display_name": "Original Image",
                "description": "Starting point with the original uploaded image",
                "depends_on": [],
                "category": "input",
            },
            {
                "name": "color_preprocessing",
                "display_name": "Color Preprocessing",
                "description": "Grayscale conversion and color inversion",
                "depends_on": ["original"],
                "category": "preprocessing",
            },
            {
                "name": "document_detection",
                "display_name": "Document Detection",
                "description": "Detect document boundaries and corners",
                "depends_on": ["color_preprocessing"],
                "category": "detection",
            },
            {
                "name": "document_flattening",
                "display_name": "Document Flattening",
                "description": "Geometrically flatten the detected document",
                "depends_on": ["document_detection"],
                "category": "correction",
            },
            {
                "name": "perspective_correction",
                "display_name": "Perspective Correction",
                "description": "Correct perspective distortion",
                "depends_on": ["document_flattening"],
                "category": "correction",
            },
            {
                "name": "orientation_correction",
                "display_name": "Orientation Correction",
                "description": "Correct document orientation",
                "depends_on": ["perspective_correction"],
                "category": "correction",
            },
            {
                "name": "noise_elimination",
                "display_name": "Noise Elimination",
                "description": "Remove noise from the image",
                "depends_on": ["orientation_correction"],
                "category": "enhancement",
            },
            {
                "name": "brightness_adjustment",
                "display_name": "Brightness Adjustment",
                "description": "Intelligently adjust image brightness",
                "depends_on": ["noise_elimination"],
                "category": "enhancement",
            },
            {
                "name": "enhancement",
                "display_name": "Image Enhancement",
                "description": "Apply final image enhancements",
                "depends_on": ["brightness_adjustment"],
                "category": "enhancement",
            },
            {
                "name": "final",
                "display_name": "Final Result",
                "description": "Complete preprocessing pipeline result",
                "depends_on": ["enhancement"],
                "category": "output",
            },
        ]

    def can_execute_stage(self, stage_name: str, executed_stages: set) -> bool:
        """
        Check if a stage can be executed based on dependencies.

        Args:
            stage_name: Name of the stage to check
            executed_stages: Set of already executed stage names

        Returns:
            True if stage can be executed
        """
        stages = self.get_pipeline_stages()
        stage_info = next((s for s in stages if s["name"] == stage_name), None)
        if not stage_info:
            return False

        # Check if all dependencies are satisfied
        return all(dep in executed_stages for dep in stage_info["depends_on"])

    def execute_stage(self, stage_name: str, image: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a single pipeline stage.

        Args:
            stage_name: Name of the stage to execute
            image: Input image for the stage
            config: Pipeline configuration

        Returns:
            Dictionary with execution results and metadata
        """
        start_time = time.time()

        try:
            result = {"success": False, "image": image.copy(), "metadata": {}}

            if stage_name == "original":
                result["success"] = True
                result["metadata"]["description"] = "Original input image"

            elif stage_name == "color_preprocessing":
                # Apply color preprocessing
                current_image = image.copy()

                if config.get("convert_to_grayscale", False):
                    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                    current_image = cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR)
                    result["metadata"]["grayscale_applied"] = True

                if config.get("color_inversion", False):
                    current_image = cv2.bitwise_not(current_image)
                    result["metadata"]["color_inverted"] = True

                result["image"] = current_image
                result["success"] = True

            elif stage_name == "document_detection":
                # Use the pipeline's document detection logic
                try:
                    detected = self.pipeline.corner_detector.detect_corners(image)
                    if detected.corners is not None:
                        # Select ordered quadrilateral from detected corners
                        image_height, image_width = image.shape[:2]
                        selected_quad = self.pipeline.corner_selector.select_quadrilateral(detected, (image_height, image_width))
                        if selected_quad is not None:
                            # Draw detected corners and quadrilateral
                            vis_image = image.copy()
                            for corner in detected.corners:
                                cv2.circle(vis_image, tuple(map(int, corner)), 3, (0, 255, 0), -1)
                            quad_corners = selected_quad.corners.astype(int)
                            for i in range(4):
                                cv2.line(vis_image, tuple(quad_corners[i]), tuple(quad_corners[(i + 1) % 4]), (255, 0, 0), 2)

                            result["image"] = vis_image
                            result["metadata"]["corners_detected"] = len(detected.corners)
                            result["metadata"]["quadrilateral_selected"] = True
                            result["metadata"]["confidence"] = float(selected_quad.confidence)
                        else:
                            # Just show detected corners
                            vis_image = image.copy()
                            for corner in detected.corners:
                                cv2.circle(vis_image, tuple(map(int, corner)), 5, (0, 255, 0), -1)
                            result["image"] = vis_image
                            result["metadata"]["corners_detected"] = len(detected.corners)
                            result["metadata"]["quadrilateral_selected"] = False
                    else:
                        result["image"] = image.copy()
                        result["metadata"]["corners_detected"] = 0

                    result["success"] = True
                except Exception as e:
                    result["error"] = f"Document detection failed: {str(e)}"
                    result["image"] = image.copy()

            elif stage_name == "document_flattening":
                # This would need corner information from previous stage
                # For now, return input image with metadata
                result["image"] = image.copy()
                result["success"] = True
                result["metadata"]["note"] = "Document flattening requires corner detection results"

            elif stage_name == "perspective_correction":
                result["image"] = image.copy()
                result["success"] = True
                result["metadata"]["note"] = "Perspective correction requires corner detection results"

            elif stage_name == "orientation_correction":
                result["image"] = image.copy()
                result["success"] = True
                result["metadata"]["note"] = "Orientation correction requires corner detection results"

            elif stage_name == "noise_elimination":
                try:
                    noise_result = self.pipeline.noise_eliminator.eliminate_noise(image)
                    if hasattr(noise_result, "cleaned_image") and noise_result.cleaned_image is not None:
                        result["image"] = noise_result.cleaned_image
                        result["metadata"]["noise_elimination_applied"] = True
                    else:
                        result["image"] = image.copy()
                        result["metadata"]["noise_elimination_applied"] = False
                    result["success"] = True
                except Exception as e:
                    result["error"] = f"Noise elimination failed: {str(e)}"
                    result["image"] = image.copy()

            elif stage_name == "brightness_adjustment":
                try:
                    brightness_result = self.pipeline.brightness_adjuster.adjust_brightness(image)
                    if hasattr(brightness_result, "adjusted_image") and brightness_result.adjusted_image is not None:
                        result["image"] = brightness_result.adjusted_image
                        result["metadata"]["brightness_adjustment_applied"] = True
                    else:
                        result["image"] = image.copy()
                        result["metadata"]["brightness_adjustment_applied"] = False
                    result["success"] = True
                except Exception as e:
                    result["error"] = f"Brightness adjustment failed: {str(e)}"
                    result["image"] = image.copy()

            elif stage_name == "enhancement":
                method = config.get("enhancement_method", "conservative")
                try:
                    enhanced, _ = self.pipeline.image_enhancer.enhance(image, method)
                    result["image"] = enhanced
                    result["metadata"]["enhancement_applied"] = True
                    result["metadata"]["enhancement_method"] = method
                    result["success"] = True
                except Exception as e:
                    result["error"] = f"Enhancement failed: {str(e)}"
                    result["image"] = image.copy()

            elif stage_name == "final":
                result["image"] = image.copy()
                result["success"] = True
                result["metadata"]["description"] = "Final processed result"

            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            result["metadata"]["execution_time_seconds"] = execution_time

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "image": image.copy(),
                "error": str(e),
                "execution_time": execution_time,
                "metadata": {"error": str(e), "execution_time_seconds": execution_time},
            }

    def render_stage_controls(self, stages: list[dict[str, Any]], config: dict[str, Any]) -> None:
        """
        Render the step-by-step stage execution controls.

        Args:
            stages: List of pipeline stages
            config: Pipeline configuration
        """
        st.subheader("🎯 Step-by-Step Pipeline Execution")

        # Progress overview
        self._render_progress_overview(stages)

        # Stage execution controls
        st.markdown("### Stage Controls")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # Stage selector
            stage_options = [f"{stage['display_name']} ({stage['name']})" for stage in stages]
            selected_stage_display = st.selectbox(
                "Select Stage to Execute:",
                stage_options,
                index=st.session_state.pipeline_stage,
                help="Choose which pipeline stage to execute next",
            )

            # Extract stage name from display text
            selected_stage_name = selected_stage_display.split(" (")[1].rstrip(")")

        with col2:
            # Execute button
            executed_stages = st.session_state.pipeline_executed_stages
            can_execute = self.can_execute_stage(selected_stage_name, executed_stages)

            execute_disabled = not can_execute or selected_stage_name in executed_stages or st.session_state.pipeline_current_image is None

            if st.button(
                f"Execute {selected_stage_name}",
                disabled=execute_disabled,
                help="Execute the selected pipeline stage" if can_execute else "Dependencies not satisfied",
            ):
                self._execute_selected_stage(selected_stage_name, config)

        with col3:
            # Reset button
            if st.button("🔄 Reset Pipeline", help="Reset the entire pipeline execution"):
                self.reset_pipeline()
                st.rerun()

        # Dependency status
        if not can_execute and selected_stage_name not in executed_stages:
            stage_info = next((s for s in stages if s["name"] == selected_stage_name), None)
            if stage_info and stage_info["depends_on"]:
                missing_deps = [dep for dep in stage_info["depends_on"] if dep not in executed_stages]
                st.warning(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")

    def _render_progress_overview(self, stages: list[dict[str, Any]]) -> None:
        """Render the pipeline progress overview."""
        st.markdown("### Progress Overview")

        executed_stages = st.session_state.pipeline_executed_stages
        total_stages = len(stages)
        completed_stages = len(executed_stages)

        # Progress bar
        progress = completed_stages / total_stages if total_stages > 0 else 0
        st.progress(progress, text=f"{completed_stages}/{total_stages} stages completed")

        # Stage status grid
        cols = st.columns(min(len(stages), 5))
        for i, stage in enumerate(stages):
            with cols[i % 5]:
                status_icon = (
                    "✅" if stage["name"] in executed_stages else "⏳" if self.can_execute_stage(stage["name"], executed_stages) else "❌"
                )
                st.markdown(f"{status_icon} **{stage['display_name']}**")
                if stage["name"] in st.session_state.pipeline_stage_times:
                    exec_time = st.session_state.pipeline_stage_times[stage["name"]]
                    st.caption(f"{exec_time:.3f}s")

    def _execute_selected_stage(self, stage_name: str, config: dict[str, Any]) -> None:
        """Execute the selected pipeline stage."""
        if st.session_state.pipeline_current_image is None:
            st.error("No image loaded. Please upload an image first.")
            return

        with st.spinner(f"Executing {stage_name}..."):
            result = self.execute_stage(stage_name, st.session_state.pipeline_current_image, config)

            if result["success"]:
                # Update session state
                st.session_state.pipeline_results[stage_name] = result
                st.session_state.pipeline_executed_stages.add(stage_name)
                st.session_state.pipeline_current_image = result["image"]
                st.session_state.pipeline_stage_times[stage_name] = result["execution_time"]

                st.success(f"✅ {stage_name} executed successfully in {result['execution_time']:.3f}s")

                # Auto-advance to next stage
                stages = self.get_pipeline_stages()
                current_idx = next((i for i, s in enumerate(stages) if s["name"] == stage_name), 0)
                if current_idx < len(stages) - 1:
                    st.session_state.pipeline_stage = current_idx + 1
            else:
                error_msg = result.get("error", "Unknown error")
                st.error(f"❌ Failed to execute {stage_name}: {error_msg}")

    def render_stage_results(self) -> None:
        """Render the current stage results and navigation."""
        st.subheader("📊 Current Results")

        if not st.session_state.pipeline_results:
            st.info("No stages executed yet. Select a stage above to begin.")
            return

        # Current image display
        if st.session_state.pipeline_current_image is not None:
            col1, col2 = st.columns([3, 1])

            with col1:
                st.image(
                    cv2.cvtColor(st.session_state.pipeline_current_image, cv2.COLOR_BGR2RGB),
                    caption="Current Pipeline Result",
                    use_column_width=True,
                )

            with col2:
                # Stage navigation
                executed_stages = list(st.session_state.pipeline_executed_stages)
                if executed_stages:
                    st.markdown("### Stage Navigation")
                    selected_result = st.selectbox(
                        "View Stage Result:",
                        executed_stages,
                        index=len(executed_stages) - 1,
                        help="Select a completed stage to view its result",
                    )

                    if selected_result in st.session_state.pipeline_results:
                        result = st.session_state.pipeline_results[selected_result]

                        # Stage metadata
                        st.markdown(f"**Stage:** {selected_result}")
                        if "execution_time" in result:
                            st.markdown(f"⏱️ {result['execution_time']:.3f}s")
                        if result.get("metadata"):
                            with st.expander("Stage Metadata"):
                                st.json(result["metadata"])

                        # Rollback option
                        if selected_result and st.button(
                            f"🔙 Rollback to {selected_result}", help=f"Reset pipeline to {selected_result} stage"
                        ):
                            self._rollback_to_stage(selected_result)
                            st.rerun()

    def _rollback_to_stage(self, stage_name: str) -> None:
        """Rollback pipeline execution to a specific stage."""
        stages = self.get_pipeline_stages()
        target_stage_idx = next((i for i, s in enumerate(stages) if s["name"] == stage_name), -1)

        if target_stage_idx >= 0:
            # Remove all stages after the target stage
            stages_to_remove = []
            for stage in stages[target_stage_idx + 1 :]:
                if stage["name"] in st.session_state.pipeline_executed_stages:
                    stages_to_remove.append(stage["name"])

            for stage in stages_to_remove:
                st.session_state.pipeline_executed_stages.discard(stage)
                st.session_state.pipeline_results.pop(stage, None)
                st.session_state.pipeline_stage_times.pop(stage, None)

            # Reset current image to target stage result
            if stage_name in st.session_state.pipeline_results:
                st.session_state.pipeline_current_image = st.session_state.pipeline_results[stage_name]["image"]

            # Update current stage
            st.session_state.pipeline_stage = target_stage_idx

    def initialize_with_image(self, image: np.ndarray) -> None:
        """
        Initialize the pipeline with an uploaded image.

        Args:
            image: Input image to start the pipeline with
        """
        self.reset_pipeline()
        st.session_state.pipeline_current_image = image.copy()

        # Auto-execute the original stage
        result = self.execute_stage("original", image, {})
        st.session_state.pipeline_results["original"] = result
        st.session_state.pipeline_executed_stages.add("original")
