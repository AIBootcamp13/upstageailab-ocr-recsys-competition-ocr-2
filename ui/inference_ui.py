# inference_ui.py
"""
OCR Inference UI - Streamlit Application

A Streamlit application for real-time OCR inference with uploaded images.
Users can upload images, select trained models, and see OCR predictions instantly.
"""

import logging
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# Add project root to path
try:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
except NameError:
    project_root = Path(".").resolve()

# Import project modules
try:
    # Try to import inference engine, but don't fail if it's not available
    from ui.utils.inference_engine import get_available_checkpoints

    INFERENCE_ENGINE_AVAILABLE = True
except ImportError:
    INFERENCE_ENGINE_AVAILABLE = False
    # This st.warning will be called in the main app logic to be visible to the user


def display_image_with_ocr_predictions(image: np.ndarray, predictions: dict, title: str = "OCR Predictions"):
    """Display image with OCR predictions overlaid."""
    try:
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image, "RGBA")

        polygons_str = predictions.get("polygons", "")
        if polygons_str:
            polygons = polygons_str.split("|")
            texts = predictions.get("texts", [])
            confidences = predictions.get("confidences", [])

            for i, polygon_str in enumerate(polygons):
                coords = [int(x) for x in polygon_str.split(",")]
                if len(coords) >= 8 and len(coords) % 2 == 0:
                    points = [(coords[j], coords[j + 1]) for j in range(0, len(coords), 2)]

                    # Draw polygon
                    draw.polygon(points, outline=(255, 0, 0, 255), fill=(255, 0, 0, 30))

                    # Prepare and draw text label with confidence
                    if points:
                        min_x = min(p[0] for p in points)
                        min_y = min(p[1] for p in points)

                        text = texts[i] if i < len(texts) else f"Det_{i + 1}"
                        conf = confidences[i] if i < len(confidences) else 0
                        label = f"{text} ({conf:.1%})"

                        # Position text above the box
                        text_pos = (min_x, min_y - 20)

                        # Draw a background rectangle for the text
                        try:
                            bbox = draw.textbbox(text_pos, label)
                            draw.rectangle(bbox, fill=(255, 0, 0, 180))
                            draw.text(text_pos, label, fill=(255, 255, 255, 255))
                        except AttributeError:  # Fallback for older PIL versions
                            draw.text(text_pos, label, fill=(255, 0, 0, 255))

        st.image(pil_image, caption=title, use_container_width=True)
    except Exception as e:
        logging.error(f"Failed to draw predictions: {e}", exc_info=True)
        st.error("Could not render predictions on the image.")
        st.image(image, caption="Original Image (render failed)", use_container_width=True)


def main():
    """Main Streamlit application for OCR inference."""
    st.set_page_config(
        page_title="OCR Inference",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔍 Real-time OCR Inference")
    st.markdown("Upload images and get instant OCR predictions.")

    # Initialize session state
    if "inference_results" not in st.session_state:
        st.session_state.inference_results = []
    if "selected_images" not in st.session_state:
        st.session_state.selected_images = set()
    if "processed_images" not in st.session_state:
        st.session_state.processed_images = set()

    # Create two columns for layout
    col1, col2 = st.columns([1, 2])

    with col1:
        render_inference_controls()

    with col2:
        render_results_display()


def render_inference_controls():
    """Render the inference controls in the sidebar."""
    st.header("⚙️ Inference Controls")

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    # Model Selection
    st.subheader("Model Selection")
    available_models = get_available_models()
    selected_model = st.selectbox(
        "Select Trained Model",
        available_models,
        help="Choose a trained OCR model for inference.",
    )

    # Reset results if model changes
    if st.session_state.selected_model != selected_model:
        st.session_state.inference_results = []
        st.session_state.selected_model = selected_model
        st.rerun()

    if not INFERENCE_ENGINE_AVAILABLE:
        st.warning("⚠️ **Inference Engine Not Available**: Using mock predictions for demonstration.")
    elif "No trained models" in selected_model:
        st.warning("⚠️ **No Models Found**: Please train a model first or use demo mode.")
    else:
        st.success("✅ **Real Inference Available**: Using trained OCR model for predictions.")

    # Image Upload
    st.subheader("Image Upload")
    uploaded_files = st.file_uploader(
        "Upload Images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        help="Upload one or more images for OCR inference",
    )

    # Handle image selection
    if uploaded_files:
        # Get current uploaded filenames
        current_filenames = {file.name for file in uploaded_files}

        # If new files were uploaded, deselect old images and select new ones
        if "previous_uploaded_files" not in st.session_state or st.session_state.previous_uploaded_files != current_filenames:
            # Deselect old images
            st.session_state.selected_images = set()
            # Select all new images by default
            st.session_state.selected_images.update(current_filenames)
            st.session_state.previous_uploaded_files = current_filenames

        # Image Selection Checkboxes
        st.subheader("Select Images for Inference")
        st.markdown("Choose which images to run inference on:")

        # Create checkboxes for each uploaded file
        selected_files = []
        for file in uploaded_files:
            is_selected = file.name in st.session_state.selected_images
            if st.checkbox(f"📄 {file.name}", value=is_selected, key=f"select_{file.name}"):
                selected_files.append(file)
                st.session_state.selected_images.add(file.name)
            else:
                st.session_state.selected_images.discard(file.name)

        # Update selected files list
        selected_files = [file for file in uploaded_files if file.name in st.session_state.selected_images]

        # Show selection summary
        if selected_files:
            st.success(f"✅ {len(selected_files)} of {len(uploaded_files)} images selected for inference")
        else:
            st.warning("⚠️ No images selected for inference")

        # Inference Button
        if selected_files:
            if st.button("🚀 Run Inference", type="primary", use_container_width=True):
                run_inference(selected_files, selected_model)
        else:
            st.info("📤 Select images above to run inference.")
    else:
        st.info("📤 Upload one or more images to get started.")

    # Clear Results Button
    if st.session_state.inference_results:
        st.divider()
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.inference_results = []
            st.session_state.processed_images = set()
            st.rerun()


def render_results_display():
    """Render the main inference results display area."""
    st.header("📊 Inference Results")

    if not st.session_state.inference_results:
        st.info("No inference results yet. Upload images and run inference to see results.")
        return

    # --- Calculate and Display Overall Summary ---
    successful_results = [r for r in st.session_state.inference_results if r.get("success", False)]
    total_images = len(st.session_state.inference_results)
    successful = len(successful_results)
    failed = total_images - successful

    all_confidences = [conf for r in successful_results for conf in r.get("predictions", {}).get("confidences", [])]
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", total_images)
    with col2:
        st.metric("Successful", successful)
    with col3:
        st.metric("Failed", failed)
    with col4:
        st.metric("Avg. Confidence", f"{avg_confidence:.2%}")

    st.divider()

    # --- Display Individual Image Results ---
    for i, result in enumerate(st.session_state.inference_results):
        expander_title = f"Image {i + 1}: {result.get('filename', '')}"
        with st.expander(expander_title, expanded=(i == 0)):
            display_single_result(result)


def display_single_result(result: dict):
    """Display a single inference result inside an expander."""
    if not result.get("success", False):
        st.error(f"❌ Inference failed: {result.get('error', 'Unknown error')}")
        return

    # --- Calculate and Display Per-Image Stats ---
    predictions = result.get("predictions", {})
    confidences = predictions.get("confidences", [])
    num_detections = len(confidences)
    avg_confidence = sum(confidences) / num_detections if num_detections > 0 else 0

    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.metric("Detections", f"{num_detections}")
    with stat_col2:
        st.metric("Avg. Confidence", f"{avg_confidence:.2%}")

    # --- Display Image with Prediction Overlays ---
    if "image" in result and "predictions" in result:
        display_image_with_ocr_predictions(
            result["image"],
            result["predictions"],
            title=f"OCR Predictions for {result.get('filename', '')}",
        )

    # --- Display Raw Data in a Collapsible Section ---
    if "predictions" in result:
        with st.expander("🔧 Raw Prediction Data"):
            st.json(result["predictions"])


def get_available_models() -> list[str]:
    """Get a list of available trained models."""
    if INFERENCE_ENGINE_AVAILABLE:
        try:
            checkpoints = get_available_checkpoints()
            if checkpoints and "No checkpoints found" not in checkpoints[0]:
                return checkpoints
        except Exception as e:
            logging.error(f"Could not get checkpoints: {e}")

    # Fallback list
    return ["No trained models found - using Demo Mode"]


def run_inference(uploaded_files: list, model_path: str):
    """Run inference on selected uploaded files, skipping already processed images."""
    total_files = len(uploaded_files)
    new_results = []

    progress_bar = st.progress(0, text=f"Starting inference for {total_files} images...")

    for i, uploaded_file in enumerate(uploaded_files):
        filename = uploaded_file.name

        # Check if this image has already been processed
        if filename in st.session_state.processed_images:
            st.info(f"⏭️ Skipping {filename} - already processed")
            progress_bar.progress(
                (i + 1) / total_files,
                text=f"Skipped {filename} (already processed)... ({i + 1}/{total_files})",
            )
            continue

        progress_bar.progress((i) / total_files, text=f"Processing {filename}... ({i + 1}/{total_files})")

        # Save uploaded file to a temporary path to be read by OpenCV
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        try:
            result = perform_inference(temp_path, model_path, filename)
            new_results.append(result)
            # Mark this image as processed
            st.session_state.processed_images.add(filename)
        finally:
            # Clean up the temporary file
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    # Add new results to existing results
    st.session_state.inference_results.extend(new_results)

    progress_bar.progress(1.0, text=f"✅ Inference complete! Processed {len(new_results)} new images.")
    time.sleep(1)
    progress_bar.empty()


def perform_inference(image_path: str, model_path: str, filename: str) -> dict:
    """Perform OCR inference on a single image and return a results dictionary."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictions = None
        if INFERENCE_ENGINE_AVAILABLE and "Demo Mode" not in model_path:
            try:
                from ui.utils.inference_engine import run_inference_on_image

                predictions = run_inference_on_image(image_path, model_path)
                if predictions is None:
                    raise ValueError("Inference engine returned no results.")
            except Exception as e:
                st.warning(f"⚠️ Real inference failed ({e}), using mock predictions as a fallback.")
                predictions = None  # Ensure fallback is triggered

        if predictions is None:
            predictions = generate_mock_predictions(image.shape)

        return {
            "filename": filename,
            "success": True,
            "image": image_rgb,
            "predictions": predictions,
        }
    except Exception as e:
        return {"filename": filename, "success": False, "error": str(e)}


def generate_mock_predictions(image_shape: tuple) -> dict:
    """Generate mock predictions for demonstration purposes."""
    height, width, _ = image_shape
    # Create dynamic mock boxes based on image size
    box1 = [int(width * 0.1), int(height * 0.1), int(width * 0.4), int(height * 0.2)]
    box2 = [int(width * 0.5), int(height * 0.4), int(width * 0.9), int(height * 0.5)]
    box3 = [int(width * 0.2), int(height * 0.7), int(width * 0.7), int(height * 0.8)]
    mock_boxes = [box1, box2, box3]

    return {
        "polygons": "|".join([f"{b[0]},{b[1]},{b[2]},{b[1]},{b[2]},{b[3]},{b[0]},{b[3]}" for b in mock_boxes]),
        "texts": ["Sample Text 1", "Another Example", "Third Line"],
        "confidences": [0.95, 0.87, 0.92],
    }


if __name__ == "__main__":
    main()
