from pydantic import ValidationError

"""Service layer orchestrating inference requests.

Before modifying behaviour, consult the Streamlit protocols in
``docs/ai_handbook/02_protocols/`` and the UI configuration in
``configs/ui/inference.yaml``. Hyperparameter defaults originate from that YAML
and related schemas; keep them authoritative instead of adding ad-hoc logic
here.
"""

# AI_DOCS[
#   bundle: streamlit-maintenance
#   priority: high
#   path: docs/ai_handbook/02_protocols/11_streamlit_maintenance_protocol.md#4-maintenance-checklist
#   path: docs/ai_handbook/02_protocols/05_modular_refactor.md#phase-4-finalise--document
#   path: docs/ai_handbook/02_protocols/08_context_checkpointing.md#how-to-create-a-checkpoint
# ]

import logging
import re
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import streamlit as st

from ..models.batch_request import BatchPredictionRequest
from ..models.config import PreprocessingConfig
from ..models.data_contracts import InferenceResult, Predictions, PreprocessingInfo
from ..models.ui_events import InferenceRequest
from ..state import InferenceState
from .submission_writer import SubmissionWriter

LOGGER = logging.getLogger(__name__)

_POLYGON_TOKEN_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")

try:
    from ui.utils.inference import run_inference_on_image
except ImportError:  # pragma: no cover - mocked in dev environments
    run_inference_on_image = None  # type: ignore[assignment]

ENGINE_AVAILABLE = run_inference_on_image is not None

try:
    from ocr.datasets.preprocessing import (
        DOCTR_AVAILABLE,
        DocumentPreprocessor,
    )
except ImportError:  # pragma: no cover - optional dependency guard
    DOCTR_AVAILABLE = False
    DocumentPreprocessor = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from ocr.datasets.preprocessing import (
        DocumentPreprocessor as DocumentPreprocessorType,
    )
else:  # pragma: no cover - runtime fallback for optional dependency
    DocumentPreprocessorType = Any


class InferenceService:
    def run(self, state: InferenceState, request: InferenceRequest, hyperparams: dict[str, float]) -> None:
        # Validate request data contract
        try:
            validated_request = InferenceRequest.model_validate(request)
        except ValidationError as exc:
            LOGGER.error("Invalid inference request: %s", exc)
            st.error(f"❌ Invalid request data: {exc}")
            return

        mode_key = "docTR:on" if validated_request.use_preprocessing else "docTR:off"
        state.ensure_processed_bucket(validated_request.model_path, mode_key)
        total_files = len(validated_request.files)
        new_results: list[InferenceResult] = []

        progress = st.progress(0.0, text=f"Starting inference for {total_files} images...")

        preprocessor = None
        if validated_request.use_preprocessing and DocumentPreprocessor is not None:
            preprocessor = self._build_preprocessor(validated_request.preprocessing_config)

        for index, uploaded_file in enumerate(validated_request.files):
            filename = uploaded_file.name
            processed_bucket = state.processed_images[validated_request.model_path][mode_key]
            if filename in processed_bucket:
                progress.progress(
                    (index + 1) / total_files,
                    text=f"Skipped {filename} (already processed)... ({index + 1}/{total_files})",
                )
                continue

            progress.progress(index / total_files, text=f"Processing {filename}... ({index + 1}/{total_files})")

            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = Path(tmp_file.name)

            try:
                result = self._perform_inference(
                    temp_path,
                    Path(validated_request.model_path),
                    validated_request.config_path,
                    filename,
                    hyperparams,
                    validated_request.use_preprocessing,
                    preprocessor,
                )
                new_results.append(result)
                processed_bucket.add(filename)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

        state.inference_results.extend(new_results)
        state.persist()

        progress.progress(1.0, text=f"✅ Inference complete! Processed {len(new_results)} new images.")
        time.sleep(1)
        progress.empty()

    def _perform_inference(
        self,
        image_path: Path,
        model_path: Path,
        config_path: str | None,
        filename: str,
        hyperparams: dict[str, float],
        use_preprocessing: bool,
        preprocessor: DocumentPreprocessorType | None,
    ) -> InferenceResult:
        processed_temp_path: Path | None = None
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            inference_rgb = image_rgb
            preprocessing_info = PreprocessingInfo(
                enabled=use_preprocessing,
                metadata=None,
                original=image_rgb,
                processed=None,
                doctr_available=DOCTR_AVAILABLE,
                mode="docTR:on" if use_preprocessing else "docTR:off",
            )
            inference_target_path = image_path

            if use_preprocessing and preprocessor is not None:
                try:
                    processed = preprocessor(image_rgb.copy())
                    processed_image = processed.get("image")
                    if processed_image is None:
                        raise ValueError("DocumentPreprocessor returned no image result.")
                    inference_rgb = np.asarray(processed_image)
                    metadata = processed.get("metadata")
                    preprocessing_info.metadata = metadata if isinstance(metadata, dict) else None
                    preprocessing_info.processed = inference_rgb
                    inference_target_path = self._write_temp_image(inference_rgb, suffix=image_path.suffix)
                    processed_temp_path = inference_target_path
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("docTR preprocessing failed for %s: %s", filename, exc)
                    preprocessing_info.error = str(exc)
                    preprocessing_info.enabled = False
                    inference_rgb = image_rgb
                    inference_target_path = image_path

            predictions = None
            if ENGINE_AVAILABLE:
                inference_fn = run_inference_on_image
                assert inference_fn is not None  # Guaranteed by ENGINE_AVAILABLE
                try:
                    max_candidates_val = hyperparams.get("max_candidates")
                    min_detection_size_val = hyperparams.get("min_detection_size")
                    raw_predictions = inference_fn(
                        str(inference_target_path),
                        str(model_path),
                        config_path,
                        hyperparams.get("binarization_thresh"),
                        hyperparams.get("box_thresh"),
                        int(max_candidates_val) if max_candidates_val is not None else None,
                        int(min_detection_size_val) if min_detection_size_val is not None else None,
                    )
                    if raw_predictions is None:
                        raise ValueError("Inference engine returned no results.")
                    LOGGER.info(f"Inference engine returned predictions: {raw_predictions}")
                    if not self._are_predictions_valid(raw_predictions, inference_rgb.shape):
                        LOGGER.warning(f"Predictions failed validation: {raw_predictions}")
                        raise ValueError("Inference engine returned invalid predictions.")
                    predictions = Predictions(**raw_predictions)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Real inference failed; using mock predictions fallback: %s", exc)
                    st.error(f"❌ Inference failed: {exc}")
                    predictions = None

            if predictions is None:
                predictions = self._generate_mock_predictions(inference_rgb.shape)

            assert predictions is not None  # Should always be set by now

            # Create and validate the result data contract
            result = InferenceResult(
                filename=filename,
                success=True,
                image=inference_rgb,
                predictions=predictions,
                preprocessing=preprocessing_info,
            )
            try:
                validated_result = InferenceResult.model_validate(result)
                return validated_result
            except ValidationError as exc:
                LOGGER.error("Invalid inference result for %s: %s", filename, exc)
                return InferenceResult(
                    filename=filename,
                    success=False,
                    image=inference_rgb,
                    predictions=Predictions(polygons="", texts=[], confidences=[]),
                    preprocessing=preprocessing_info,
                    error=f"Result validation failed: {exc}",
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Inference failed for %s", filename)
            return InferenceResult(
                filename=filename,
                success=False,
                image=np.zeros((1, 1, 3), dtype=np.uint8),  # Placeholder image for error case
                predictions=Predictions(polygons="", texts=[], confidences=[]),  # Empty predictions
                preprocessing=PreprocessingInfo(enabled=False),  # Default preprocessing
                error=str(exc),
            )
        finally:
            if processed_temp_path and processed_temp_path.exists():
                processed_temp_path.unlink()

    @staticmethod
    def _generate_mock_predictions(image_shape: Sequence[int]) -> Predictions:
        height, width, _ = image_shape
        box1 = [int(width * 0.1), int(height * 0.1), int(width * 0.4), int(height * 0.2)]
        box2 = [int(width * 0.5), int(height * 0.4), int(width * 0.9), int(height * 0.5)]
        box3 = [int(width * 0.2), int(height * 0.3), int(width * 0.7), int(height * 0.4)]  # Moved up from bottom
        mock_boxes = [box1, box2, box3]

        return Predictions(
            # Competition format uses space-separated coordinates, not commas
            polygons="|".join(f"{b[0]} {b[1]} {b[2]} {b[1]} {b[2]} {b[3]} {b[0]} {b[3]}" for b in mock_boxes),
            texts=["Sample Text 1", "Another Example", "Third Line"],
            confidences=[0.95, 0.87, 0.92],
        )

    @staticmethod
    def _write_temp_image(image_rgb: np.ndarray, suffix: str) -> Path:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(tmp_file.name, image_bgr):
                raise ValueError("Failed to serialize preprocessed image for inference.")
            return Path(tmp_file.name)

    @staticmethod
    def _build_preprocessor(config: PreprocessingConfig | None) -> DocumentPreprocessorType | None:
        if DocumentPreprocessor is None:
            return None

        if config is None:
            return DocumentPreprocessor()

        kwargs = config.to_kwargs()
        return DocumentPreprocessor(**kwargs)

    def run_batch_prediction(
        self,
        state: InferenceState,
        request: BatchPredictionRequest,
    ) -> list[InferenceResult]:
        """Execute batch predictions on a directory of images.

        Args:
            state: Current inference state for tracking processed images
            request: Validated batch prediction request with all parameters

        Returns:
            List of InferenceResult objects for all processed images

        Raises:
            ValidationError: If request validation fails
        """
        # Validate request data contract
        try:
            validated_request = BatchPredictionRequest.model_validate(request)
        except ValidationError as exc:
            LOGGER.error("Invalid batch prediction request: %s", exc)
            st.error(f"❌ Invalid batch request data: {exc}")
            return []

        # Get list of image files to process
        try:
            image_files = validated_request.get_image_files()
        except ValueError as exc:
            LOGGER.error("Failed to scan input directory: %s", exc)
            st.error(f"❌ {exc}")
            return []

        total_files = len(image_files)
        LOGGER.info(f"Found {total_files} images to process in {validated_request.input_dir}")

        # Set up progress tracking
        progress = st.progress(0.0, text=f"Starting batch prediction for {total_files} images...")
        st.info(f"📁 Processing directory: {validated_request.input_dir}")

        # Determine preprocessing mode
        mode_key = "docTR:on" if validated_request.use_preprocessing else "docTR:off"
        state.ensure_processed_bucket(validated_request.model_path, mode_key)

        # Initialize preprocessor if needed
        preprocessor = None
        if validated_request.use_preprocessing and DocumentPreprocessor is not None:
            preprocessor = self._build_preprocessor(None)  # Use default config for batch

        # Process each image
        batch_results: list[InferenceResult] = []
        hyperparams = validated_request.hyperparameters.to_dict()

        for index, image_path in enumerate(image_files):
            filename = image_path.name
            progress_pct = index / total_files
            progress.progress(
                progress_pct,
                text=f"Processing {filename}... ({index + 1}/{total_files})",
            )

            try:
                result = self._perform_inference(
                    image_path,
                    Path(validated_request.model_path),
                    validated_request.config_path,
                    filename,
                    hyperparams,
                    validated_request.use_preprocessing,
                    preprocessor,
                )
                batch_results.append(result)

                if result.success:
                    LOGGER.info(f"✓ Successfully processed {filename}")
                else:
                    LOGGER.warning(f"✗ Failed to process {filename}: {result.error}")
                    st.warning(f"⚠️ Failed to process {filename}: {result.error}")

            except Exception as exc:  # noqa: BLE001
                LOGGER.exception(f"Unexpected error processing {filename}")
                st.error(f"❌ Error processing {filename}: {exc}")
                # Create error result
                error_result = InferenceResult(
                    filename=filename,
                    success=False,
                    image=np.zeros((1, 1, 3), dtype=np.uint8),
                    predictions=Predictions(polygons="", texts=[], confidences=[]),
                    preprocessing=PreprocessingInfo(enabled=False),
                    error=str(exc),
                )
                batch_results.append(error_result)

        # Complete progress
        progress.progress(1.0, text=f"✅ Batch processing complete! Processed {total_files} images.")
        time.sleep(1)
        progress.empty()

        # Report statistics
        successful = sum(1 for r in batch_results if r.success)
        failed = total_files - successful
        st.success(f"✅ Batch prediction complete: {successful} successful, {failed} failed")

        # Write submission files
        written_files = {}
        if batch_results:
            try:
                json_path = validated_request.get_output_path(".json") if validated_request.output_config.save_json else None
                csv_path = validated_request.get_output_path(".csv") if validated_request.output_config.save_csv else None

                written_files = SubmissionWriter.write_batch_results(
                    batch_results,
                    json_path=json_path,
                    csv_path=csv_path,
                    include_confidence=validated_request.output_config.include_confidence,
                )

                # Store output files in state for download buttons
                state.batch_output_files = {format_name: str(file_path) for format_name, file_path in written_files.items()}
                state.persist()

                # Display output file paths
                for format_name, file_path in written_files.items():
                    st.success(f"📄 {format_name.upper()} output: {file_path}")
                    LOGGER.info(f"Wrote {format_name.upper()} submission to {file_path}")

                # Display summary statistics
                stats = SubmissionWriter.generate_summary_stats(batch_results)
                st.info(
                    f"📊 **Summary Statistics:**\n"
                    f"- Total Images: {stats['total_images']}\n"
                    f"- Successful: {stats['successful']}\n"
                    f"- Failed: {stats['failed']}\n"
                    f"- Success Rate: {stats['success_rate']}\n"
                    f"- Total Polygons Detected: {stats['total_polygons_detected']}\n"
                    f"- Avg Polygons/Image: {stats['avg_polygons_per_image']}"
                )

            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to write submission files")
                st.error(f"❌ Failed to write submission files: {exc}")

        return batch_results

    @staticmethod
    def _are_predictions_valid(predictions: dict[str, Any], image_shape: tuple[int, ...]) -> bool:
        """Validate prediction format and bounds."""
        if not isinstance(predictions, dict):
            LOGGER.warning(f"Predictions is not a dict: {type(predictions)}")
            return False

        polygons_text = predictions.get("polygons", "")
        if polygons_text is None:
            LOGGER.warning(f"Predictions missing polygons field. Keys: {list(predictions.keys())}")
            return False

        if not isinstance(polygons_text, str):
            LOGGER.warning(f"Polygons field is not a string: {type(polygons_text)}")
            return False

        height, width = image_shape[:2]
        LOGGER.debug(f"Validating predictions for image {width}x{height}, polygons length: {len(polygons_text)}")

        if not polygons_text.strip():
            LOGGER.debug("Polygons string is empty - this is valid (no text detected)")
            return True

        for polygon_str in polygons_text.split("|"):
            if not polygon_str.strip():
                continue  # Skip empty polygons

            tokens = _POLYGON_TOKEN_PATTERN.findall(polygon_str)
            if len(tokens) < 8:
                LOGGER.warning(f"Polygon has too few coordinates: {len(tokens)} < 8 in '{polygon_str}'")
                return False

            if len(tokens) % 2 != 0:
                LOGGER.warning(f"Polygon has odd number of coordinates: {len(tokens)} in '{polygon_str}'")
                return False

            try:
                coords = [float(token) for token in tokens]
            except (ValueError, TypeError) as exc:  # pragma: no cover - guarded by regex
                LOGGER.warning(f"Invalid coordinate value in polygon '{polygon_str}': {exc}")
                return False

            for i in range(0, len(coords), 2):
                x, y = coords[i], coords[i + 1]
                # Allow polygons to extend reasonably outside image bounds
                if x < -width or x > 2 * width or y < -height or y > 2 * height:
                    LOGGER.warning(f"Polygon coordinate out of bounds: ({x}, {y}) for image {width}x{height}")
                    return False

        return True
