# inference_engine.py
"""
Inference utilities for OCR UI applications.

This module provides functions to run OCR inference on images using trained models.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Add project root to path
try:
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
except NameError:
    # Handle cases where __file__ is not defined (e.g., in some interactive environments)
    project_root = Path(".").resolve()

# Lazy import for OCR modules to avoid errors if not installed
try:
    import lightning.pytorch as pl
    import torch
    import torchvision.transforms as transforms
    import yaml
    from omegaconf import DictConfig, ListConfig

    from ocr.models import get_model_by_cfg

    OCR_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import OCR modules: {e}. InferenceEngine will use mock predictions.")
    OCR_MODULES_AVAILABLE = False
    # Define dummy classes to prevent NameErrors later
    torch = None
    DictConfig = dict
    ListConfig = None
    pl = None
    yaml = None
    transforms = None


class InferenceEngine:
    """OCR Inference Engine for real-time predictions."""

    def __init__(self):
        self.model = None
        self.trainer = None
        self.config = None
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

        # Configurable preprocessing parameters
        self.image_size = (640, 640)  # Default fallback
        self.normalize_mean = [0.485, 0.456, 0.406]  # Default ImageNet mean
        self.normalize_std = [0.229, 0.224, 0.225]  # Default ImageNet std

        # Configurable postprocessing parameters
        self.binarization_thresh = 0.3  # Default threshold
        self.box_thresh = 0.4  # Default box threshold
        self.max_candidates = 300  # Default max candidates
        self.min_detection_size = 5  # Default minimum detection size

        logging.info(f"Using device: {self.device}")

    def load_model(self, checkpoint_path: str, config_path: str | None = None) -> bool:
        """
        Load a trained OCR model from checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint
            config_path: Path to the config file (optional, will try to infer)

        Returns:
            bool: True if model loaded successfully
        """
        if not OCR_MODULES_AVAILABLE:
            logging.error("OCR modules are not installed. Cannot load a real model.")
            return False

        try:
            return self._extracted_from_load_model_18(config_path, checkpoint_path)
        except Exception as e:
            logging.error(f"Error loading model: {e}", exc_info=True)
            return False

    # TODO Rename this here and in `load_model`
    def _extracted_from_load_model_18(self, config_path, checkpoint_path):
        # If no config path provided, try to find it
        if config_path is None:
            config_path = self._find_config_for_checkpoint(checkpoint_path)

        if not config_path:
            logging.error(f"Could not find a valid config file for checkpoint: {checkpoint_path}")
            return False

        logging.info(f"Loading model from checkpoint: {checkpoint_path}")
        logging.info(f"Using config file: {config_path}")

        # Load config
        with open(config_path) as f:
            config_dict = yaml.safe_load(f) if config_path.endswith((".yaml", ".yml")) else json.load(f)
        self.config = DictConfig(config_dict) if OCR_MODULES_AVAILABLE else config_dict
        # Extract preprocessing parameters from config
        self._extract_config_parameters()

        # Initialize model
        logging.info(f"Instantiating model with architecture: {self.config.model.get('architecture_name', 'custom')}")
        self.model = get_model_by_cfg(self.config.model)
        logging.info(f"Model instantiated with {sum(p.numel() for p in self.model.parameters())} parameters")

        # Load checkpoint
        checkpoint = self._load_checkpoint(checkpoint_path)
        if checkpoint is None:
            logging.error("Failed to load checkpoint %s", checkpoint_path)
            return False

        # Find the state dictionary in the checkpoint file
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            # Add fallbacks for other common key names
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))

        # Filter out unnecessary keys (e.g., from the optimizer) and remove 'model.' prefix
        model_state_dict = self.model.state_dict()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_key = k[len("model.") :]
                if new_key in model_state_dict:
                    filtered_state_dict[new_key] = v

        dropped_keys = {k for k in state_dict if k.startswith("model.")} - {f"model.{k}" for k in filtered_state_dict}
        missing_keys = set(model_state_dict) - set(filtered_state_dict)

        if dropped_keys:
            logging.warning(f"The following keys from the loaded state_dict were dropped and not loaded into the model: {dropped_keys}")
        if missing_keys:
            logging.warning(f"The following keys expected by the model are missing from the loaded state_dict: {missing_keys}")

        try:
            self.model.load_state_dict(filtered_state_dict, strict=False)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                logging.error(
                    f"Model architecture mismatch: The checkpoint was trained with a different model configuration. "
                    f"Checkpoint: {checkpoint_path}, Config: {config_path}"
                )
                logging.error(f"Size mismatch details: {e}")
                return False
            else:
                raise

        self.model.to(self.device)
        self.model.eval()

        logging.info("Model loaded successfully.")
        return True

    def _load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any] | None:
        if torch is None:
            return None

        self._register_safe_globals()

        try:
            return torch.load(checkpoint_path, map_location=self.device)
        except TypeError:
            pass
        except Exception as exc:  # noqa: BLE001
            logging.debug("Initial torch.load failed for %s: %s", checkpoint_path, exc)

        try:
            return torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as exc:  # noqa: BLE001
            logging.error("Unable to load checkpoint %s: %s", checkpoint_path, exc)
            return None

    @staticmethod
    def _register_safe_globals() -> None:
        if torch is None or ListConfig is None:
            return

        try:
            from torch.serialization import add_safe_globals
        except (ImportError, AttributeError):
            return

        try:
            add_safe_globals([ListConfig])
        except Exception as exc:  # noqa: BLE001
            logging.debug("Could not register ListConfig as a safe global: %s", exc)

    def _extract_config_parameters(self):
        """Extract preprocessing and postprocessing parameters from config."""
        try:
            # Extract preprocessing parameters
            if hasattr(self.config, "preprocessing") and self.config.preprocessing:
                preprocessing = self.config.preprocessing
                if hasattr(preprocessing, "target_size") and preprocessing.target_size:
                    self.image_size = tuple(preprocessing.target_size)
                    logging.info(f"Using configured image size: {self.image_size}")

            # Try to extract from transforms if preprocessing section not available
            elif hasattr(self.config, "transforms") and self.config.transforms:
                transforms_config = self.config.transforms
                # Look for predict_transform or test_transform
                transform_key = "predict_transform"
                if not hasattr(transforms_config, transform_key):
                    transform_key = "test_transform"

                if hasattr(transforms_config, transform_key):
                    transform_config = getattr(transforms_config, transform_key)
                    if hasattr(transform_config, "transforms"):
                        for t in transform_config.transforms:
                            if hasattr(t, "max_size"):
                                # Found LongestMaxSize transform
                                self.image_size = (t.max_size, t.max_size)
                                logging.info(f"Using transform image size: {self.image_size}")
                                break
                            elif hasattr(t, "min_width") and hasattr(t, "min_height"):
                                # Found PadIfNeeded transform
                                self.image_size = (t.min_width, t.min_height)
                                logging.info(f"Using pad transform image size: {self.image_size}")
                                break

                    # Extract normalization parameters
                    for t in transform_config.transforms:
                        if hasattr(t, "mean") and hasattr(t, "std"):
                            self.normalize_mean = list(t.mean)
                            self.normalize_std = list(t.std)
                            logging.info(f"Using configured normalization: mean={self.normalize_mean}, std={self.normalize_std}")
                            break

            # Extract postprocessing parameters from head config
            if hasattr(self.config, "model") and hasattr(self.config.model, "head"):
                head_config = self.config.model.head
                if hasattr(head_config, "postprocess"):
                    postprocess = head_config.postprocess
                    if hasattr(postprocess, "thresh"):
                        self.binarization_thresh = float(postprocess.thresh)
                        logging.info(f"Using configured binarization threshold: {self.binarization_thresh}")
                    if hasattr(postprocess, "box_thresh"):
                        self.box_thresh = float(postprocess.box_thresh)
                        logging.info(f"Using configured box threshold: {self.box_thresh}")
                    if hasattr(postprocess, "max_candidates"):
                        self.max_candidates = int(postprocess.max_candidates)
                        logging.info(f"Using configured max candidates: {self.max_candidates}")

            # Note: min_detection_size is not in config yet, keeping default

        except Exception as e:
            logging.warning(f"Could not extract config parameters, using defaults: {e}")

    def update_postprocessor_params(
        self,
        binarization_thresh: float | None = None,
        box_thresh: float | None = None,
        max_candidates: int | None = None,
        min_detection_size: int | None = None,
    ):
        """Update postprocessor parameters dynamically."""
        if self.model is None:
            logging.warning("Model not loaded, cannot update postprocessor parameters.")
            return

        # Try to update the model's postprocessor if it exists
        if hasattr(self.model, "head") and hasattr(self.model.head, "postprocess"):
            postprocess = self.model.head.postprocess
            if binarization_thresh is not None and hasattr(postprocess, "thresh"):
                postprocess.thresh = binarization_thresh
                logging.info(f"Updated binarization threshold to: {binarization_thresh}")
            if box_thresh is not None and hasattr(postprocess, "box_thresh"):
                postprocess.box_thresh = box_thresh
                logging.info(f"Updated box threshold to: {box_thresh}")
            if max_candidates is not None and hasattr(postprocess, "max_candidates"):
                postprocess.max_candidates = max_candidates
                logging.info(f"Updated max candidates to: {max_candidates}")
            if min_detection_size is not None and hasattr(postprocess, "min_size"):
                postprocess.min_size = min_detection_size
                logging.info(f"Updated min detection size to: {min_detection_size}")
        else:
            # Fallback to updating our own parameters
            if binarization_thresh is not None:
                self.binarization_thresh = binarization_thresh
            if box_thresh is not None:
                self.box_thresh = box_thresh
            if max_candidates is not None:
                self.max_candidates = max_candidates
            if min_detection_size is not None:
                self.min_detection_size = min_detection_size
            logging.info("Updated inference engine parameters (model postprocessor not found)")

    def predict_image(
        self,
        image_path: str,
        binarization_thresh: float | None = None,
        box_thresh: float | None = None,
        max_candidates: int | None = None,
        min_detection_size: int | None = None,
    ) -> dict[str, Any] | None:
        """
        Run inference on a single image.

        Args:
            image_path: Path to the image file

        Returns:
            dict: Prediction results or None if failed
        """
        if self.model is None:
            logging.warning("Model not loaded. Returning mock predictions.")
            return self._get_mock_predictions()

        # Update postprocessor parameters if provided
        if binarization_thresh is not None or box_thresh is not None or max_candidates is not None or min_detection_size is not None:
            self.update_postprocessor_params(binarization_thresh, box_thresh, max_candidates, min_detection_size)

        try:
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Failed to read image at path: {image_path}")
                return None

            # Preprocess image
            processed_image = self._preprocess_image(image)

            # Run inference
            with torch.no_grad():
                batch_input = {"images": processed_image.to(self.device)}
                predictions = self.model(return_loss=False, **batch_input)

            # Try to use model's built-in postprocessor if available
            if hasattr(self.model, "head") and hasattr(self.model.head, "get_polygons_from_maps"):
                # Create batch dict for postprocessor
                inverse_matrix = self._compute_inverse_matrix(processed_image, image.shape)
                batch = {
                    "images": processed_image,
                    "shape": [image.shape],  # Original shape
                    "filename": [image_path],
                    "inverse_matrix": inverse_matrix,
                }
                polygons_result = self.model.head.get_polygons_from_maps(batch, predictions)

                # Convert to expected format
                if polygons_result:
                    boxes_batch, scores_batch = polygons_result
                    if boxes_batch:
                        boxes = boxes_batch[0]
                        scores = scores_batch[0] if scores_batch else []
                        polygons: list[str] = []
                        texts: list[str] = []
                        confidences: list[float] = []

                        for index, box in enumerate(boxes):
                            if not box or len(box) < 4:
                                continue
                            xs = [point[0] for point in box]
                            ys = [point[1] for point in box]
                            x_min, x_max = min(xs), max(xs)
                            y_min, y_max = min(ys), max(ys)
                            polygon_coords = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
                            polygons.append(",".join(map(str, polygon_coords)))
                            texts.append(f"Text_{index + 1}")
                            confidence = float(scores[index]) if index < len(scores) else 0.0
                            confidences.append(confidence)

                        return {
                            "polygons": "|".join(polygons) if polygons else "",
                            "texts": texts,
                            "confidences": confidences,
                        }

            # Fallback to custom postprocessing
            results = self._postprocess_predictions(predictions, image.shape)
            return results

        except Exception as e:
            logging.error(f"Error during inference: {e}", exc_info=True)
            return None

    @staticmethod
    def _compute_inverse_matrix(processed_image: Any, original_shape: tuple[int, ...]) -> list[np.ndarray]:
        """Return inverse affine matrices mapping model coords back to original image size."""

        if torch is None:
            return [np.eye(3, dtype=np.float32)]

        model_height = int(processed_image.shape[-2])
        model_width = int(processed_image.shape[-1])
        original_height = int(original_shape[0])
        original_width = int(original_shape[1])

        if model_width == 0 or model_height == 0:
            return [np.eye(3, dtype=np.float32)]

        scale_x = original_width / model_width
        scale_y = original_height / model_height
        matrix = np.array(
            [
                [scale_x, 0.0, 0.0],
                [0.0, scale_y, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return [matrix]

    def _preprocess_image(self, image: np.ndarray) -> Any:
        """Preprocess image for model input."""
        # Convert BGR (OpenCV) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Define image transformations based on configured parameters
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.image_size),  # Use configured image size
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std),  # Use configured normalization
            ]
        )
        return transform(image_rgb).unsqueeze(0)

    def _postprocess_predictions(self, predictions: Any, original_shape: tuple[int, ...]) -> dict[str, Any]:
        """
        Post-process model predictions, scaling them to the original image size.
        """
        try:
            prob_map = predictions.get("prob_maps")
            if prob_map is None:
                raise ValueError("'prob_maps' key not found in model predictions.")

            # Ensure prob_map is a NumPy array on the CPU
            if isinstance(prob_map, torch.Tensor):
                prob_map = prob_map.detach().cpu().numpy()

            # Squeeze batch and channel dimensions: (1, 1, H, W) -> (H, W)
            if prob_map.ndim >= 2:
                prob_map = np.squeeze(prob_map)

            # Binarize the probability map to get a mask
            binary_map = (prob_map > self.binarization_thresh).astype(np.uint8)  # Use configured threshold

            # Calculate scaling factors to map coordinates back to the original image
            original_height, original_width, _ = original_shape
            model_height, model_width = prob_map.shape[:2]
            if model_width == 0 or model_height == 0:
                raise ValueError(f"Invalid model dimensions: height={model_height}, width={model_width}")
            scale_x = original_width / model_width
            scale_y = original_height / model_height

            # Find contours on the resized binary map
            contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            polygons, texts, confidences = [], [], []
            for i, contour in enumerate(contours):
                # Limit the number of detection boxes
                if len(polygons) >= self.max_candidates:
                    logging.info(f"Reached maximum candidates limit: {self.max_candidates}")
                    break
                # Get bounding box for each contour
                x, y, w, h = cv2.boundingRect(contour)

                if w < self.min_detection_size or h < self.min_detection_size:  # Use configured minimum size
                    continue

                # Scale the bounding box coordinates to the original image dimensions
                orig_x = int(x * scale_x)
                orig_y = int(y * scale_y)
                orig_w = int(w * scale_x)
                orig_h = int(h * scale_y)

                # Create a rectangular polygon from the scaled bounding box
                polygon_coords = [
                    orig_x,
                    orig_y,
                    orig_x + orig_w,
                    orig_y,
                    orig_x + orig_w,
                    orig_y + orig_h,
                    orig_x,
                    orig_y + orig_h,
                ]
                polygons.append(",".join(map(str, polygon_coords)))

                # TODO: Replace with actual text recognition logic
                texts.append(f"Text_{i + 1}")
                # Calculate confidence from the relevant area of the probability map
                prob_slice = prob_map[y : y + h, x : x + w]
                if prob_slice.size > 0:
                    confidences.append(float(prob_slice.mean()))
                else:
                    confidences.append(0.0)

            return {
                "polygons": "|".join(polygons),
                "texts": texts,
                "confidences": confidences,
            }
        except Exception as e:
            logging.error(f"Error in post-processing: {e}", exc_info=True)
            return self._get_mock_predictions()  # Fallback to mock data on error

    def _get_mock_predictions(self) -> dict[str, Any]:
        """Generate mock predictions for demonstration or fallback."""
        logging.info("Generating mock predictions.")
        return {
            "polygons": "100,100,300,100,300,180,100,180|350,250,600,250,600,300,350,300",
            "texts": ["Mock Text 1", "Mock Text 2"],
            "confidences": [0.98, 0.95],
        }

    def _find_config_for_checkpoint(self, checkpoint_path: str) -> str | None:
        """
        Attempt to find a configuration file associated with a model checkpoint.
        """
        checkpoint_dir = Path(checkpoint_path).parent
        # Search in the checkpoint's directory and its parent
        search_dirs = [checkpoint_dir, checkpoint_dir.parent]

        # Common config file names to look for
        config_patterns = ["config.yaml", "hparams.yaml", "train.yaml", "predict.yaml"]

        for directory in search_dirs:
            for pattern in config_patterns:
                config_path = directory / pattern
                if config_path.exists():
                    return str(config_path)

        # Special check for Hydra's output directory structure
        hydra_config = checkpoint_dir.parent / ".hydra" / "config.yaml"
        if hydra_config.exists():
            return str(hydra_config)

        return None


def run_inference_on_image(
    image_path: str,
    checkpoint_path: str,
    binarization_thresh: float | None = None,
    box_thresh: float | None = None,
    max_candidates: int | None = None,
    min_detection_size: int | None = None,
) -> dict[str, Any] | None:
    """
    Convenience function to initialize the engine and run inference on a single image.

    Args:
        image_path: Path to the image
        checkpoint_path: Path to the model checkpoint
        binarization_thresh: Threshold for binarization (0.0-1.0)
        box_thresh: Threshold for text region proposals (0.0-1.0)
        max_candidates: Maximum number of text region proposals
        min_detection_size: Minimum size of text regions

    Returns:
        dict: Inference results or None on failure
    """
    engine = InferenceEngine()
    if not engine.load_model(checkpoint_path):
        logging.error("Failed to load model in convenience function.")
        return None
    return engine.predict_image(image_path, binarization_thresh, box_thresh, max_candidates, min_detection_size)


def get_available_checkpoints() -> list[str]:
    """
    Scans the project 'outputs' directory for available .ckpt files.
    """
    outputs_dir = project_root / "outputs"
    if not outputs_dir.exists():
        return ["No 'outputs' directory found"]

    checkpoints = [str(ckpt_file.relative_to(project_root)) for ckpt_file in outputs_dir.rglob("*.ckpt")]

    return checkpoints or ["No checkpoints found in 'outputs' directory"]
