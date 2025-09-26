# inference_engine.py
"""
Inference utilities for OCR UI applications.

This module provides functions to run OCR inference on images using trained models.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    from omegaconf import DictConfig

    from ocr.lightning_modules import get_pl_modules_by_cfg

    OCR_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import OCR modules: {e}. InferenceEngine will use mock predictions.")
    OCR_MODULES_AVAILABLE = False
    # Define dummy classes to prevent NameErrors later
    torch = None
    DictConfig = dict
    pl = None
    yaml = None
    transforms = None


class InferenceEngine:
    """OCR Inference Engine for real-time predictions."""

    def __init__(self):
        self.model_module = None
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

    def load_model(self, checkpoint_path: str, config_path: Optional[str] = None) -> bool:
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
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) if config_path.endswith((".yaml", ".yml")) else json.load(f)
        self.config = DictConfig(config_dict) if OCR_MODULES_AVAILABLE else config_dict
        # Extract preprocessing parameters from config
        self._extract_config_parameters()

        # Initialize model
        self.model_module, _ = get_pl_modules_by_cfg(self.config)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Find the state dictionary in the checkpoint file
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            # Add fallbacks for other common key names
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))

        # Filter out unnecessary keys (e.g., from the optimizer)
        model_state_dict = self.model_module.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

        dropped_keys = set(state_dict.keys()) - set(model_state_dict.keys())
        missing_keys = set(model_state_dict.keys()) - set(state_dict.keys())

        if dropped_keys:
            logging.warning(f"The following keys from the loaded state_dict were dropped and not loaded " f"into the model: {dropped_keys}")
        if missing_keys:
            logging.warning(f"The following keys expected by the model are missing from the loaded state_dict: {missing_keys}")

        self.model_module.load_state_dict(filtered_state_dict, strict=False)
        self.model_module.to(self.device)
        self.model_module.eval()

        logging.info("Model loaded successfully.")
        return True

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
            if hasattr(self.config, "models") and hasattr(self.config.models, "head"):
                head_config = self.config.models.head
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

    def predict_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Run inference on a single image.

        Args:
            image_path: Path to the image file

        Returns:
            dict: Prediction results or None if failed
        """
        if self.model_module is None:
            logging.warning("Model not loaded. Returning mock predictions.")
            return self._get_mock_predictions()

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
                predictions = self.model_module(batch_input)

            # Post-process predictions, passing the original image shape for scaling
            results = self._postprocess_predictions(predictions, image.shape)
            return results

        except Exception as e:
            logging.error(f"Error during inference: {e}", exc_info=True)
            return None

    def _preprocess_image(self, image: np.ndarray) -> "torch.Tensor":
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

    def _postprocess_predictions(self, predictions: Any, original_shape: Tuple[int, int, int]) -> Dict[str, Any]:
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
                texts.append(f"Text_{i+1}")
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

    def _get_mock_predictions(self) -> Dict[str, Any]:
        """Generate mock predictions for demonstration or fallback."""
        logging.info("Generating mock predictions.")
        return {
            "polygons": "100,100,300,100,300,180,100,180|350,250,600,250,600,300,350,300",
            "texts": ["Mock Text 1", "Mock Text 2"],
            "confidences": [0.98, 0.95],
        }

    def _find_config_for_checkpoint(self, checkpoint_path: str) -> Optional[str]:
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


def run_inference_on_image(image_path: str, checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to initialize the engine and run inference on a single image.

    Args:
        image_path: Path to the image
        checkpoint_path: Path to the model checkpoint

    Returns:
        dict: Inference results or None on failure
    """
    engine = InferenceEngine()
    if not engine.load_model(checkpoint_path):
        logging.error("Failed to load model in convenience function.")
        return None
    return engine.predict_image(image_path)


def get_available_checkpoints() -> List[str]:
    """
    Scans the project 'outputs' directory for available .ckpt files.
    """
    outputs_dir = project_root / "outputs"
    if not outputs_dir.exists():
        return ["No 'outputs' directory found"]

    checkpoints = [str(ckpt_file.relative_to(project_root)) for ckpt_file in outputs_dir.rglob("*.ckpt")]

    return checkpoints or ["No checkpoints found in 'outputs' directory"]
