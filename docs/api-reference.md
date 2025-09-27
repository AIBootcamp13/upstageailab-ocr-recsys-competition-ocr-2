# API Reference

## Core Components

### OCRLightningModule

Main PyTorch Lightning module for OCR training and inference.

```python
class OCRLightningModule(pl.LightningModule):
    """PyTorch Lightning module for OCR models."""

    def __init__(self, config: DictConfig):
        """Initialize OCR Lightning module.

        Args:
            config: Complete configuration object containing model,
                   training, and data parameters.
        """
        pass

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass through the model.

        Args:
            x: Input batch of images [B, C, H, W]

        Returns:
            Model predictions
        """
        pass

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Training batch containing images and targets
            batch_idx: Batch index

        Returns:
            Loss value
        """
        pass

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Validation step.

        Args:
            batch: Validation batch
            batch_idx: Batch index

        Returns:
            Validation metrics
        """
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers.

        Returns:
            Optimizer and scheduler configuration
        """
        pass
```

### OCRModel

Main model class that combines encoder, decoder, head, and loss components.

```python
class OCRModel(nn.Module):
    """OCR model combining encoder, decoder, head, and loss."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        head: nn.Module,
        loss: nn.Module
    ):
        """Initialize OCR model.

        Args:
            encoder: Feature encoder network
            decoder: Feature decoder network
            head: Task-specific head
            loss: Loss function
        """
        pass

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Model predictions
        """
        pass

    def compute_loss(self, predictions: Any, targets: Any) -> torch.Tensor:
        """Compute loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Loss value
        """
        pass
```

## Model Components

### TimmBackbone

Timm-based backbone encoder for feature extraction.

```python
class TimmBackbone(nn.Module):
    """Timm backbone wrapper for feature extraction."""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        in_channels: int = 3
    ):
        """Initialize Timm backbone.

        Args:
            backbone: Timm backbone name (e.g., 'resnet50', 'efficientnet_b0')
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone parameters
            in_channels: Number of input channels
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Feature tensor [B, C', H', W']
        """
        pass

    @property
    def output_channels(self) -> int:
        """Number of output channels."""
        pass

    @property
    def output_stride(self) -> int:
        """Output stride relative to input."""
        pass
```

### UNetDecoder

U-Net decoder for feature upsampling and refinement.

```python
class UNetDecoder(nn.Module):
    """U-Net decoder with skip connections."""

    def __init__(
        self,
        in_channels: int = 2048,
        out_channels: int = 256,
        bilinear: bool = True
    ):
        """Initialize U-Net decoder.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bilinear: Whether to use bilinear upsampling
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode features.

        Args:
            x: Input features [B, C, H, W]

        Returns:
            Decoded features [B, C', H', W']
        """
        pass
```

### DBNetHead

DBNet-specific head for text detection.

```python
class DBNetHead(nn.Module):
    """DBNet head for text detection."""

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 1,
        k: int = 50
    ):
        """Initialize DBNet head.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            k: Kernel size for adaptive average pooling
        """
        pass

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input features [B, C, H, W]

        Returns:
            Tuple of (probability_map, threshold_map)
        """
        pass

    def postprocess(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor],
        threshold: float = 0.3,
        **kwargs
    ) -> Dict[str, Any]:
        """Post-process predictions.

        Args:
            predictions: Raw predictions from forward pass
            threshold: Binarization threshold
            **kwargs: Additional post-processing parameters

        Returns:
            Dictionary containing polygons and scores
        """
        pass
```

## Loss Functions

### DBLoss

Combined loss for DBNet training.

```python
class DBLoss(nn.Module):
    """DBNet combined loss function."""

    def __init__(
        self,
        weight_bce: float = 1.0,
        weight_dice: float = 1.0,
        weight_l1: float = 0.01
    ):
        """Initialize DB loss.

        Args:
            weight_bce: Weight for binary cross-entropy loss
            weight_dice: Weight for dice loss
            weight_l1: Weight for L1 loss
        """
        pass

    def forward(
        self,
        pred_maps: torch.Tensor,
        pred_thresholds: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute DB loss.

        Args:
            pred_maps: Predicted probability maps [B, 1, H, W]
            pred_thresholds: Predicted threshold maps [B, 1, H, W]
            targets: Ground truth masks [B, 1, H, W]

        Returns:
            Combined loss value
        """
        pass

    def get_loss_components(self) -> Dict[str, torch.Tensor]:
        """Get individual loss components.

        Returns:
            Dictionary of loss components
        """
        pass
```

### DiceLoss

Dice coefficient-based loss function.

```python
class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""

    def __init__(self, smooth: float = 1.0):
        """Initialize dice loss.

        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        pass

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute dice loss.

        Args:
            pred: Predictions [B, C, H, W]
            target: Targets [B, C, H, W]

        Returns:
            Dice loss value
        """
        pass
```

## Dataset Classes

### OCRDataset

Base dataset class for OCR tasks.

```python
class OCRDataset(torch.utils.data.Dataset):
    """Base dataset for OCR tasks."""

    def __init__(
        self,
        data_dir: str,
        transforms: Optional[Callable] = None,
        split: str = "train"
    ):
        """Initialize OCR dataset.

        Args:
            data_dir: Path to data directory
            transforms: Data transformations
            split: Data split ('train', 'val', 'test')
        """
        pass

    def __len__(self) -> int:
        """Get dataset length."""
        pass

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item.

        Args:
            idx: Item index

        Returns:
            Dictionary containing image, polygons, and metadata
        """
        pass

    def load_annotations(self, annotation_file: str) -> List[Dict[str, Any]]:
        """Load annotations from file.

        Args:
            annotation_file: Path to annotation file

        Returns:
            List of annotations
        """
        pass
```

## Data Transformations

### OCRTransforms

Data augmentation and preprocessing transforms.

```python
class OCRTransforms:
    """OCR data transforms with augmentations."""

    def __init__(
        self,
        is_training: bool = True,
        image_size: Tuple[int, int] = (512, 512),
        augmentations: Optional[List[Dict[str, Any]]] = None
    ):
        """Initialize OCR transforms.

        Args:
            is_training: Whether transforms are for training
            image_size: Target image size
            augmentations: List of augmentation configurations
        """
        pass

    def __call__(self, image: np.ndarray, annotations: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply transforms.

        Args:
            image: Input image
            annotations: Image annotations

        Returns:
            Tuple of (transformed_image, transformed_annotations)
        """
        pass

    def apply_augmentations(self, image: np.ndarray, annotations: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply data augmentations.

        Args:
            image: Input image
            annotations: Image annotations

        Returns:
            Augmented image and annotations
        """
        pass
```

## Metrics

### CLEvalMetric

CLEval-based evaluation metric for OCR.

```python
class CLEvalMetric(torchmetrics.Metric):
    """CLEval metric for OCR evaluation."""

    def __init__(
        self,
        eval_type: str = "text_detection",
        iou_threshold: float = 0.5,
        area_precision_constrain: float = 0.5
    ):
        """Initialize CLEval metric.

        Args:
            eval_type: Type of evaluation ('text_detection', 'text_recognition')
            iou_threshold: IoU threshold for matching
            area_precision_constrain: Area precision constraint
        """
        pass

    def update(self, preds: List[Dict[str, Any]], targets: List[Dict[str, Any]]):
        """Update metric state.

        Args:
            preds: List of predictions
            targets: List of ground truth targets
        """
        pass

    def compute(self) -> Dict[str, float]:
        """Compute final metrics.

        Returns:
            Dictionary containing precision, recall, f1, etc.
        """
        pass

    def reset(self):
        """Reset metric state."""
        pass
```

## Utilities

### OCRUtils

General OCR utility functions.

```python
class OCRUtils:
    """OCR utility functions."""

    @staticmethod
    def polygons_to_mask(
        polygons: List[List[float]],
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """Convert polygons to binary mask.

        Args:
            polygons: List of polygons
            image_size: Image size (H, W)

        Returns:
            Binary mask [H, W]
        """
        pass

    @staticmethod
    def mask_to_polygons(
        mask: np.ndarray,
        min_area: int = 10
    ) -> List[List[float]]:
        """Convert binary mask to polygons.

        Args:
            mask: Binary mask [H, W]
            min_area: Minimum polygon area

        Returns:
            List of polygons
        """
        pass

    @staticmethod
    def calculate_iou(
        polygon1: List[float],
        polygon2: List[float]
    ) -> float:
        """Calculate IoU between two polygons.

        Args:
            polygon1: First polygon
            polygon2: Second polygon

        Returns:
            IoU score
        """
        pass

    @staticmethod
    def visualize_predictions(
        image: np.ndarray,
        polygons: List[List[float]],
        scores: Optional[List[float]] = None
    ) -> np.ndarray:
        """Visualize predictions on image.

        Args:
            image: Input image
            polygons: Detected polygons
            scores: Confidence scores

        Returns:
            Image with visualizations
        """
        pass
```

## Configuration Classes

### ConfigValidator

Configuration validation utilities.

```python
class ConfigValidator:
    """Configuration validation utilities."""

    @staticmethod
    def validate_model_config(config: DictConfig) -> List[str]:
        """Validate model configuration.

        Args:
            config: Model configuration

        Returns:
            List of validation errors
        """
        pass

    @staticmethod
    def validate_training_config(config: DictConfig) -> List[str]:
        """Validate training configuration.

        Args:
            config: Training configuration

        Returns:
            List of validation errors
        """
        pass

    @staticmethod
    def validate_data_config(config: DictConfig) -> List[str]:
        """Validate data configuration.

        Args:
            config: Data configuration

        Returns:
            List of validation errors
        """
        pass

    @staticmethod
    def get_config_schema() -> Dict[str, Any]:
        """Get configuration schema.

        Returns:
            Configuration schema dictionary
        """
        pass
```

## Runner Classes

### TrainRunner

Training runner for experiments.

```python
class TrainRunner:
    """Training runner for OCR experiments."""

    def __init__(self, config: DictConfig):
        """Initialize training runner.

        Args:
            config: Complete experiment configuration
        """
        pass

    def setup(self):
        """Setup training components."""
        pass

    def run(self):
        """Run training experiment."""
        pass

    def cleanup(self):
        """Cleanup after training."""
        pass
```

### TestRunner

Testing and evaluation runner.

```python
class TestRunner:
    """Testing and evaluation runner."""

    def __init__(self, config: DictConfig):
        """Initialize test runner.

        Args:
            config: Test configuration
        """
        pass

    def setup(self):
        """Setup testing components."""
        pass

    def run_inference(self) -> List[Dict[str, Any]]:
        """Run inference on test set.

        Returns:
            List of predictions
        """
        pass

    def evaluate(self) -> Dict[str, float]:
        """Evaluate predictions.

        Returns:
            Evaluation metrics
        """
        pass

    def save_results(self, output_dir: str):
        """Save results to directory.

        Args:
            output_dir: Output directory path
        """
        pass
```

### PredictRunner

Inference runner for new images.

```python
class PredictRunner:
    """Inference runner for prediction on new images."""

    def __init__(self, config: DictConfig, checkpoint_path: str):
        """Initialize prediction runner.

        Args:
            config: Prediction configuration
            checkpoint_path: Path to model checkpoint
        """
        pass

    def setup(self):
        """Setup prediction components."""
        pass

    def predict_image(self, image_path: str) -> Dict[str, Any]:
        """Predict on single image.

        Args:
            image_path: Path to input image

        Returns:
            Prediction results
        """
        pass

    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Predict on batch of images.

        Args:
            image_paths: List of image paths

        Returns:
            List of prediction results
        """
        pass

    def visualize_results(
        self,
        image_path: str,
        predictions: Dict[str, Any],
        output_path: str
    ):
        """Visualize predictions on image.

        Args:
            image_path: Input image path
            predictions: Prediction results
            output_path: Output visualization path
        """
        pass
```

## Exception Classes

### ConfigurationError

Configuration-related errors.

```python
class ConfigurationError(ValueError):
    """Configuration validation error."""

    def __init__(self, message: str, field: Optional[str] = None):
        """Initialize configuration error.

        Args:
            message: Error message
            field: Configuration field that caused the error
        """
        pass
```

### DatasetError

Dataset-related errors.

```python
class DatasetError(RuntimeError):
    """Dataset loading or processing error."""

    def __init__(self, message: str, dataset_path: Optional[str] = None):
        """Initialize dataset error.

        Args:
            message: Error message
            dataset_path: Path to problematic dataset
        """
        pass
```

### ModelError

Model-related errors.

```python
class ModelError(RuntimeError):
    """Model loading or inference error."""

    def __init__(self, message: str, model_path: Optional[str] = None):
        """Initialize model error.

        Args:
            message: Error message
            model_path: Path to problematic model
        """
        pass
```

This API reference provides comprehensive documentation for all major classes, functions, and interfaces in the OCR framework. Each component includes detailed parameter descriptions, return types, and usage examples.
