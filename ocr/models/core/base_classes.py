"""Abstract base classes for OCR model components."""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for OCR encoders/backbones.

    Encoders are responsible for extracting features from input images.
    They should return a list of feature maps at different scales.
    """

    def __init__(self, **kwargs):
        """Initialize the encoder.

        Args:
            **kwargs: Configuration parameters specific to the encoder implementation.
        """
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract features from input images.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            List of feature tensors at different scales, typically from shallow to deep layers.
            Each tensor should have shape (B, C_i, H_i, W_i) where H_i, W_i decrease with depth.
        """
        pass

    @property
    @abstractmethod
    def out_channels(self) -> list[int]:
        """Return the number of output channels for each feature level.

        Returns:
            List of integers representing output channels for each feature level.
        """
        pass

    @property
    @abstractmethod
    def strides(self) -> list[int]:
        """Return the stride (downsampling factor) for each feature level.

        Returns:
            List of integers representing stride for each feature level.
        """
        pass

    def get_feature_info(self) -> dict[str, Any]:
        """Get information about the features produced by this encoder.

        Returns:
            Dictionary containing feature information like channels, strides, etc.
        """
        return {
            "out_channels": self.out_channels,
            "strides": self.strides,
            "num_levels": len(self.out_channels),
        }


class BaseDecoder(nn.Module, ABC):
    """Abstract base class for OCR decoders.

    Decoders take encoder features and produce higher-resolution feature maps
    suitable for dense prediction tasks like text detection.
    """

    def __init__(self, in_channels: list[int], **kwargs):
        """Initialize the decoder.

        Args:
            in_channels: List of input channels from encoder feature levels.
            **kwargs: Additional configuration parameters.
        """
        super().__init__()
        self.in_channels = in_channels

    @abstractmethod
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Decode features to produce dense prediction maps.

        Args:
            features: List of feature tensors from encoder, ordered from shallow to deep.

        Returns:
            Decoded feature tensor suitable for head prediction.
            Typically shape (B, C, H, W) where H, W match input image dimensions.
        """
        pass

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Return the number of output channels from the decoder.

        Returns:
            Number of output channels.
        """
        pass


class BaseHead(nn.Module, ABC):
    """Abstract base class for OCR prediction heads.

    Heads take decoded features and produce final predictions like
    text region segmentation maps or bounding boxes.
    """

    def __init__(self, in_channels: int, **kwargs):
        """Initialize the head.

        Args:
            in_channels: Number of input channels from decoder.
            **kwargs: Additional configuration parameters.
        """
        super().__init__()
        self.in_channels = in_channels

    @abstractmethod
    def forward(self, x: torch.Tensor, return_loss: bool = True) -> dict[str, torch.Tensor]:
        """Produce predictions from decoded features.

        Args:
            x: Input tensor from decoder of shape (B, C, H, W)
            return_loss: Whether to include loss computation in output

        Returns:
            Dictionary containing predictions and optionally loss.
            Must include keys like 'binary_map', 'thresh_map', etc.
        """
        pass

    @abstractmethod
    def get_polygons_from_maps(
        self, pred_maps: dict[str, torch.Tensor], ground_truth: dict[str, torch.Tensor] | None = None
    ) -> list[list[list[float]]]:
        """Extract polygons from prediction maps.

        Args:
            pred_maps: Dictionary of prediction maps from forward pass
            ground_truth: Optional ground truth for evaluation

        Returns:
            List of polygons, where each polygon is a list of [x, y] coordinates.
            Shape: (batch_size, num_polygons, num_points, 2)
        """
        pass


class BaseLoss(nn.Module, ABC):
    """Abstract base class for OCR loss functions.

    Loss functions compute training objectives for text detection models.
    """

    def __init__(self, **kwargs):
        """Initialize the loss function.

        Args:
            **kwargs: Configuration parameters for the loss function.
        """
        super().__init__()

    @abstractmethod
    def forward(
        self, pred: dict[str, torch.Tensor], gt_binary: torch.Tensor, gt_thresh: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute loss from predictions and ground truth.

        Args:
            pred: Dictionary of predictions from head
            gt_binary: Ground truth binary map of shape (B, 1, H, W)
            gt_thresh: Ground truth threshold map of shape (B, 1, H, W)
            **kwargs: Additional loss-specific parameters

        Returns:
            Tuple of (total_loss, loss_dict) where:
            - total_loss: Scalar tensor for optimization
            - loss_dict: Dictionary of individual loss components for logging
        """
        pass


class BaseMetric(nn.Module, ABC):
    """Abstract base class for OCR evaluation metrics.

    Metrics compute evaluation scores for text detection performance.
    """

    def __init__(self, **kwargs):
        """Initialize the metric.

        Args:
            **kwargs: Configuration parameters for the metric.
        """
        super().__init__()

    @abstractmethod
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric state with new predictions and targets.

        Args:
            preds: Predicted polygons or maps
            targets: Ground truth polygons or maps
        """
        pass

    @abstractmethod
    def compute(self) -> dict[str, torch.Tensor]:
        """Compute final metric values.

        Returns:
            Dictionary of metric values (precision, recall, f1, etc.)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset metric state for new evaluation."""
        pass
