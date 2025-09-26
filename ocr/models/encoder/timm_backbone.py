"""Timm-based backbone encoder for OCR models."""

import timm
import torch

from ..core import BaseEncoder


class TimmBackbone(BaseEncoder):
    """Timm-based backbone encoder that extracts multi-scale features.

    Uses pretrained models from the timm library to extract features at different
    scales for use in OCR detection models.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        select_features: list[int] | None = None,
        pretrained: bool = True,
        **kwargs,
    ):
        """Initialize the Timm backbone.

        Args:
            model_name: Name of the timm model to use (e.g., 'resnet18', 'efficientnet_b0')
            select_features: Indices of feature levels to extract. If None, uses [1, 2, 3, 4]
            pretrained: Whether to use pretrained weights
            **kwargs: Additional arguments passed to timm.create_model
        """
        super().__init__(**kwargs)

        if select_features is None:
            select_features = [1, 2, 3, 4]

        # Create timm model for feature extraction only
        self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, **kwargs)

        # Store feature selection indices
        self.select_features = select_features

        # Cache feature information for property access
        self._feature_info = self.model.feature_info
        self._out_channels = [self._feature_info[i]["num_chs"] for i in self.select_features]
        self._strides = [self._feature_info[i]["reduction"] for i in self.select_features]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract features from input images.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            List of feature tensors at selected levels
        """
        features = self.model(x)
        return [features[i] for i in self.select_features]

    @property
    def out_channels(self) -> list[int]:
        """Return the number of output channels for each feature level."""
        return self._out_channels

    @property
    def strides(self) -> list[int]:
        """Return the stride (downsampling factor) for each feature level."""
        return self._strides
