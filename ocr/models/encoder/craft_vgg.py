"""VGG-based encoder tailored for the CRAFT text detector."""

from __future__ import annotations

from collections.abc import Iterable

import timm
import torch
import torch.nn as nn

from ..core import BaseEncoder


class CraftVGGEncoder(BaseEncoder):
    """Feature extractor for CRAFT built on top of VGG-style backbones.

    The original CRAFT paper uses a VGG16-BN backbone with additional convolution
    layers to enhance high-level feature representations. This implementation
    reproduces that behaviour using the flexible `timm` feature extraction API.
    """

    def __init__(
        self,
        model_name: str = "vgg16_bn",
        pretrained: bool = True,
        output_indices: Iterable[int] | None = None,
        extra_channels: int = 512,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        if output_indices is None:
            # Match the four feature levels used in the original CRAFT network.
            output_indices = (1, 2, 3, 4)

        self.output_indices = tuple(output_indices)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.output_indices,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Additional convolution to enrich the deepest feature map.
        deepest_channels = self.backbone.feature_info[self.output_indices[-1]]["num_chs"]
        self.enhance = nn.Sequential(
            nn.Conv2d(deepest_channels, extra_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(extra_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(extra_channels, extra_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(extra_channels),
            nn.ReLU(inplace=True),
        )

        self._out_channels = [self.backbone.feature_info[index]["num_chs"] for index in self.output_indices]
        self._out_channels[-1] = extra_channels
        self._strides = [self.backbone.feature_info[index]["reduction"] for index in self.output_indices]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = self.backbone(x)
        if len(features) != len(self.output_indices):
            raise RuntimeError(
                f"Unexpected number of feature maps from backbone. Expected {len(self.output_indices)}, got {len(features)}."
            )

        features = list(features)
        features[-1] = self.enhance(features[-1])
        return features

    @property
    def out_channels(self) -> list[int]:
        return list(self._out_channels)

    @property
    def strides(self) -> list[int]:
        return list(self._strides)
