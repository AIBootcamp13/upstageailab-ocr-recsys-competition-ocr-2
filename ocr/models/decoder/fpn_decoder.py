"""Feature Pyramid Network (FPN) style decoder for OCR models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import BaseDecoder


def _conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> nn.Sequential:
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class FPNDecoder(BaseDecoder):
    """Top-down feature pyramid decoder suitable for text detection."""

    def __init__(
        self,
        in_channels: Sequence[int],
        inner_channels: int = 256,
        out_channels: int = 128,
        output_channels: int | None = None,
    ) -> None:
        if len(in_channels) < 2:
            raise ValueError("FPNDecoder requires at least two feature maps from the encoder.")

        super().__init__(list(in_channels))

        if output_channels is not None:
            out_channels = output_channels

        self.lateral_convs = nn.ModuleList(_conv_bn_relu(ch, inner_channels, kernel_size=1) for ch in in_channels)
        self.output_convs = nn.ModuleList(_conv_bn_relu(inner_channels, inner_channels, kernel_size=3) for _ in in_channels)

        fused_channels = inner_channels * len(in_channels)
        self.fusion = _conv_bn_relu(fused_channels, out_channels, kernel_size=3)
        self._output_channels = out_channels

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.lateral_convs):
            raise ValueError(f"Expected {len(self.lateral_convs)} feature maps, received {len(features)}.")

        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features, strict=False)]

        last_inner = laterals[-1]
        pyramid_features: list[torch.Tensor] = [self.output_convs[-1](last_inner)]
        for idx in range(len(laterals) - 2, -1, -1):
            current = laterals[idx]
            upsampled = F.interpolate(last_inner, size=current.shape[-2:], mode="nearest")
            last_inner = current + upsampled
            pyramid_features.insert(0, self.output_convs[idx](last_inner))

        base_size = pyramid_features[0].shape[-2:]
        aligned = [
            feature if feature.shape[-2:] == base_size else F.interpolate(feature, size=base_size, mode="nearest")
            for feature in pyramid_features
        ]
        fused = torch.cat(aligned, dim=1)
        return self.fusion(fused)

    @property
    def out_channels(self) -> int:
        return self._output_channels
