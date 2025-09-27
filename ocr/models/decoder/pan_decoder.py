"""Path Aggregation Network (PAN) decoder for OCR models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import BaseDecoder


def _conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
) -> nn.Sequential:
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class PANDecoder(BaseDecoder):
    """Bi-directional feature fusion decoder inspired by PANet/DBNet."""

    def __init__(
        self,
        in_channels: Sequence[int],
        inner_channels: int = 256,
        out_channels: int = 128,
        output_channels: int | None = None,
    ) -> None:
        if len(in_channels) < 2:
            raise ValueError("PANDecoder requires at least two feature maps from the encoder.")

        super().__init__(list(in_channels))

        if output_channels is not None:
            out_channels = output_channels

        self.reduce_convs = nn.ModuleList(_conv_bn_relu(ch, inner_channels, kernel_size=1) for ch in in_channels)
        self.top_down_smooth = nn.ModuleList(
            _conv_bn_relu(inner_channels, inner_channels, kernel_size=3) for _ in range(len(in_channels) - 1)
        )
        self.bottom_up = nn.ModuleList(
            _conv_bn_relu(inner_channels, inner_channels, kernel_size=3, stride=2) for _ in range(len(in_channels) - 1)
        )

        fused_channels = inner_channels * len(in_channels)
        self.output_conv = _conv_bn_relu(fused_channels, out_channels, kernel_size=3)
        self._output_channels = out_channels

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.reduce_convs):
            raise ValueError(f"Expected {len(self.reduce_convs)} feature maps, received {len(features)}.")

        reduced = [conv(feat) for conv, feat in zip(self.reduce_convs, features, strict=False)]

        # Top-down pathway
        top_down: list[torch.Tensor] = [reduced[-1]]
        for idx, smooth_conv in enumerate(self.top_down_smooth, start=1):
            current = reduced[-(idx + 1)]
            upsampled = F.interpolate(top_down[-1], size=current.shape[-2:], mode="nearest")
            top_down.append(smooth_conv(current + upsampled))
        top_down = list(reversed(top_down))

        # Bottom-up enhancement
        fused_features: list[torch.Tensor] = [top_down[0]]
        for idx, down_conv in enumerate(self.bottom_up, start=0):
            downsampled = F.max_pool2d(fused_features[-1], kernel_size=2, stride=2)
            fused = down_conv(downsampled + top_down[idx + 1])
            fused_features.append(fused)

        base_size = fused_features[0].shape[-2:]
        aligned = [
            feature if feature.shape[-2:] == base_size else F.interpolate(feature, size=base_size, mode="nearest")
            for feature in fused_features
        ]
        concatenated = torch.cat(aligned, dim=1)
        return self.output_conv(concatenated)

    @property
    def out_channels(self) -> int:
        return self._output_channels
