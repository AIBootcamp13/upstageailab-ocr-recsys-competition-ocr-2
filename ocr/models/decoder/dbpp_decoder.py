"""Enhanced decoder inspired by DBNet++."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import BaseDecoder


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class DBPPDecoder(BaseDecoder):
    """Bi-directional feature pyramid decoder for DBNet++."""

    def __init__(
        self,
        in_channels: Sequence[int],
        inner_channels: int = 256,
        out_channels: int = 128,
    ) -> None:
        super().__init__(in_channels=list(in_channels))

        self.lateral_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, inner_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(inner_channels),
                    nn.ReLU(inplace=True),
                )
                for ch in in_channels
            ]
        )

        self.top_down = nn.ModuleList(DepthwiseSeparableConv(inner_channels, inner_channels) for _ in range(len(in_channels) - 1))
        self.bottom_up = nn.ModuleList(DepthwiseSeparableConv(inner_channels, inner_channels) for _ in range(len(in_channels) - 1))

        self.output_conv = nn.Sequential(
            nn.Conv2d(inner_channels * len(in_channels), out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._output_channels = out_channels

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.lateral_convs):
            raise ValueError(f"Expected {len(self.lateral_convs)} feature maps, received {len(features)}.")

        lateral = [conv(feat) for conv, feat in zip(self.lateral_convs, features, strict=False)]

        top_down_feats: list[torch.Tensor] = [lateral[-1]]
        for idx, module in enumerate(self.top_down, start=1):
            current = lateral[-(idx + 1)]
            upsampled = F.interpolate(top_down_feats[-1], size=current.shape[-2:], mode="nearest")
            top_down_feats.append(module(current + upsampled))
        top_down_feats = list(reversed(top_down_feats))

        fused: list[torch.Tensor] = [top_down_feats[0]]
        for idx, module in enumerate(self.bottom_up, start=1):
            current = top_down_feats[idx]
            downsampled = F.interpolate(fused[-1], size=current.shape[-2:], mode="nearest")
            fused.append(module(current + downsampled))

        base_size = fused[0].shape[-2:]
        pyramid = [
            F.interpolate(feature, size=base_size, mode="nearest") if feature.shape[-2:] != base_size else feature for feature in fused
        ]
        concatenated = torch.cat(pyramid, dim=1)
        return self.output_conv(concatenated)

    @property
    def out_channels(self) -> int:
        return self._output_channels
