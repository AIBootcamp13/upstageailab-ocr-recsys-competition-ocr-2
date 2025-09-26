"""Decoder module for the CRAFT text detection architecture."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import BaseDecoder


class CraftDecoder(BaseDecoder):
    """Multi-scale feature fusion decoder for CRAFT."""

    def __init__(
        self,
        in_channels: Sequence[int],
        inner_channels: int = 256,
        out_channels: int = 256,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__(in_channels=list(in_channels))

        norm_layer = nn.BatchNorm2d if use_batch_norm else nn.Identity

        def make_reduce_layer(in_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, inner_channels, kernel_size=1, bias=not use_batch_norm),
                norm_layer(inner_channels),
                nn.ReLU(inplace=True),
            )

        def make_smooth_layer() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1, bias=not use_batch_norm),
                norm_layer(inner_channels),
                nn.ReLU(inplace=True),
            )

        self.reduce_layers = nn.ModuleList([make_reduce_layer(ch) for ch in in_channels])
        self.smooth_layers = nn.ModuleList([make_smooth_layer() for _ in in_channels])

        self.out_projection = nn.Sequential(
            nn.Conv2d(inner_channels * len(in_channels), out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

        self._output_channels = out_channels

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.reduce_layers):
            raise ValueError(f"CraftDecoder expected {len(self.reduce_layers)} feature maps but received {len(features)}.")

        reduced = [layer(feat) for layer, feat in zip(self.reduce_layers, features, strict=False)]

        fused_levels: list[torch.Tensor | None] = [None] * len(reduced)
        for idx in reversed(range(len(reduced))):
            if idx == len(reduced) - 1:
                fused_levels[idx] = self.smooth_layers[idx](reduced[idx])
            else:
                upsampled = F.interpolate(
                    fused_levels[idx + 1],
                    size=reduced[idx].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                fused_levels[idx] = self.smooth_layers[idx](reduced[idx] + upsampled)

        fused = [level for level in fused_levels if level is not None]

        base_size = fused[0].shape[-2:]
        pyramid = [fused[0]] + [F.interpolate(feature, size=base_size, mode="bilinear", align_corners=False) for feature in fused[1:]]

        concatenated = torch.cat(pyramid, dim=1)
        return self.out_projection(concatenated)

    @property
    def out_channels(self) -> int:
        return self._output_channels
