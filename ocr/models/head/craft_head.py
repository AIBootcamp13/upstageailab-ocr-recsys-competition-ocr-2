"""CRAFT prediction head."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core import BaseHead
from .craft_postprocess import CraftPostProcessor


class CraftHead(BaseHead):
    """CRAFT head producing region and affinity score maps."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        postprocess: dict | None = None,
    ) -> None:
        super().__init__(in_channels=in_channels)
        self.region_branch = self._make_branch(in_channels, hidden_channels)
        self.affinity_branch = self._make_branch(in_channels, hidden_channels)
        self.sigmoid = nn.Sigmoid()
        self.postprocess = CraftPostProcessor(**postprocess) if postprocess else CraftPostProcessor()

    @staticmethod
    def _make_branch(in_channels: int, hidden_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, return_loss: bool = True) -> dict[str, torch.Tensor]:
        region_logits = self.region_branch(x)
        affinity_logits = self.affinity_branch(x)

        region_score = self.sigmoid(region_logits)
        affinity_score = self.sigmoid(affinity_logits)

        return {
            "region_logits": region_logits,
            "affinity_logits": affinity_logits,
            "region_score": region_score,
            "affinity_score": affinity_score,
            # Compatibility aliases for existing training pipeline
            "binary_logits": region_logits,
            "prob_maps": region_score,
            "thresh_map": affinity_score,
        }

    def get_polygons_from_maps(self, batch, pred):
        return self.postprocess.represent(batch, pred)
