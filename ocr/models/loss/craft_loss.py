"""Loss function for the CRAFT text detector."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core import BaseLoss


class CraftLoss(BaseLoss):
    """Mean-squared error loss over region and affinity maps."""

    def __init__(
        self,
        region_weight: float = 1.0,
        affinity_weight: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.region_weight = region_weight
        self.affinity_weight = affinity_weight
        self.eps = eps

    def forward(  # type: ignore[override]
        self,
        pred: dict[str, torch.Tensor],
        gt_binary: torch.Tensor,
        gt_thresh: torch.Tensor,
        region_mask: torch.Tensor | None = None,
        affinity_mask: torch.Tensor | None = None,
        **_: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        region_pred = self._resolve_prediction(pred, "region_score", "region_logits")
        affinity_pred = self._resolve_prediction(pred, "affinity_score", "affinity_logits")

        region_loss = self._masked_mse(region_pred, gt_binary, region_mask)
        affinity_loss = self._masked_mse(affinity_pred, gt_thresh, affinity_mask)

        total_loss = self.region_weight * region_loss + self.affinity_weight * affinity_loss
        loss_dict = {
            "loss_region": region_loss.detach(),
            "loss_affinity": affinity_loss.detach(),
        }
        return total_loss, loss_dict

    def _masked_mse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if prediction.shape[-2:] != target.shape[-2:]:
            prediction = F.interpolate(prediction, size=target.shape[-2:], mode="bilinear", align_corners=False)
        loss = F.mse_loss(prediction, target, reduction="none")
        if mask is not None:
            loss = loss * mask
            denom = mask.sum() + self.eps
        else:
            denom = torch.tensor(loss.numel(), device=loss.device, dtype=loss.dtype)
        return loss.sum() / denom

    @staticmethod
    def _resolve_prediction(pred: dict[str, torch.Tensor], score_key: str, logit_key: str) -> torch.Tensor:
        score = pred.get(score_key)
        if score is not None:
            return score
        logits = pred.get(logit_key)
        if logits is not None:
            return logits.sigmoid()
        raise KeyError(f"CRAFT predictions must include '{score_key}' or '{logit_key}'.")
