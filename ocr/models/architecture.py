import torch.nn as nn
from hydra.utils import instantiate

from .decoder import get_decoder_by_cfg
from .encoder import get_encoder_by_cfg
from .head import get_head_by_cfg
from .loss import get_loss_by_cfg


class OCRModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 각 모듈 instantiate
        self.encoder = get_encoder_by_cfg(cfg.encoder)

        # Dynamically configure decoder with encoder's output channels
        decoder_cfg = cfg.decoder.copy()
        if hasattr(decoder_cfg, "in_channels") or "in_channels" in decoder_cfg:
            # Use encoder's output channels for decoder input
            encoder_out_channels = self.encoder.out_channels
            decoder_cfg.in_channels = encoder_out_channels

        self.decoder = get_decoder_by_cfg(decoder_cfg)
        self.head = get_head_by_cfg(cfg.head)
        self.loss = get_loss_by_cfg(cfg.loss)

    def forward(self, images, return_loss=True, **kwargs):
        encoded_features = self.encoder(images)
        decoded_features = self.decoder(encoded_features)
        pred = self.head(decoded_features, return_loss)

        # Loss 계산
        if return_loss:
            # Extract ground truth from kwargs
            gt_binary = kwargs.get("prob_maps")
            gt_thresh = kwargs.get("thresh_maps")
            if gt_binary is not None and gt_thresh is not None:
                loss, loss_dict = self.loss(pred, gt_binary, gt_thresh, **kwargs)
            else:
                # Fallback for cases where ground truth is not available
                loss, loss_dict = self.loss(pred, **kwargs)
            pred.update(loss=loss, loss_dict=loss_dict)

        return pred

    def get_optimizers(self):
        optimizer_config = self.cfg.optimizer
        optimizer = instantiate(optimizer_config, params=self.parameters())

        scheduler = None
        if "scheduler" in self.cfg:
            scheduler_config = self.cfg.scheduler
            scheduler = instantiate(scheduler_config, optimizer=optimizer)

        return optimizer, scheduler

    def get_polygons_from_maps(self, batch, pred):
        return self.head.get_polygons_from_maps(batch, pred)
