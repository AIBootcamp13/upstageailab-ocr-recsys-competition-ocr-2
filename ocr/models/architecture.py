from typing import Any, cast

import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .core import get_registry
from .decoder import get_decoder_by_cfg
from .encoder import get_encoder_by_cfg
from .head import get_head_by_cfg
from .loss import get_loss_by_cfg


class OCRModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.architecture_name = getattr(cfg, "architecture_name", None)

        if self.architecture_name:
            self._init_from_registry(cfg)
        else:
            self._init_from_components(cfg)

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

        return [optimizer], [scheduler] if scheduler else []

    def get_polygons_from_maps(self, batch, pred):
        return self.head.get_polygons_from_maps(batch, pred)

    def _init_from_registry(self, cfg):
        registry = get_registry()
        component_configs = self._prepare_component_configs(cfg)
        architecture_name = str(self.architecture_name)
        components = registry.create_architecture_components(architecture_name, **component_configs)
        self.encoder = components["encoder"]
        self.decoder = components["decoder"]
        self.head = components["head"]
        self.loss = components["loss"]

    def _init_from_components(self, cfg):
        self.encoder = get_encoder_by_cfg(cfg.encoder)

        decoder_cfg = cfg.decoder.copy()
        if hasattr(decoder_cfg, "in_channels") or "in_channels" in decoder_cfg:
            encoder_out_channels = self.encoder.out_channels
            decoder_cfg.in_channels = encoder_out_channels

        self.decoder = get_decoder_by_cfg(decoder_cfg)
        self.head = get_head_by_cfg(cfg.head)
        self.loss = get_loss_by_cfg(cfg.loss)

    def _prepare_component_configs(self, cfg) -> dict:
        overrides = getattr(cfg, "component_overrides", None)
        component_configs: dict[str, Any] = {}
        if overrides is not None:
            for name in ("encoder", "decoder", "head", "loss"):
                if name not in overrides or overrides[name] is None:
                    continue

                section = self._to_container(overrides[name])
                if not isinstance(section, dict):
                    raise TypeError(f"Component override for '{name}' must resolve to a mapping, got {type(section)!r}.")
                section_dict = cast(dict[str, Any], dict(section))
                component_name_key = f"{name}_name"
                component_config_key = f"{name}_config"

                override_name: str | None = None
                override_params: dict[str, Any] | None

                if "name" in section_dict:
                    override_name = section_dict.get("name")
                    params = section_dict.get("params")
                    if params is None:
                        params = {k: v for k, v in section_dict.items() if k != "name"}
                    override_params = dict(params) if isinstance(params, dict) else {}
                else:
                    override_params = section_dict  # legacy format: plain parameter dict

                if override_name:
                    component_configs[component_name_key] = override_name

                component_configs[component_config_key] = override_params or {}
        return component_configs

    @staticmethod
    def _to_container(config_section):
        if isinstance(config_section, DictConfig):
            result = OmegaConf.to_container(config_section, resolve=True)
        elif isinstance(config_section, dict):
            result = dict(config_section)
        else:
            raise TypeError(f"Unsupported override configuration type: {type(config_section)!r}")

        if not isinstance(result, dict):
            raise TypeError("Component overrides must resolve to a dictionary of keyword arguments.")
        return result
