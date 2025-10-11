"""Utility helpers for configuration extraction in the OCR Lightning module."""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf


def extract_metric_kwargs(metric_cfg: DictConfig | None) -> dict:
    """Return kwargs for CLEval metrics from an OmegaConf node."""
    if metric_cfg is None:
        return {}

    cfg_dict = OmegaConf.to_container(metric_cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        return {}

    cfg_dict.pop("_target_", None)
    return cfg_dict


def extract_normalize_stats(config) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Locate normalize transform statistics if they exist in the config."""
    transforms_cfg = getattr(config, "transforms", None)
    if transforms_cfg is None:
        return None, None

    sections: list[ListConfig] = []
    for attr in ("train_transform", "val_transform", "test_transform", "predict_transform"):
        section = getattr(transforms_cfg, attr, None)
        if section is None:
            continue
        transforms = getattr(section, "transforms", None)
        if isinstance(transforms, ListConfig):
            sections.append(transforms)

    for transforms in sections:
        for transform in transforms:
            transform_dict = OmegaConf.to_container(transform, resolve=True)
            if not isinstance(transform_dict, dict):
                continue
            target = transform_dict.get("_target_")
            if target != "albumentations.Normalize":
                continue
            mean = transform_dict.get("mean")
            std = transform_dict.get("std")
            if mean is None or std is None:
                continue
            try:
                mean_array = np.array(mean, dtype=np.float32)
                std_array = np.array(std, dtype=np.float32)
            except Exception:  # noqa: BLE001
                continue
            if mean_array.size == std_array.size and mean_array.size in {1, 3}:
                return mean_array, std_array

    return None, None
