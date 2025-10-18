from __future__ import annotations

"""Configuration helpers for OCR inference."""

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dependencies import OCR_MODULES_AVAILABLE, DictConfig, yaml

LOGGER = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = (640, 640)
DEFAULT_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_NORMALIZE_STD = [0.229, 0.224, 0.225]
DEFAULT_BINARIZATION_THRESH = 0.3
DEFAULT_BOX_THRESH = 0.4
DEFAULT_MAX_CANDIDATES = 300
DEFAULT_MIN_DETECTION_SIZE = 5


@dataclass(slots=True)
class NormalizationSettings:
    mean: list[float]
    std: list[float]


@dataclass(slots=True)
class PreprocessSettings:
    image_size: tuple[int, int]
    normalization: NormalizationSettings


@dataclass(slots=True)
class PostprocessSettings:
    binarization_thresh: float
    box_thresh: float
    max_candidates: int
    min_detection_size: int


@dataclass(slots=True)
class ModelConfigBundle:
    raw_config: Any
    preprocess: PreprocessSettings
    postprocess: PostprocessSettings


def resolve_config_path(checkpoint_path: str | Path, explicit_config: str | Path | None, search_dirs: Iterable[Path]) -> Path | None:
    """Return the configuration file path for a checkpoint."""

    if explicit_config is not None:
        config_path = Path(explicit_config)
        if config_path.exists():
            return config_path
        LOGGER.warning("Explicit config %s does not exist.", config_path)

    checkpoint_path = Path(checkpoint_path).resolve()

    # First, check for .config.json file alongside the checkpoint (new filesystem refactoring)
    config_json_path = checkpoint_path.with_suffix(".config.json")
    if config_json_path.exists():
        return config_json_path

    checkpoint_parent = checkpoint_path.parent
    candidates = list(search_dirs)
    candidates.extend([checkpoint_parent, checkpoint_parent.parent])

    for directory in candidates:
        for pattern in ("config.yaml", "hparams.yaml", "train.yaml", "predict.yaml"):
            candidate = directory / pattern
            if candidate.exists():
                return candidate

    # Check .hydra directories at different levels
    hydra_candidates = [
        checkpoint_parent.parent / ".hydra" / "config.yaml",  # checkpoints/.hydra/config.yaml
        checkpoint_parent.parent.parent / ".hydra" / "config.yaml",  # experiment/.hydra/config.yaml
    ]
    for candidate in hydra_candidates:
        if candidate.exists():
            return candidate

    return None


def load_model_config(config_path: str | Path) -> ModelConfigBundle:
    """Load and parse a model configuration from disk."""

    path = Path(config_path)
    LOGGER.info("Using config file: %s", path)

    with path.open("r", encoding="utf-8") as handle:
        if path.suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is not available to parse configuration files.")
            config_dict = yaml.safe_load(handle)
        else:
            config_dict = json.load(handle)

    # Try to resolve Hydra defaults if present
    if OCR_MODULES_AVAILABLE and isinstance(config_dict, dict) and "defaults" in config_dict:
        try:
            from hydra import compose, initialize
            from hydra.core.global_hydra import GlobalHydra

            # Use Hydra to resolve the config with defaults
            GlobalHydra.instance().clear()
            config_dir = path.parent
            config_name = path.stem

            with initialize(config_path=str(config_dir), version_base=None):
                resolved_cfg = compose(config_name=config_name)
                config_dict = resolved_cfg.to_container() if hasattr(resolved_cfg, "to_container") else dict(resolved_cfg)
                LOGGER.info("Resolved Hydra config with defaults: %s", path)

        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to resolve Hydra config %s, using raw config: %s", path, exc)

    config_container = DictConfig(config_dict) if OCR_MODULES_AVAILABLE else config_dict

    preprocess_settings = _extract_preprocess_settings(config_container)
    postprocess_settings = _extract_postprocess_settings(config_container)

    return ModelConfigBundle(
        raw_config=config_container,
        preprocess=preprocess_settings,
        postprocess=postprocess_settings,
    )


def _extract_preprocess_settings(config: Any) -> PreprocessSettings:
    image_size = DEFAULT_IMAGE_SIZE
    mean = DEFAULT_NORMALIZE_MEAN.copy()
    std = DEFAULT_NORMALIZE_STD.copy()

    preprocessing = _get_attr(config, "preprocessing")
    if preprocessing and (target_size := _coerce_tuple(_get_attr(preprocessing, "target_size"))):
        image_size = target_size
    elif transforms_section := _get_attr(config, "transforms"):
        transform_key = "predict_transform" if _has_attr(transforms_section, "predict_transform") else "test_transform"
        if transform_config := _get_attr(transforms_section, transform_key):
            transforms_list = _get_attr(transform_config, "transforms") or []
            for transform in transforms_list:
                max_size = _get_attr(transform, "max_size")
                min_width = _get_attr(transform, "min_width")
                min_height = _get_attr(transform, "min_height")
                if max_size:
                    image_size = (int(max_size), int(max_size))
                    break
                if min_width and min_height:
                    image_size = (int(min_width), int(min_height))
                    break

            for transform in transforms_list:
                mean_candidate = _get_attr(transform, "mean")
                std_candidate = _get_attr(transform, "std")
                if mean_candidate and std_candidate:
                    mean = [float(value) for value in _as_sequence(mean_candidate)]
                    std = [float(value) for value in _as_sequence(std_candidate)]
                    break

    normalization = NormalizationSettings(mean=mean, std=std)
    return PreprocessSettings(image_size=image_size, normalization=normalization)


def _extract_postprocess_settings(config: Any) -> PostprocessSettings:
    binarization = DEFAULT_BINARIZATION_THRESH
    box_thresh = DEFAULT_BOX_THRESH
    max_candidates = DEFAULT_MAX_CANDIDATES

    head_config = _get_attr(_get_attr(config, "model"), "head")
    postprocess = _get_attr(head_config, "postprocess") if head_config else None
    if postprocess:
        thresh = _get_attr(postprocess, "thresh")
        box = _get_attr(postprocess, "box_thresh")
        max_cands = _get_attr(postprocess, "max_candidates")
        if thresh is not None:
            binarization = float(thresh)
        if box is not None:
            box_thresh = float(box)
        if max_cands is not None:
            max_candidates = int(max_cands)

    return PostprocessSettings(
        binarization_thresh=binarization,
        box_thresh=box_thresh,
        max_candidates=max_candidates,
        min_detection_size=DEFAULT_MIN_DETECTION_SIZE,
    )


def _has_attr(obj: Any, attr: str) -> bool:
    return hasattr(obj, attr) or (isinstance(obj, dict) and attr in obj)


def _get_attr(obj: Any, attr: str, default: Any | None = None) -> Any:
    if obj is None:
        return default
    if hasattr(obj, attr):
        return getattr(obj, attr)
    return obj.get(attr, default) if isinstance(obj, dict) else default


def _coerce_tuple(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    sequence = _as_sequence(value)
    return (int(sequence[0]), int(sequence[1])) if len(sequence) >= 2 else None


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, list | tuple):
        return value
    if hasattr(value, "__iter__") and not isinstance(value, str | bytes):
        return list(value)
    return [value]
