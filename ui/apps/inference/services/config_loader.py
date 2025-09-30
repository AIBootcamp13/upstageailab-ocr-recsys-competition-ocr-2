from __future__ import annotations

"""Load Streamlit UI configuration from YAML.

For maintenance and refactor guidance see
``docs/ai_handbook/02_protocols/11_streamlit_maintenance_protocol.md`` and
``docs/ai_handbook/02_protocols/12_streamlit_refactoring_protocol.md``. The
values here must mirror ``configs/ui/inference.yaml`` (or the corresponding
app config) and any copy stored in ``ui_meta/``—keep those files authoritative
instead of hard-coding defaults. Pydantic models validate UI payloads only;
Hydra/OmegaConf continue to govern runtime configs.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from ..models.config import (
    AppSection,
    ModelSelectorConfig,
    NotificationConfig,
    PathConfig,
    PreprocessingConfig,
    ResultsConfig,
    SliderConfig,
    UIConfig,
    UploadConfig,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_RELATIVE_PATH = Path("configs") / "ui" / "inference.yaml"


def _discover_default_config_path() -> Path:
    """Locate the default UI config by walking up the repository tree."""

    for parent in Path(__file__).resolve().parents:
        candidate = parent / DEFAULT_CONFIG_RELATIVE_PATH
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"UI config not found relative to {__file__}")


def load_ui_config(config_path: Path | None = None) -> UIConfig:
    if config_path is None:
        config_path = _discover_default_config_path()

    if not config_path.exists():
        raise FileNotFoundError(f"UI config not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        raw_config: dict[str, Any] = yaml.safe_load(fp)

    app_section = AppSection(**raw_config.get("app", {}))
    model_selector = ModelSelectorConfig(**raw_config.get("model_selector", {}))
    upload = UploadConfig(**raw_config.get("upload", {}))
    results = ResultsConfig(**raw_config.get("results", {}))
    notifications = NotificationConfig(**raw_config.get("notifications", {}))
    path_section = dict(raw_config.get("paths", {}))
    outputs_dir = path_section.get("outputs_dir")
    if outputs_dir is not None:
        path_section["outputs_dir"] = Path(outputs_dir)
    paths = PathConfig(**path_section)

    hyperparameters_section = raw_config.get("hyperparameters", {})
    hyperparameters = {key: SliderConfig.from_mapping(key, value) for key, value in hyperparameters_section.items()}

    preprocessing_section = dict(raw_config.get("preprocessing", {}))
    target_size = preprocessing_section.get("target_size")
    if isinstance(target_size, list | tuple) and len(target_size) == 2:
        preprocessing_section["target_size"] = tuple(int(value) for value in target_size)
    preprocessing = PreprocessingConfig(**preprocessing_section)

    return UIConfig(
        app=app_section,
        model_selector=model_selector,
        hyperparameters=hyperparameters,
        upload=upload,
        results=results,
        notifications=notifications,
        paths=paths,
        preprocessing=preprocessing,
    )
