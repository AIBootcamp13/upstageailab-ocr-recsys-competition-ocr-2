from __future__ import annotations

"""Legacy checkpoint catalog service (V1).

This module provides backward-compatible catalog building for the UI.
Internally uses the V2 catalog system for improved performance.

DEPRECATION NOTICE:
    This module is maintained for backward compatibility only.
    New code should use: ui.apps.inference.services.checkpoint.build_catalog()

Performance:
    - With .metadata.yaml files: 40-100x faster than legacy
    - Wandb fallback: ~10-20x faster (with caching)
    - Legacy mode: Same as before (slow)

Feature Flags:
    - CHECKPOINT_CATALOG_USE_V2: Enable V2 catalog (default: True)
      Set to "0" or "false" to disable V2 and use legacy implementation
"""

import json
import logging
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..models.checkpoint import CheckpointInfo, CheckpointMetadata, DecoderSignature, HeadSignature
from ..models.config import PathConfig
from .checkpoint import CheckpointCatalogBuilder, CheckpointCatalogEntry
from .schema_validator import ModelCompatibilitySchema, load_schema

LOGGER = logging.getLogger(__name__)

# Feature flag: Enable V2 catalog system (default: True)
# Set CHECKPOINT_CATALOG_USE_V2=0 or CHECKPOINT_CATALOG_USE_V2=false to disable
_USE_V2_CATALOG = os.getenv("CHECKPOINT_CATALOG_USE_V2", "1").lower() not in ("0", "false", "no")

_EPOCH_PATTERN = re.compile(r"epoch[=\-_](?P<epoch>\d+)")
DEFAULT_OUTPUTS_RELATIVE_PATH = Path("outputs")
_METADATA_SUFFIXES = (".metadata.json", ".metadata.yaml")
_CONFIG_SUFFIXES = (".config.json", ".config.yaml", ".config.yml")


@dataclass(slots=True)
class CatalogOptions:
    outputs_dir: Path
    hydra_config_filenames: tuple[str, ...]

    @classmethod
    def from_paths(cls, paths: PathConfig) -> CatalogOptions:
        outputs_dir = paths.outputs_dir
        if not outputs_dir.is_absolute():
            outputs_dir = _discover_outputs_path(outputs_dir)
        return cls(outputs_dir=outputs_dir, hydra_config_filenames=tuple(paths.hydra_config_filenames))


def build_catalog(options: CatalogOptions, schema: ModelCompatibilitySchema | None = None) -> list[CheckpointMetadata]:
    if schema is None:
        schema = load_schema()

    if not options.outputs_dir.exists():
        LOGGER.info("Outputs directory not found at %s", options.outputs_dir)
        return []

    records: list[CheckpointMetadata] = []
    for checkpoint_path in _list_checkpoints(options):
        metadata = _collect_metadata(checkpoint_path, options)
        metadata = schema.validate(metadata)
        if metadata.epochs and metadata.epochs > 0:
            records.append(metadata)

    records.sort(key=lambda meta: (meta.architecture, meta.backbone, meta.epochs or 0, meta.checkpoint_path.name))
    return records


def build_lightweight_catalog(options: CatalogOptions) -> list[CheckpointInfo]:
    """Build lightweight checkpoint catalog.

    By default uses V2 catalog system for improved performance (40-100x faster).
    Can fall back to legacy implementation via CHECKPOINT_CATALOG_USE_V2=0.

    Performance improvements (V2):
        - Fast path (with .metadata.yaml): <10ms per checkpoint vs 2-5s legacy
        - Wandb fallback (cached): ~100-500ms vs 2-5s legacy
        - Expected speedup: 40-100x with metadata coverage

    Args:
        options: Catalog building options

    Returns:
        List of CheckpointInfo objects compatible with UI
    """
    if _USE_V2_CATALOG:
        return _build_lightweight_catalog_v2(options)
    else:
        LOGGER.warning("Using legacy catalog implementation (V2 disabled by feature flag)")
        return _build_lightweight_catalog_legacy(options)


def _build_lightweight_catalog_v2(options: CatalogOptions) -> list[CheckpointInfo]:
    """Build catalog using V2 system (fast path).

    Args:
        options: Catalog building options

    Returns:
        List of CheckpointInfo objects
    """
    LOGGER.info("Building catalog using V2 system (outputs_dir=%s)", options.outputs_dir)

    # Use V2 catalog builder
    builder = CheckpointCatalogBuilder(
        outputs_dir=options.outputs_dir,
        use_cache=True,
        use_wandb_fallback=True,
        config_filenames=options.hydra_config_filenames,
    )

    catalog = builder.build_catalog()

    # Log performance metrics
    LOGGER.info(
        "V2 Catalog built: %d entries, %.1f%% metadata coverage, %.3fs build time",
        catalog.total_count,
        catalog.metadata_coverage_percent,
        catalog.catalog_build_time_seconds,
    )

    # Convert V2 entries to legacy CheckpointInfo
    infos = [_convert_v2_entry_to_checkpoint_info(entry) for entry in catalog.entries]

    return infos


def _build_lightweight_catalog_legacy(options: CatalogOptions) -> list[CheckpointInfo]:
    """Build catalog using legacy implementation (slow path).

    This is the original implementation preserved for rollback capability.

    Args:
        options: Catalog building options

    Returns:
        List of CheckpointInfo objects
    """
    if not options.outputs_dir.exists():
        LOGGER.info("Outputs directory not found at %s", options.outputs_dir)
        return []

    infos: list[CheckpointInfo] = []
    for checkpoint_path in _list_checkpoints(options):
        info = _collect_basic_info(checkpoint_path, options)
        if info.epochs and info.epochs > 0:
            infos.append(info)

    infos.sort(key=lambda i: (i.architecture or "", i.backbone or "", i.epochs or 0, i.checkpoint_path.name))
    return infos


def _convert_v2_entry_to_checkpoint_info(entry: CheckpointCatalogEntry) -> CheckpointInfo:
    """Convert V2 CheckpointCatalogEntry to legacy CheckpointInfo.

    This adapter ensures backward compatibility with the UI while using
    the new V2 catalog system internally.

    Args:
        entry: V2 catalog entry

    Returns:
        Legacy CheckpointInfo object
    """
    return CheckpointInfo(
        checkpoint_path=entry.checkpoint_path,
        config_path=entry.config_path,
        display_name=entry.display_name,
        exp_name=entry.exp_name,
        epochs=entry.epochs,
        created_timestamp=entry.created_timestamp,
        hmean=entry.hmean,
        architecture=entry.architecture,
        backbone=entry.backbone,
        monitor=entry.monitor,
        monitor_mode=entry.monitor_mode,
    )


def _list_checkpoints(options: CatalogOptions) -> list[Path]:
    return sorted(options.outputs_dir.rglob("*.ckpt"))


def _collect_basic_info(checkpoint_path: Path, options: CatalogOptions) -> CheckpointInfo:
    raw_metadata = _load_metadata_dict(checkpoint_path)
    config_path = _resolve_config_path(checkpoint_path, options.hydra_config_filenames)
    config_data = _load_config_dict(config_path)
    hydra_config = _load_hydra_config(checkpoint_path)

    info = CheckpointInfo(checkpoint_path=checkpoint_path)
    info.config_path = config_path
    info.display_name = checkpoint_path.stem
    info.exp_name = _infer_experiment_name(checkpoint_path)
    info.epochs = _extract_epoch(raw_metadata, checkpoint_path.stem)
    info.created_timestamp = _extract_created_timestamp(raw_metadata, checkpoint_path)
    info.hmean = _extract_metric(raw_metadata, "val/hmean")

    architecture, encoder_name, _components = _resolve_model_details(raw_metadata, config_data)
    if architecture:
        info.architecture = architecture
    if encoder_name:
        info.backbone = encoder_name
    detected_arch, detected_backbone, _, _ = _infer_names_from_path(checkpoint_path)
    if not info.architecture and detected_arch:
        info.architecture = detected_arch
    if (not info.backbone or info.backbone == "unknown") and detected_backbone:
        info.backbone = detected_backbone

    trainer_cfg = _resolve_trainer_config(config_data, hydra_config)
    if info.epochs is None and trainer_cfg:
        info.epochs = _maybe_int(trainer_cfg.get("max_epochs"))

    monitor, mode, _save_top_k = _extract_checkpointing_settings(raw_metadata)
    info.monitor = monitor
    info.monitor_mode = mode
    if info.monitor is None or info.monitor_mode is None:
        cb_monitor, cb_mode, _ = _extract_checkpointing_from_config(config_data, hydra_config)
        info.monitor = info.monitor or cb_monitor
        info.monitor_mode = info.monitor_mode or cb_mode

    checkpoint_data: dict[str, Any] | None = None
    if info.epochs is None or info.hmean is None or info.monitor is None:
        checkpoint_data = _load_checkpoint(checkpoint_path)
        if checkpoint_data:
            train_epoch, _train_global_step = _extract_training_from_checkpoint(checkpoint_data)
            if info.epochs is None:
                info.epochs = train_epoch
            if info.hmean is None:
                info.hmean = _extract_cleval_metric(checkpoint_data, "hmean")
            if info.monitor is None or info.monitor_mode is None:
                ckpt_monitor, ckpt_mode, _ = _extract_checkpointing_from_checkpoint(checkpoint_data)
                info.monitor = info.monitor or ckpt_monitor
                info.monitor_mode = info.monitor_mode or ckpt_mode

    metadata_for_resolution = CheckpointMetadata(checkpoint_path=checkpoint_path)
    metadata_for_resolution.config_path = config_path
    if info.architecture:
        metadata_for_resolution.architecture = info.architecture
    if info.backbone:
        metadata_for_resolution.backbone = info.backbone
        metadata_for_resolution.encoder_name = info.backbone
    metadata_for_resolution.exp_name = info.exp_name
    metadata_for_resolution.epochs = info.epochs
    metadata_for_resolution.hmean = info.hmean

    resolved_path, _ = _ensure_resolved_config(
        checkpoint_path,
        metadata_for_resolution,
        config_data,
        checkpoint_data,
    )
    if resolved_path:
        info.config_path = resolved_path

    if metadata_for_resolution.architecture:
        info.architecture = metadata_for_resolution.architecture
    if metadata_for_resolution.backbone:
        info.backbone = metadata_for_resolution.backbone

    return info


def _collect_metadata(checkpoint_path: Path, options: CatalogOptions) -> CheckpointMetadata:
    raw_metadata = _load_metadata_dict(checkpoint_path)
    config_path = _resolve_config_path(checkpoint_path, options.hydra_config_filenames)
    config_data = _load_config_dict(config_path)
    hydra_config = _load_hydra_config(checkpoint_path)
    trainer_cfg = _resolve_trainer_config(config_data, hydra_config)

    metadata = CheckpointMetadata(checkpoint_path=checkpoint_path)
    metadata.config_path = config_path
    metadata.display_name = checkpoint_path.stem
    metadata.exp_name = _infer_experiment_name(checkpoint_path)
    metadata.epochs = _extract_epoch(raw_metadata, checkpoint_path.stem)
    metadata.created_timestamp = _extract_created_timestamp(raw_metadata, checkpoint_path)

    architecture, encoder_name, components = _resolve_model_details(raw_metadata, config_data)
    if architecture:
        metadata.architecture = architecture
    if encoder_name:
        metadata.encoder_name = encoder_name
        metadata.backbone = encoder_name

    model_cfg = config_data.get("model") if isinstance(config_data, dict) else None

    metadata.decoder = _build_decoder_signature(components, model_cfg)
    metadata.head = _build_head_signature(components, model_cfg)

    metadata.validation_loss = _extract_metric(raw_metadata, "val/loss")
    metadata.recall = _extract_metric(raw_metadata, "val/recall")
    metadata.hmean = _extract_metric(raw_metadata, "val/hmean")
    metadata.precision = _extract_metric(raw_metadata, "val/precision")
    metadata.metrics = _extract_metrics(raw_metadata)

    training_epoch, global_step, training_phase = _extract_training_info(raw_metadata)
    metadata.training_epoch = metadata.training_epoch or training_epoch
    metadata.global_step = metadata.global_step or global_step
    metadata.training_phase = metadata.training_phase or training_phase

    monitor, mode, save_top_k = _extract_checkpointing_settings(raw_metadata)
    metadata.monitor = metadata.monitor or monitor
    metadata.monitor_mode = metadata.monitor_mode or mode
    metadata.save_top_k = metadata.save_top_k or save_top_k

    if metadata.monitor is None or metadata.monitor_mode is None or metadata.save_top_k is None:
        cb_monitor, cb_mode, cb_top_k = _extract_checkpointing_from_config(config_data, hydra_config)
        metadata.monitor = metadata.monitor or cb_monitor
        metadata.monitor_mode = metadata.monitor_mode or cb_mode
        metadata.save_top_k = metadata.save_top_k or cb_top_k

    if metadata.epochs is None and trainer_cfg:
        metadata.epochs = _maybe_int(trainer_cfg.get("max_epochs"))

    if metadata.training_epoch is None and trainer_cfg:
        metadata.training_epoch = _maybe_int(trainer_cfg.get("max_epochs"))

    checkpoint_data: dict[str, Any] | None = None
    if (
        metadata.validation_loss is None
        or metadata.hmean is None
        or metadata.recall is None
        or metadata.precision is None
        or metadata.monitor is None
        or metadata.monitor_mode is None
        or metadata.save_top_k is None
        or metadata.training_epoch is None
        or metadata.global_step is None
    ):
        checkpoint_data = _load_checkpoint(checkpoint_path)

    if checkpoint_data:
        if metadata.training_epoch is None:
            train_epoch, global_step = _extract_training_from_checkpoint(checkpoint_data)
            metadata.training_epoch = metadata.training_epoch or train_epoch
            metadata.global_step = metadata.global_step or global_step
        if metadata.metrics:
            metrics_from_checkpoint = _extract_checkpoint_metrics(checkpoint_data)
            metadata.metrics.update(metrics_from_checkpoint)
        else:
            metadata.metrics = _extract_checkpoint_metrics(checkpoint_data)

        if metadata.hmean is None:
            metadata.hmean = _extract_cleval_metric(checkpoint_data, "hmean")
        if metadata.recall is None:
            metadata.recall = _extract_cleval_metric(checkpoint_data, "recall")
        if metadata.precision is None:
            metadata.precision = _extract_cleval_metric(checkpoint_data, "precision")

        if metadata.validation_loss is None:
            metadata.validation_loss = _extract_validation_metric(checkpoint_data)

        if metadata.monitor is None or metadata.monitor_mode is None or metadata.save_top_k is None:
            ckpt_monitor, ckpt_mode, ckpt_top_k = _extract_checkpointing_from_checkpoint(checkpoint_data)
            metadata.monitor = metadata.monitor or ckpt_monitor
            metadata.monitor_mode = metadata.monitor_mode or ckpt_mode
            metadata.save_top_k = metadata.save_top_k or ckpt_top_k

    resolved_path, resolved_config = _ensure_resolved_config(checkpoint_path, metadata, config_data, checkpoint_data)
    if resolved_path is not None:
        metadata.config_path = resolved_path
    if resolved_config is not None:
        config_data = resolved_config

    if raw_metadata is None and config_data is None:
        metadata.issues.append("No metadata or config sidecar found; partial fields inferred from filenames.")

    return metadata


def _load_metadata_dict(checkpoint_path: Path) -> dict[str, Any] | None:
    for suffix in _METADATA_SUFFIXES:
        candidate = checkpoint_path.with_suffix(suffix)
        if candidate.exists():
            data = _read_structured_file(candidate)
            if isinstance(data, dict):
                return data
    return None


def _load_config_dict(config_path: Path | None) -> dict[str, Any] | None:
    if config_path is None or not config_path.exists():
        return None

    data = _read_structured_file(config_path)
    if isinstance(data, dict):
        return data
    return None


def _load_hydra_config(checkpoint_path: Path) -> dict[str, Any] | None:
    candidates = [
        checkpoint_path.parent.parent / ".hydra" / "config.yaml",
        checkpoint_path.parent.parent.parent / ".hydra" / "config.yaml",
    ]

    for candidate in candidates:
        if candidate.exists():
            data = _read_structured_file(candidate)
            if isinstance(data, dict):
                return data
    return None


def _read_structured_file(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fp:
            if path.suffix in {".yaml", ".yml"}:
                return yaml.safe_load(fp)
            return json.load(fp)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Failed to parse %s: %s", path, exc)
        return None


def _resolve_config_path(checkpoint_path: Path, filenames: Iterable[str]) -> Path | None:
    for suffix in _CONFIG_SUFFIXES:
        sidecar = checkpoint_path.with_suffix(suffix)
        if sidecar.exists():
            return sidecar

    search_dirs: list[Path] = [checkpoint_path.parent, checkpoint_path.parent.parent]
    search_dirs.extend(
        [
            checkpoint_path.parent.parent / ".hydra",
            checkpoint_path.parent.parent.parent / ".hydra",
        ]
    )

    project_root = _discover_project_root(checkpoint_path)
    if project_root:
        search_dirs.append(project_root / "configs")

    for directory in search_dirs:
        if not directory.exists():
            continue
        for filename in filenames:
            candidate = directory / filename
            if candidate.exists():
                return candidate

    return None


def _discover_outputs_path(relative_path: Path) -> Path:
    relative_path = relative_path if relative_path != Path(".") else DEFAULT_OUTPUTS_RELATIVE_PATH
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Outputs directory not found relative to {__file__}")


def _discover_project_root(checkpoint_path: Path) -> Path | None:
    for parent in checkpoint_path.parents:
        if (parent / "pyproject.toml").exists() or (parent / "setup.cfg").exists() or (parent / "setup.py").exists():
            return parent
    return None


def _infer_experiment_name(checkpoint_path: Path) -> str | None:
    for parent in checkpoint_path.parents:
        if parent.name == "checkpoints":
            grandparent = parent.parent
            return grandparent.name if grandparent.name else None
    return None


def _extract_epoch(metadata: dict[str, Any] | None, filename: str) -> int | None:
    if metadata:
        training = metadata.get("training")
        if isinstance(training, dict):
            epoch = _maybe_int(training.get("epoch"))
            if epoch is not None:
                return epoch

    if filename.startswith("best"):
        return 999
    if filename.startswith("last"):
        return 998

    if match := _EPOCH_PATTERN.search(filename):
        try:
            return int(match.group("epoch"))
        except ValueError:
            return None
    return None


def _extract_created_timestamp(metadata: dict[str, Any] | None, checkpoint_path: Path) -> str | None:
    if metadata:
        created_at = metadata.get("created_at")
        if isinstance(created_at, str):
            return created_at

    try:
        stat = checkpoint_path.parent.stat()
        timestamp = getattr(stat, "st_birthtime", stat.st_mtime)
        return datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M")
    except (OSError, ValueError):
        return None


def _extract_metric(metadata: dict[str, Any] | None, key: str) -> float | None:
    if not metadata:
        return None
    metrics = metadata.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if value is None and "/" in key:
        fallback_key = key.split("/")[-1]
        value = metrics.get(fallback_key)
    return _maybe_float(value)


def _extract_metrics(metadata: dict[str, Any] | None) -> dict[str, float]:
    if not metadata:
        return {}
    metrics = metadata.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    plain = _to_plain_data(metrics)
    collected: dict[str, float] = {}

    def _flatten(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_key = f"{prefix}/{key}" if prefix else str(key)
                _flatten(child_key, child)
        else:
            numeric = _maybe_float(value)
            if numeric is not None:
                label = prefix if prefix else "value"
                collected[label] = numeric

    _flatten("", plain)
    return collected


def _extract_training_info(metadata: dict[str, Any] | None) -> tuple[int | None, int | None, str | None]:
    if not metadata:
        return None, None, None
    training_block = metadata.get("training")
    if not isinstance(training_block, dict):
        return None, None, None
    epoch = _maybe_int(training_block.get("epoch"))
    global_step = _maybe_int(training_block.get("global_step"))
    phase = training_block.get("training_phase") or training_block.get("phase")
    phase_value = phase if isinstance(phase, str) else None
    return epoch, global_step, phase_value


def _extract_checkpointing_settings(metadata: dict[str, Any] | None) -> tuple[str | None, str | None, int | None]:
    if not metadata:
        return None, None, None
    config_block = metadata.get("config")
    if not isinstance(config_block, dict):
        return None, None, None
    monitor = config_block.get("monitor")
    mode = config_block.get("mode")
    save_top_k = _maybe_int(config_block.get("save_top_k"))
    return (
        monitor if isinstance(monitor, str) else None,
        mode if isinstance(mode, str) else None,
        save_top_k,
    )


def _extract_checkpointing_from_config(
    config_data: dict[str, Any] | None,
    hydra_config: dict[str, Any] | None,
) -> tuple[str | None, str | None, int | None]:
    for source in (config_data, hydra_config):
        if not isinstance(source, dict):
            continue
        callbacks = source.get("callbacks")
        if not isinstance(callbacks, dict):
            continue
        for callback_config in callbacks.values():
            if not isinstance(callback_config, dict):
                continue
            monitor = callback_config.get("monitor")
            mode = callback_config.get("mode")
            save_top_k = _maybe_int(callback_config.get("save_top_k"))
            if monitor or mode or save_top_k is not None:
                return (
                    monitor if isinstance(monitor, str) else None,
                    mode if isinstance(mode, str) else None,
                    save_top_k,
                )
    return None, None, None


def _resolve_model_details(
    metadata: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> tuple[str | None, str | None, dict[str, Any]]:
    architecture = None
    encoder_name = None
    components: dict[str, Any] = {}

    if metadata:
        model_block = metadata.get("model")
        if isinstance(model_block, dict):
            architecture = model_block.get("architecture") or architecture
            encoder_name = model_block.get("encoder") or encoder_name
            raw_components = model_block.get("components")
            if isinstance(raw_components, dict):
                components = _to_plain_data(raw_components)

    if not components and config:
        model_cfg = config.get("model") if isinstance(config, dict) else None
        if isinstance(model_cfg, dict):
            architecture = architecture or model_cfg.get("architectures") or model_cfg.get("architecture_name")
            encoder_cfg = model_cfg.get("encoder")
            if isinstance(encoder_cfg, dict):
                encoder_name = encoder_name or encoder_cfg.get("model_name") or encoder_cfg.get("name")
            overrides = model_cfg.get("component_overrides")
            if isinstance(overrides, dict):
                components = _to_plain_data(overrides)

    return architecture, encoder_name, components


def _resolve_trainer_config(config_data: dict[str, Any] | None, hydra_config: dict[str, Any] | None) -> dict[str, Any] | None:
    if isinstance(config_data, dict):
        trainer_cfg = config_data.get("trainer")
        if isinstance(trainer_cfg, dict):
            return trainer_cfg
    if isinstance(hydra_config, dict):
        trainer_cfg = hydra_config.get("trainer")
        if isinstance(trainer_cfg, dict):
            return trainer_cfg
    return None


def _build_decoder_signature(components: dict[str, Any], model_cfg: dict[str, Any] | None) -> DecoderSignature:
    signature = DecoderSignature()
    decoder_cfg = components.get("decoder") if isinstance(components, dict) else None
    if isinstance(decoder_cfg, dict):
        params = decoder_cfg.get("params")
        if isinstance(params, dict):
            in_channels = params.get("in_channels")
            if isinstance(in_channels, list | tuple):
                cleaned = [value for value in (_maybe_int(item) for item in in_channels) if value is not None]
                signature.in_channels = cleaned
            signature.inner_channels = _maybe_int(params.get("inner_channels"))
            out_val = params.get("output_channels") or params.get("out_channels")
            signature.output_channels = _maybe_int(out_val)
        else:
            signature.inner_channels = _maybe_int(decoder_cfg.get("inner_channels"))
            out_val = decoder_cfg.get("output_channels") or decoder_cfg.get("out_channels")
            signature.output_channels = _maybe_int(out_val)
    if isinstance(model_cfg, dict):
        base_decoder = model_cfg.get("decoder")
        if isinstance(base_decoder, dict):
            if not signature.in_channels:
                in_channels = base_decoder.get("in_channels")
                if isinstance(in_channels, list | tuple):
                    signature.in_channels = [value for value in (_maybe_int(item) for item in in_channels) if value is not None]
            if signature.inner_channels is None:
                signature.inner_channels = _maybe_int(base_decoder.get("inner_channels"))
            if signature.output_channels is None:
                out_val = base_decoder.get("output_channels") or base_decoder.get("out_channels")
                signature.output_channels = _maybe_int(out_val)
    return signature


def _build_head_signature(components: dict[str, Any], model_cfg: dict[str, Any] | None) -> HeadSignature:
    signature = HeadSignature()
    head_cfg = components.get("head") if isinstance(components, dict) else None
    if isinstance(head_cfg, dict):
        params = head_cfg.get("params")
        if isinstance(params, dict):
            signature.in_channels = _maybe_int(params.get("in_channels"))
        else:
            signature.in_channels = _maybe_int(head_cfg.get("in_channels"))
    if signature.in_channels is None and isinstance(model_cfg, dict):
        base_head = model_cfg.get("head")
        if isinstance(base_head, dict):
            signature.in_channels = _maybe_int(base_head.get("in_channels"))
    return signature


def _to_plain_data(value: Any) -> Any:
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf  # type: ignore

        if isinstance(value, DictConfig | ListConfig):
            return _to_plain_data(OmegaConf.to_container(value, resolve=True))
    except Exception:  # pragma: no cover - optional dependency
        pass

    if isinstance(value, dict):
        return {str(key): _to_plain_data(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(val) for val in value]
    if isinstance(value, tuple):
        return [_to_plain_data(val) for val in value]
    if isinstance(value, set):
        return [_to_plain_data(val) for val in sorted(value)]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - optional dependency
            return value
    return value


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _infer_names_from_path(checkpoint_path: Path) -> tuple[str | None, str | None, str | None, str | None]:
    exp_dir = checkpoint_path.parent.parent.name.lower() if len(checkpoint_path.parents) > 1 else ""
    stem = checkpoint_path.stem.lower()
    path_str = f"{checkpoint_path.as_posix().lower()} {exp_dir} {stem}"

    architecture = None
    for candidate in ("dbnetpp", "dbnet", "craft", "pan", "psenet"):
        if candidate in path_str:
            architecture = candidate
            break

    backbone = None
    encoder_patterns = (
        "mobilenetv3_small_050",
        "mobilenetv3_small_075",
        "mobilenetv3_large_100",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "efficientnet_b0",
        "efficientnet_b3",
        "efficientnet_v2_s",
    )
    for candidate in encoder_patterns:
        if candidate in path_str:
            backbone = candidate
            break

    decoder_name = None
    decoder_patterns = {
        "pan_decoder": "pan_decoder",
        "fpn_decoder": "fpn_decoder",
        "unet_decoder": "unet",
        "unet": "unet",
        "craft_decoder": "craft_decoder",
    }
    for token, name in decoder_patterns.items():
        if token in path_str:
            decoder_name = name
            break

    head_name = None
    head_patterns = {
        "db_head": "db_head",
        "craft_head": "craft_head",
    }
    for token, name in head_patterns.items():
        if token in path_str:
            head_name = name
            break

    return architecture, backbone, decoder_name, head_name


def _extract_state_signatures_from_checkpoint(
    checkpoint_data: dict[str, Any],
) -> tuple[str | None, DecoderSignature, str | None, HeadSignature]:
    state_dict = None
    if isinstance(checkpoint_data, dict):
        state_dict = checkpoint_data.get("state_dict") or checkpoint_data.get("model_state_dict")
        if state_dict is None and all(isinstance(value, int | float) for value in checkpoint_data.values()):
            state_dict = checkpoint_data

    if not isinstance(state_dict, dict):
        return None, DecoderSignature(), None, HeadSignature()

    decoder_sig = DecoderSignature()
    head_sig = HeadSignature()

    prefix = ""
    if any(key.startswith("model.") for key in state_dict):
        prefix = "model."
    decoder_prefix = f"{prefix}decoder."
    head_prefix = f"{prefix}head."

    decoder_name = None
    if any(key.startswith(f"{decoder_prefix}bottom_up") for key in state_dict):
        decoder_name = "pan_decoder"
    elif any(key.startswith(f"{decoder_prefix}fusion") for key in state_dict):
        decoder_name = "fpn_decoder"
    elif any(key.startswith(f"{decoder_prefix}inners") for key in state_dict):
        decoder_name = "unet"

    if decoder_name == "pan_decoder":
        pan_keys = [
            f"{decoder_prefix}bottom_up.0.depthwise.weight",
            f"{decoder_prefix}bottom_up.0.pointwise.weight",
            f"{decoder_prefix}bottom_up.0.0.weight",
        ]
        for key in pan_keys:
            weight = state_dict.get(key)
            if weight is None:
                continue
            try:
                shape = tuple(weight.shape)
            except AttributeError:
                continue
            decoder_sig.output_channels = int(shape[0])
            if len(shape) > 1:
                decoder_sig.inner_channels = int(shape[1])
            break

        lateral_channels: list[int] = []
        index = 0
        while True:
            lateral_key = f"{decoder_prefix}lateral_convs.{index}.0.weight"
            weight = state_dict.get(lateral_key)
            if weight is None:
                break
            try:
                lateral_channels.append(int(weight.shape[1]))
            except AttributeError:
                break
            index += 1
        if lateral_channels:
            decoder_sig.in_channels = lateral_channels

    elif decoder_name == "fpn_decoder":
        fusion_key = f"{decoder_prefix}fusion.0.weight"
        weight = state_dict.get(fusion_key)
        if weight is not None:
            try:
                decoder_sig.output_channels = int(weight.shape[0])
                if len(weight.shape) > 1:
                    decoder_sig.inner_channels = int(weight.shape[1]) // 4 if weight.shape[1] >= 4 else int(weight.shape[1])
            except AttributeError:
                pass
        fpn_lateral_channels: list[int] = []
        index = 0
        while True:
            lateral_key = f"{decoder_prefix}lateral_convs.{index}.0.weight"
            weight = state_dict.get(lateral_key)
            if weight is None:
                break
            try:
                fpn_lateral_channels.append(int(weight.shape[1]))
            except AttributeError:
                break
            index += 1
        if fpn_lateral_channels:
            decoder_sig.in_channels = fpn_lateral_channels

    elif decoder_name == "unet":
        output_key = f"{decoder_prefix}outers.0.weight"
        weight = state_dict.get(output_key)
        if weight is None:
            weight = state_dict.get(f"{decoder_prefix}outers.0.0.weight")
        if weight is not None:
            try:
                decoder_sig.output_channels = int(weight.shape[0])
            except AttributeError:
                pass
        inner_channels: list[int] = []
        index = 0
        while True:
            inner_key = f"{decoder_prefix}inners.{index}.weight"
            weight = state_dict.get(inner_key)
            if weight is None:
                break
            try:
                inner_channels.append(int(weight.shape[1]))
            except AttributeError:
                break
            index += 1
        if inner_channels:
            decoder_sig.in_channels = inner_channels

    # Head signature
    head_name = None
    if any(key.startswith(f"{head_prefix}binarize") for key in state_dict):
        head_name = "db_head"
    elif any(key.startswith(f"{head_prefix}craft") for key in state_dict):
        head_name = "craft_head"

    for key in (
        f"{head_prefix}binarize.0.weight",
        f"{head_prefix}0.weight",
    ):
        weight = state_dict.get(key)
        if weight is not None:
            try:
                if len(weight.shape) > 1:
                    head_sig.in_channels = int(weight.shape[1])
            except AttributeError:
                pass
            break

    return head_name if head_name else None, decoder_sig, decoder_name, head_sig


def _infer_encoder_from_checkpoint(checkpoint_data: dict[str, Any]) -> str | None:
    state_dict = None
    if isinstance(checkpoint_data, dict):
        state_dict = checkpoint_data.get("state_dict") or checkpoint_data.get("model_state_dict")
        if state_dict is None and all(isinstance(value, int | float) for value in checkpoint_data.values()):
            state_dict = checkpoint_data

    if not isinstance(state_dict, dict):
        return None

    keys = state_dict.keys()

    def _get_shape(key: str) -> tuple[int, ...] | None:
        weight = state_dict.get(key)
        if weight is None:
            return None
        try:
            return tuple(int(dim) for dim in weight.shape)
        except AttributeError:
            return None

    conv_stem_key = next((key for key in keys if "encoder.model.conv_stem.weight" in key), None)
    if conv_stem_key:
        shape = _get_shape(conv_stem_key)
        if shape:
            out_channels = shape[0]
            if out_channels <= 16:
                return "mobilenetv3_small_050"
            if out_channels <= 24:
                return "mobilenetv3_small_075"
            return "mobilenetv3_large_100"
        return "mobilenetv3_small_050"

    features_key = next((key for key in keys if "encoder.model.features.0.0.weight" in key), None)
    if features_key:
        shape = _get_shape(features_key)
        if shape and shape[0] <= 40:
            return "efficientnet_b0"
        return "efficientnet_b3"

    layer_key = next(
        (
            key
            for key in keys
            if "encoder.model.layer3.0.conv1.weight" in key
            or "encoder.model.layer2.0.conv1.weight" in key
            or "encoder.model.layer1.0.conv1.weight" in key
        ),
        None,
    )
    if layer_key:
        weight_shape = _get_shape(layer_key)
        if weight_shape and weight_shape[0] >= 256:
            return "resnet50"
        return "resnet18"

    return None


def _build_config_dict(
    architecture: str | None,
    encoder_name: str | None,
    decoder_name: str | None,
    head_name: str | None,
    decoder_sig: DecoderSignature,
    head_sig: HeadSignature,
) -> dict[str, Any]:
    architecture_name = architecture or "dbnet"
    encoder_model = encoder_name or "resnet18"

    model_cfg: dict[str, Any] = {
        "_target_": "ocr.models.architecture.OCRModel",
        "architectures": architecture_name,
        "encoder": {
            "model_name": encoder_model,
        },
        "component_overrides": {},
    }

    component_overrides: dict[str, Any] = model_cfg["component_overrides"]

    if decoder_name:
        decoder_cfg: dict[str, Any] = {"name": decoder_name}
        decoder_params: dict[str, Any] = {}
        if decoder_sig.in_channels:
            decoder_params["in_channels"] = decoder_sig.in_channels
        if decoder_sig.inner_channels is not None:
            decoder_params["inner_channels"] = decoder_sig.inner_channels
        if decoder_sig.output_channels is not None:
            decoder_params["output_channels"] = decoder_sig.output_channels
            decoder_params.setdefault("out_channels", decoder_sig.output_channels)
        if decoder_params:
            decoder_cfg["params"] = decoder_params
        component_overrides["decoder"] = decoder_cfg

    if head_name:
        head_cfg: dict[str, Any] = {"name": head_name}
        head_params: dict[str, Any] = {}
        if head_sig.in_channels is not None:
            head_params["in_channels"] = head_sig.in_channels
        if head_params:
            head_params.setdefault(
                "postprocess",
                {
                    "box_thresh": 0.3,
                    "max_candidates": 300,
                    "thresh": 0.2,
                    "use_polygon": True,
                },
            )
            head_cfg["params"] = head_params
        component_overrides["head"] = head_cfg

    component_overrides.setdefault("loss", {"name": "db_loss"})

    return {"model": model_cfg}


def _ensure_resolved_config(
    checkpoint_path: Path,
    metadata: CheckpointMetadata,
    config_data: dict[str, Any] | None,
    checkpoint_data: dict[str, Any] | None,
) -> tuple[Path | None, dict[str, Any] | None]:
    detected_arch, detected_backbone, detected_decoder, detected_head = _infer_names_from_path(checkpoint_path)

    if detected_arch and metadata.architecture not in (detected_arch, None):
        metadata.architecture = detected_arch
    elif metadata.architecture in (None, "unknown", "custom") and detected_arch:
        metadata.architecture = detected_arch

    if detected_backbone and metadata.backbone not in (detected_backbone, None):
        metadata.backbone = detected_backbone
        metadata.encoder_name = detected_backbone
    elif metadata.backbone in (None, "unknown") and detected_backbone:
        metadata.backbone = detected_backbone
        metadata.encoder_name = detected_backbone
    elif metadata.encoder_name in (None, "unknown") and detected_backbone:
        metadata.encoder_name = detected_backbone

    if checkpoint_data is None:
        checkpoint_data = _load_checkpoint(checkpoint_path)

    encoder_from_state = _infer_encoder_from_checkpoint(checkpoint_data or {})
    if encoder_from_state and metadata.backbone != encoder_from_state:
        metadata.backbone = encoder_from_state
        metadata.encoder_name = encoder_from_state

    head_from_state, decoder_sig, decoder_from_state, head_sig = _extract_state_signatures_from_checkpoint(checkpoint_data or {})

    decoder_name = detected_decoder or decoder_from_state
    head_name = detected_head or head_from_state

    # Update metadata signatures if not already set
    if not metadata.decoder.in_channels and decoder_sig.in_channels:
        metadata.decoder.in_channels = decoder_sig.in_channels
    if metadata.decoder.inner_channels is None and decoder_sig.inner_channels is not None:
        metadata.decoder.inner_channels = decoder_sig.inner_channels
    if metadata.decoder.output_channels is None and decoder_sig.output_channels is not None:
        metadata.decoder.output_channels = decoder_sig.output_channels

    if metadata.head.in_channels is None and head_sig.in_channels is not None:
        metadata.head.in_channels = head_sig.in_channels

    # Determine whether existing config matches desired settings
    matches = False
    if isinstance(config_data, dict):
        model_cfg = config_data.get("model") if isinstance(config_data.get("model"), dict) else None
        if model_cfg:
            encoder_cfg = model_cfg.get("encoder") if isinstance(model_cfg.get("encoder"), dict) else None
            overrides_cfg = model_cfg.get("component_overrides") if isinstance(model_cfg.get("component_overrides"), dict) else None
            encoder_match = True
            decoder_match = True
            if metadata.backbone and encoder_cfg and encoder_cfg.get("model_name"):
                encoder_match = encoder_cfg.get("model_name") == metadata.backbone
            if metadata.backbone and encoder_cfg and encoder_cfg.get("model_name") is None:
                encoder_match = False
            if decoder_name and overrides_cfg:
                decoder_cfg = overrides_cfg.get("decoder") if isinstance(overrides_cfg.get("decoder"), dict) else None
                if decoder_cfg and decoder_cfg.get("name"):
                    decoder_match = decoder_cfg.get("name") == decoder_name
                else:
                    decoder_match = False
            matches = encoder_match and decoder_match

    if matches:
        return metadata.config_path, config_data

    desired_config = _build_config_dict(metadata.architecture, metadata.backbone, decoder_name, head_name, decoder_sig, head_sig)

    output_path = checkpoint_path.with_suffix(".resolved.config.json")
    try:
        with output_path.open("w", encoding="utf-8") as fout:
            json.dump(desired_config, fout, indent=2)
        return output_path, desired_config
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to write resolved config for %s: %s", checkpoint_path, exc)
        return metadata.config_path, config_data


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    try:
        import torch
    except Exception:  # pragma: no cover - torch optional for catalog
        LOGGER.debug("Torch not available; cannot read checkpoint %s", checkpoint_path)
        return None

    add_safe_globals: Any = None
    try:  # pragma: no cover - optional API on newer torch versions
        from torch.serialization import add_safe_globals  # type: ignore[attr-defined,no-redef]
    except Exception:  # pragma: no cover
        pass

    if add_safe_globals is not None:
        try:
            from omegaconf.listconfig import ListConfig  # type: ignore

            add_safe_globals([ListConfig])
        except Exception:  # noqa: BLE001 - optional dependency quirks
            pass

    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Unable to load checkpoint %s: %s", checkpoint_path, exc)
        return None


def _extract_training_from_checkpoint(checkpoint_data: dict[str, Any]) -> tuple[int | None, int | None]:
    epoch = _maybe_int(checkpoint_data.get("epoch"))
    global_step = _maybe_int(checkpoint_data.get("global_step"))
    return epoch, global_step


def _extract_cleval_metric(checkpoint_data: dict[str, Any], key: str) -> float | None:
    cleval_metrics = checkpoint_data.get("cleval_metrics")
    if isinstance(cleval_metrics, dict):
        return _maybe_float(cleval_metrics.get(key))
    return None


def _extract_checkpoint_metrics(checkpoint_data: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    cleval_metrics = checkpoint_data.get("cleval_metrics")
    if isinstance(cleval_metrics, dict):
        for key, value in cleval_metrics.items():
            numeric = _maybe_float(value)
            if numeric is not None:
                metrics[f"cleval/{key}"] = numeric
    callbacks = checkpoint_data.get("callbacks")
    if isinstance(callbacks, dict):
        for callback in callbacks.values():
            if isinstance(callback, dict) and "best_model_score" in callback:
                monitor_name = callback.get("monitor")
                best_score = callback.get("best_model_score")
                numeric = _maybe_float(best_score)
                if numeric is not None and isinstance(monitor_name, str):
                    metrics[f"checkpoint/{monitor_name}"] = numeric
    return metrics


def _extract_validation_metric(checkpoint_data: dict[str, Any]) -> float | None:
    callbacks = checkpoint_data.get("callbacks")
    if not isinstance(callbacks, dict):
        return None
    for callback in callbacks.values():
        if not isinstance(callback, dict):
            continue
        best_model_score = callback.get("best_model_score")
        monitor = callback.get("monitor")
        if best_model_score is None or not isinstance(monitor, str):
            continue
        numeric = _maybe_float(best_model_score)
        if numeric is not None and monitor.endswith("loss"):
            return numeric
    return None


def _extract_checkpointing_from_checkpoint(checkpoint_data: dict[str, Any]) -> tuple[str | None, str | None, int | None]:
    callbacks = checkpoint_data.get("callbacks")
    if not isinstance(callbacks, dict):
        return None, None, None

    monitor = None
    mode = None
    save_top_k = None

    for callback in callbacks.values():
        if not isinstance(callback, dict):
            continue
        monitor = monitor or (callback.get("monitor") if isinstance(callback.get("monitor"), str) else None)
        mode = mode or (callback.get("mode") if isinstance(callback.get("mode"), str) else None)
        best_models = callback.get("best_k_models")
        if save_top_k is None and isinstance(best_models, dict):
            save_top_k = len(best_models)

        if monitor and mode and save_top_k is not None:
            break

    return monitor, mode, save_top_k
