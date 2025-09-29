from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import yaml

from ..models.checkpoint import CheckpointMetadata, DecoderSignature, HeadSignature
from ..models.config import PathConfig
from .schema_validator import ModelCompatibilitySchema, load_schema

LOGGER = logging.getLogger(__name__)

_EPOCH_PATTERN = re.compile(r"epoch[=\-](?P<epoch>\d+)")
DEFAULT_OUTPUTS_RELATIVE_PATH = Path("outputs")


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

    checkpoints = []
    for ckpt_path in options.outputs_dir.rglob("*.ckpt"):
        metadata = _collect_metadata(ckpt_path, options)
        metadata = schema.validate(metadata)
        checkpoints.append(metadata)

    checkpoints.sort(key=lambda meta: (meta.architecture, meta.backbone, meta.epochs or 0, meta.checkpoint_path.name))
    return checkpoints


def _discover_outputs_path(relative_path: Path) -> Path:
    relative_path = relative_path if relative_path != Path(".") else DEFAULT_OUTPUTS_RELATIVE_PATH
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Outputs directory not found relative to {__file__}")


def _collect_metadata(checkpoint_path: Path, options: CatalogOptions) -> CheckpointMetadata:
    metadata = CheckpointMetadata(checkpoint_path=checkpoint_path)
    metadata.exp_name = checkpoint_path.parents[1].name if len(checkpoint_path.parents) > 1 else None
    metadata.display_name = checkpoint_path.stem
    metadata.epochs = _parse_epoch(checkpoint_path.name)

    config_path = _find_config_path(checkpoint_path, options.hydra_config_filenames)
    metadata.config_path = config_path

    if config_path:
        try:
            with config_path.open("r", encoding="utf-8") as fp:
                config = yaml.safe_load(fp) or {}
            model_cfg = config.get("model", {})
            metadata.architecture = _extract_architecture(model_cfg)
            metadata.encoder_name = _extract_encoder_name(model_cfg)
            metadata.backbone = metadata.encoder_name or metadata.backbone
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to parse model config at %s: %s", config_path, exc)
            metadata.issues.append(f"Unable to parse config: {exc}")

    decoder_sig, head_sig = _extract_state_signatures(checkpoint_path)
    metadata.decoder = decoder_sig
    metadata.head = head_sig

    return metadata


def _parse_epoch(filename: str) -> int | None:
    if match := _EPOCH_PATTERN.search(filename):
        try:
            return int(match.group("epoch"))
        except ValueError:
            return None
    return None


def _find_config_path(checkpoint_path: Path, filenames: Iterable[str]) -> Path | None:
    candidate_dirs = [checkpoint_path.parent, checkpoint_path.parent.parent / ".hydra"]
    for directory in candidate_dirs:
        for filename in filenames:
            candidate = directory / filename
            if candidate.exists():
                return candidate
    return None


def _extract_architecture(model_cfg: dict[str, object]) -> str:
    if arch_name := model_cfg.get("architecture_name"):
        return str(arch_name)
    target = model_cfg.get("_target_")
    return target.split(".")[-1] if isinstance(target, str) else "custom"


def _extract_encoder_name(model_cfg: dict[str, object]) -> str | None:
    encoder_cfg = model_cfg.get("encoder")
    if isinstance(encoder_cfg, dict):
        model_name = encoder_cfg.get("model_name")
        if isinstance(model_name, str):
            return model_name
    return None


@lru_cache(maxsize=64)
def _extract_state_signatures(checkpoint_path: Path) -> tuple[DecoderSignature, HeadSignature]:
    decoder_sig = DecoderSignature()
    head_sig = HeadSignature()

    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint is None:
        decoder_sig.output_channels = None
        head_sig.in_channels = None
        return decoder_sig, head_sig

    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint

    # Decoder channels
    for key in ("model.decoder.outers.0.0.weight", "model.decoder.outers.0.weight"):
        if key in state_dict:
            weight = state_dict[key]
            decoder_sig.output_channels = int(weight.shape[0])
            decoder_sig.inner_channels = int(weight.shape[1])
            break

    in_channels: list[int] = []
    index = 0
    while True:
        weight_key = f"model.decoder.inners.{index}.weight"
        if weight_key not in state_dict:
            break
        weight = state_dict[weight_key]
        in_channels.append(int(weight.shape[1]))
        index += 1
    decoder_sig.in_channels = in_channels

    # Head signature
    head_key = "model.head.binarize.0.weight"
    if head_key in state_dict:
        head_weight = state_dict[head_key]
        head_sig.in_channels = int(head_weight.shape[1])

    return decoder_sig, head_sig


def _load_checkpoint(checkpoint_path: Path) -> dict | None:
    try:
        return torch.load(checkpoint_path, map_location="cpu")
    except TypeError:
        # Older torch versions may not support weights_only parameter.
        pass
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Initial torch.load attempt failed for %s: %s", checkpoint_path, exc)

    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Unable to load checkpoint %s: %s", checkpoint_path, exc)
        return None
