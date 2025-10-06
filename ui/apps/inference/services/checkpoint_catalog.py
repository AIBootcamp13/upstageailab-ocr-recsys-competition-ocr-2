from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import torch
import yaml

from ..models.checkpoint import CheckpointMetadata, DecoderSignature, HeadSignature
from ..models.config import PathConfig
from .schema_validator import ModelCompatibilitySchema, load_schema

LOGGER = logging.getLogger(__name__)

_EPOCH_PATTERN = re.compile(r"epoch[=\-_](?P<epoch>\d+)")
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
    # Extract experiment name from the correct parent directory
    # Structure can be: outputs/{exp_name}/checkpoints/{checkpoint_file}
    # Or: outputs/{exp_name}/checkpoints/{model_name}/{checkpoint_file}
    # Find the experiment directory (parent of checkpoints)
    checkpoints_parent = None
    for parent in checkpoint_path.parents:
        if parent.name == "checkpoints":
            checkpoints_parent = parent
            break

    if checkpoints_parent and len(checkpoints_parent.parents) > 0:
        metadata.exp_name = checkpoints_parent.parents[0].name
    else:
        metadata.exp_name = None

    metadata.display_name = checkpoint_path.stem
    metadata.epochs = _parse_epoch(checkpoint_path.name)

    config_path = _find_config_path(checkpoint_path, options.hydra_config_filenames)
    metadata.config_path = config_path

    if config_path:
        try:
            with config_path.open("r", encoding="utf-8") as fp:
                config = yaml.safe_load(fp) or {}
            model_cfg = config.get("model", {})

            # Extract architecture and encoder info
            metadata.architecture = _extract_architecture(model_cfg)
            metadata.encoder_name = _extract_encoder_name(model_cfg)

            # If not found in model section, try root level (for configs that don't use model section)
            if not metadata.architecture and not metadata.encoder_name:
                metadata.architecture = _extract_architecture(config)
                metadata.encoder_name = _extract_encoder_name(config)

            # For Hydra configs with defaults, try to parse referenced preset files
            if (not metadata.architecture or not metadata.encoder_name) and "defaults" in config:
                metadata = _parse_hydra_defaults(config_path.parent, config, metadata)

        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to parse model config at %s: %s", config_path, exc)
            metadata.issues.append(f"Unable to parse config: {exc}")
            # Fallback to directory-based extraction
            metadata = _extract_from_directory_structure(checkpoint_path, metadata)

    # If no config found, or config parsing didn't give useful results, try directory-based extraction
    if not config_path or (metadata.architecture == "custom" and metadata.encoder_name is None):
        metadata = _extract_from_directory_structure(checkpoint_path, metadata)

    # Set backbone from encoder name if not already set
    if metadata.encoder_name and metadata.backbone == "unknown":
        metadata.backbone = metadata.encoder_name

    decoder_sig, head_sig = _extract_state_signatures(checkpoint_path)
    metadata.decoder = decoder_sig
    metadata.head = head_sig

    # Extract validation loss from checkpoint
    metadata.validation_loss = _extract_validation_loss(checkpoint_path)

    # Extract creation timestamp from experiment directory
    metadata.created_timestamp = _extract_creation_timestamp(checkpoint_path)

    # Extract recall and hmean (currently not available from checkpoints)
    metadata.recall = _extract_recall(checkpoint_path)
    metadata.hmean = _extract_hmean(checkpoint_path)
    metadata.precision = _extract_precision(checkpoint_path)
    return metadata


def _parse_epoch(filename: str) -> int | None:
    if match := _EPOCH_PATTERN.search(filename):
        try:
            return int(match.group("epoch"))
        except ValueError:
            return None
    return None


def _find_config_path(checkpoint_path: Path, filenames: Iterable[str]) -> Path | None:
    # Look up to 4 levels up the directory tree for config files in various locations
    candidate_dirs = [
        # Standard Hydra location
        checkpoint_path.parent,  # immediate parent
        checkpoint_path.parent.parent / ".hydra",  # grandparent/.hydra
        checkpoint_path.parent.parent.parent / ".hydra",  # great-grandparent/.hydra
        checkpoint_path.parent.parent.parent.parent / ".hydra",  # great-great-grandparent/.hydra
        # Alternative locations for older checkpoints
        checkpoint_path.parent.parent / "baseline_code" / "configs",  # grandparent/baseline_code/configs
        checkpoint_path.parent.parent.parent / "baseline_code" / "configs",  # great-grandparent/baseline_code/configs
        checkpoint_path.parent.parent.parent.parent / "baseline_code" / "configs",  # great-great-grandparent/baseline_code/configs
        # Generic configs directory
        checkpoint_path.parent.parent / "configs",  # grandparent/configs
        checkpoint_path.parent.parent.parent / "configs",  # great-grandparent/configs
        checkpoint_path.parent.parent.parent.parent / "configs",  # great-great-grandparent/configs
    ]

    # Add project-level configs directory as fallback
    project_root = _discover_project_root(checkpoint_path)
    if project_root:
        candidate_dirs.extend(
            [
                project_root / "configs",  # project/configs
            ]
        )

    for directory in candidate_dirs:
        if not directory.exists():
            continue
        for filename in filenames:
            candidate = directory / filename
            if candidate.exists():
                return candidate
    return None


def _discover_project_root(checkpoint_path: Path) -> Path | None:
    """Find the project root by looking for common project markers."""
    for parent in checkpoint_path.parents:
        if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists() or (parent / "requirements.txt").exists():
            return parent
    return None


def _parse_hydra_defaults(config_dir: Path, config: dict[str, object], metadata: CheckpointMetadata) -> CheckpointMetadata:
    """Parse Hydra defaults to find model configuration in referenced preset files."""
    try:
        defaults = config.get("defaults", [])
        if not isinstance(defaults, list):
            return metadata

        # Look for model-related preset files
        model_preset_paths = [
            config_dir / "preset" / "models" / "model_example.yaml",
            config_dir / "preset" / "models" / "encoder" / "timm_backbone.yaml",
        ]

        for preset_path in model_preset_paths:
            if preset_path.exists():
                try:
                    with preset_path.open("r", encoding="utf-8") as fp:
                        preset_config = yaml.safe_load(fp) or {}

                    # Extract model info from preset
                    model_cfg = preset_config.get("models", {})
                    if not metadata.architecture:
                        metadata.architecture = _extract_architecture(model_cfg)
                    if not metadata.encoder_name:
                        metadata.encoder_name = _extract_encoder_name(model_cfg)

                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("Failed to parse preset %s: %s", preset_path, exc)

    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Failed to parse Hydra defaults: %s", exc)

    return metadata


def _extract_from_directory_structure(checkpoint_path: Path, metadata: CheckpointMetadata) -> CheckpointMetadata:
    """Extract model information from directory names and checkpoint filenames when config is unavailable."""
    try:
        # Try to extract from checkpoint directory name
        # Examples: "craft_resnet50", "dbnetpp_resnet50", "dbnet_mobilenetv3_small_050"
        checkpoint_dir = checkpoint_path.parent.name

        # Common patterns: {architecture}_{encoder} or just {encoder}
        if "_" in checkpoint_dir:
            parts = checkpoint_dir.split("_", 1)  # Split only on first underscore
            potential_arch = parts[0].lower()
            potential_encoder = parts[1]

            # Map common architecture names
            arch_mapping = {"dbnet": "dbnet", "dbnetpp": "dbnetpp", "craft": "craft", "pan": "pan", "psenet": "psenet"}

            if potential_arch in arch_mapping:
                metadata.architecture = arch_mapping[potential_arch]
                metadata.encoder_name = potential_encoder
            else:
                # Assume the whole directory name is the encoder
                metadata.encoder_name = checkpoint_dir

        # Try to extract from parent directory name (experiment name)
        exp_dir = checkpoint_path.parents[1].name if len(checkpoint_path.parents) > 1 else ""
        if exp_dir and not metadata.architecture:
            # Look for architecture keywords in experiment name
            exp_lower = exp_dir.lower()
            for arch_name in ["dbnet", "craft", "pan", "psenet", "dbnetpp"]:
                if arch_name in exp_lower:
                    metadata.architecture = arch_name
                    break

        # Set backbone from encoder if available
        if metadata.encoder_name and not metadata.backbone:
            metadata.backbone = metadata.encoder_name

    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Failed to extract from directory structure: %s", exc)

    return metadata


def _extract_architecture(model_cfg: dict[str, object]) -> str:
    if arch_name := model_cfg.get("architecture_name"):
        return str(arch_name)
    target = model_cfg.get("_target_")
    return target.split(".")[-1] if isinstance(target, str) else "custom"


def _extract_encoder_name(model_cfg: dict[str, object]) -> str | None:
    # First try the direct encoder config
    encoder_cfg = model_cfg.get("encoder")
    if isinstance(encoder_cfg, dict):
        model_name = encoder_cfg.get("model_name")
        if isinstance(model_name, str):
            return model_name

    # Then try component_overrides.encoder
    component_overrides = model_cfg.get("component_overrides")
    if isinstance(component_overrides, dict):
        encoder_override = component_overrides.get("encoder")
        if isinstance(encoder_override, dict):
            model_name = encoder_override.get("model_name")
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

    # Try UNet-style decoder first (most common)
    decoder_found = False
    for key in ("model.decoder.outers.0.0.weight", "model.decoder.outers.0.weight"):
        if key in state_dict:
            weight = state_dict[key]
            decoder_sig.output_channels = int(weight.shape[0])
            decoder_sig.inner_channels = int(weight.shape[1])
            decoder_found = True
            break

    if not decoder_found:
        # Try PAN decoder structure
        pan_key = "model.decoder.bottom_up.0.0.weight"
        if pan_key in state_dict:
            weight = state_dict[pan_key]
            decoder_sig.output_channels = int(weight.shape[0])
            decoder_sig.inner_channels = int(weight.shape[1])
            decoder_found = True

    if not decoder_found:
        # Try FPN decoder structure (used by mobilenetv3_small_050 checkpoints)
        fpn_fusion_key = "model.decoder.fusion.0.weight"
        if fpn_fusion_key in state_dict:
            fusion_weight = state_dict[fpn_fusion_key]
            decoder_sig.output_channels = int(fusion_weight.shape[0])  # Output channels from fusion
            # For FPN, inner_channels is the output channels of lateral convs (typically 256)
            # Check first lateral conv output channels
            first_lateral_key = "model.decoder.lateral_convs.0.0.weight"
            if first_lateral_key in state_dict:
                lateral_weight = state_dict[first_lateral_key]
                decoder_sig.inner_channels = int(lateral_weight.shape[0])  # Output channels of lateral conv
            else:
                decoder_sig.inner_channels = int(fusion_weight.shape[1]) // 4  # Fallback: assume 4 inputs
            decoder_found = True

            # Extract in_channels from lateral convs
            in_channels: list[int] = []
            index = 0
            while True:
                lateral_key = f"model.decoder.lateral_convs.{index}.0.weight"
                if lateral_key not in state_dict:
                    break
                lateral_weight = state_dict[lateral_key]
                in_channels.append(int(lateral_weight.shape[1]))  # Input channels to lateral conv
                index += 1
            decoder_sig.in_channels = in_channels

    # Extract in_channels for UNet-style decoders (if not already set by FPN)
    if not decoder_sig.in_channels:
        unet_in_channels: list[int] = []
        index = 0
        while True:
            weight_key = f"model.decoder.inners.{index}.weight"
            if weight_key not in state_dict:
                break
            weight = state_dict[weight_key]
            unet_in_channels.append(int(weight.shape[1]))
            index += 1
        decoder_sig.in_channels = unet_in_channels

    # Head signature - try multiple possible head structures
    head_keys = [
        "model.head.binarize.0.weight",  # DBHead
        "model.head.0.weight",  # Other head types
    ]
    for head_key in head_keys:
        if head_key in state_dict:
            head_weight = state_dict[head_key]
            head_sig.in_channels = int(head_weight.shape[1])
            break

    return decoder_sig, head_sig


def _extract_validation_loss(checkpoint_path: Path) -> float | None:
    """Extract the best validation loss from the checkpoint."""
    checkpoint_data = _load_checkpoint(checkpoint_path)
    if checkpoint_data is None:
        return None

    # Look for validation loss in callbacks
    callbacks = checkpoint_data.get("callbacks", {})
    if not callbacks:
        return None

    # Look for ModelCheckpoint callback (it may have different names)
    for callback_key, callback_data in callbacks.items():
        if "ModelCheckpoint" in callback_key or "checkpoint" in callback_key.lower():
            if isinstance(callback_data, dict) and "best_model_score" in callback_data:
                score = callback_data["best_model_score"]
                # Handle torch.Tensor
                if hasattr(score, "item"):
                    score = score.item()
                if isinstance(score, int | float):
                    return float(score)

    return None


def _extract_precision(checkpoint_path: Path) -> float | None:
    """Extract precision score from checkpoint."""
    checkpoint_data = _load_checkpoint(checkpoint_path)
    if checkpoint_data is None:
        return None

    # Look for CLEval metrics stored in checkpoint
    cleval_metrics = checkpoint_data.get("cleval_metrics", {})
    if isinstance(cleval_metrics, dict) and "precision" in cleval_metrics:
        precision = cleval_metrics["precision"]
        if isinstance(precision, int | float):
            return float(precision)
        elif hasattr(precision, "item"):  # Handle torch.Tensor
            return float(precision.item())

    return None


def _extract_recall(checkpoint_path: Path) -> float | None:
    """Extract recall score from checkpoint."""
    checkpoint_data = _load_checkpoint(checkpoint_path)
    if checkpoint_data is None:
        return None

    # Look for CLEval metrics stored in checkpoint
    cleval_metrics = checkpoint_data.get("cleval_metrics", {})
    if isinstance(cleval_metrics, dict) and "recall" in cleval_metrics:
        recall = cleval_metrics["recall"]
        if isinstance(recall, int | float):
            return float(recall)
        elif hasattr(recall, "item"):  # Handle torch.Tensor
            return float(recall.item())

    return None


def _extract_hmean(checkpoint_path: Path) -> float | None:
    """Extract hmean (F1) score from checkpoint."""
    checkpoint_data = _load_checkpoint(checkpoint_path)
    if checkpoint_data is None:
        return None

    # Look for CLEval metrics stored in checkpoint
    cleval_metrics = checkpoint_data.get("cleval_metrics", {})
    if isinstance(cleval_metrics, dict) and "hmean" in cleval_metrics:
        hmean = cleval_metrics["hmean"]
        if isinstance(hmean, int | float):
            return float(hmean)
        elif hasattr(hmean, "item"):  # Handle torch.Tensor
            return float(hmean.item())

    return None


def _extract_creation_timestamp(experiment_dir: Path) -> str | None:
    """Extract creation timestamp from the experiment directory."""
    try:
        # Get the creation time of the experiment directory
        stat = experiment_dir.stat()
        # Use creation time if available, otherwise modification time
        timestamp = getattr(stat, "st_birthtime", stat.st_mtime)
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y%m%d_%H%M")
    except (OSError, ValueError):
        return None


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
