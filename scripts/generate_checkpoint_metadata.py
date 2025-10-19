#!/usr/bin/env python3
"""Generate metadata YAML files for existing checkpoints.

This script scans a directory for PyTorch checkpoint files and generates
corresponding .metadata.yaml files using the Checkpoint Catalog V2 schema.

Usage:
    python scripts/generate_checkpoint_metadata.py [--outputs-dir OUTPUTS_DIR] [--dry-run]

Examples:
    # Generate metadata for all checkpoints in outputs/
    python scripts/generate_checkpoint_metadata.py

    # Specify custom outputs directory
    python scripts/generate_checkpoint_metadata.py --outputs-dir /path/to/outputs

    # Dry run to preview what would be generated
    python scripts/generate_checkpoint_metadata.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def find_checkpoints(outputs_dir: Path) -> list[Path]:
    """Find all checkpoint files in outputs directory.

    Args:
        outputs_dir: Directory to search

    Returns:
        List of checkpoint paths
    """
    checkpoints = []

    for ckpt_path in outputs_dir.rglob("*.ckpt"):
        # Skip if metadata already exists
        metadata_path = ckpt_path.with_suffix(".ckpt.metadata.yaml")
        if metadata_path.exists():
            LOGGER.debug("Metadata already exists: %s", metadata_path)
            continue

        checkpoints.append(ckpt_path)

    return checkpoints


def load_checkpoint_safely(ckpt_path: Path) -> dict[str, Any] | None:
    """Load checkpoint without full deserialization.

    Args:
        ckpt_path: Path to checkpoint

    Returns:
        Checkpoint state dict, or None if failed
    """
    try:
        # Load with weights_only for security
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ckpt
    except Exception as exc:
        LOGGER.error("Failed to load checkpoint %s: %s", ckpt_path, exc)
        return None


def extract_metadata_from_checkpoint(
    ckpt_path: Path,
    outputs_dir: Path,
) -> dict[str, Any] | None:
    """Extract metadata from checkpoint file.

    Args:
        ckpt_path: Path to checkpoint file
        outputs_dir: Outputs directory for relative paths

    Returns:
        Metadata dictionary, or None if failed
    """
    # Load checkpoint
    ckpt = load_checkpoint_safely(ckpt_path)
    if ckpt is None:
        return None

    # Extract basic information
    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)

    # Try to extract experiment name from path
    # Typical structure: outputs/exp_name/checkpoints/checkpoint.ckpt
    try:
        exp_name = ckpt_path.parent.parent.name
    except Exception:
        exp_name = "unknown_experiment"

    # Extract relative checkpoint path
    try:
        checkpoint_path_str = str(ckpt_path.relative_to(outputs_dir))
    except ValueError:
        checkpoint_path_str = str(ckpt_path)

    # Look for Hydra config
    hydra_config_path = None
    hydra_config_dir = ckpt_path.parent.parent / ".hydra"
    if (hydra_config_dir / "config.yaml").exists():
        try:
            hydra_config_path = str((hydra_config_dir / "config.yaml").relative_to(outputs_dir))
        except ValueError:
            hydra_config_path = str(hydra_config_dir / "config.yaml")

    # Try to load Hydra config for additional metadata
    hydra_config = None
    if hydra_config_path:
        config_path_full = outputs_dir / hydra_config_path
        if config_path_full.exists():
            try:
                with config_path_full.open() as f:
                    hydra_config = yaml.safe_load(f)
            except Exception as exc:
                LOGGER.warning("Failed to load Hydra config: %s", exc)

    # Extract model architecture info from state dict
    state_dict = ckpt.get("state_dict", {})
    architecture = extract_architecture_from_state(state_dict, hydra_config)
    encoder_name = extract_encoder_from_state(state_dict, hydra_config)

    # Extract metrics from checkpoint
    metrics = extract_metrics_from_checkpoint(ckpt)

    # Extract Wandb run ID if available
    wandb_run_id = None
    if "wandb_run_id" in ckpt:
        wandb_run_id = ckpt["wandb_run_id"]
    elif hydra_config and "logger" in hydra_config:
        logger_cfg = hydra_config["logger"]
        if isinstance(logger_cfg, dict) and "wandb" in logger_cfg:
            wandb_cfg = logger_cfg["wandb"]
            if isinstance(wandb_cfg, dict) and "id" in wandb_cfg:
                wandb_run_id = wandb_cfg["id"]

    # Build metadata structure
    from datetime import datetime

    metadata = {
        "schema_version": "1.0",
        "checkpoint_path": checkpoint_path_str,
        "exp_name": exp_name,
        "created_at": datetime.now().isoformat(),
        "training": {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "training_phase": "training",
            "max_epochs": None,
        },
        "model": {
            "architecture": architecture,
            "encoder": {
                "model_name": encoder_name,
                "pretrained": True,
                "frozen": False,
            },
            "decoder": {
                "name": "unknown",
                "in_channels": [],
                "inner_channels": None,
                "output_channels": None,
                "params": {},
            },
            "head": {
                "name": "unknown",
                "in_channels": None,
                "params": {},
            },
            "loss": {
                "name": "unknown",
                "params": {},
            },
        },
        "metrics": metrics,
        "checkpointing": {
            "monitor": "unknown",
            "mode": "max",
            "save_top_k": 1,
            "save_last": True,
        },
        "hydra_config_path": hydra_config_path,
        "wandb_run_id": wandb_run_id,
    }

    # Enrich from Hydra config if available
    if hydra_config:
        metadata = enrich_metadata_from_hydra(metadata, hydra_config)

    return metadata


def extract_architecture_from_state(
    state_dict: dict[str, Any],
    hydra_config: dict[str, Any] | None,
) -> str:
    """Extract architecture name from state dict or config.

    Args:
        state_dict: Model state dict
        hydra_config: Hydra configuration

    Returns:
        Architecture name
    """
    # Try from Hydra config first
    if hydra_config and "model" in hydra_config:
        model_cfg = hydra_config["model"]
        if isinstance(model_cfg, dict):
            if "architecture" in model_cfg:
                return str(model_cfg["architecture"])
            if "_target_" in model_cfg:
                target = str(model_cfg["_target_"])
                # Extract class name from target
                if "." in target:
                    return target.split(".")[-1].lower()

    # Try from state dict keys
    keys = list(state_dict.keys())

    # Common patterns
    if any("fpn" in k.lower() for k in keys):
        return "fpn"
    if any("pan" in k.lower() for k in keys):
        return "pan"
    if any("dbnet" in k.lower() for k in keys):
        return "dbnet"
    if any("unet" in k.lower() for k in keys):
        return "unet"

    return "unknown"


def extract_encoder_from_state(
    state_dict: dict[str, Any],
    hydra_config: dict[str, Any] | None,
) -> str:
    """Extract encoder name from state dict or config.

    Args:
        state_dict: Model state dict
        hydra_config: Hydra configuration

    Returns:
        Encoder name
    """
    # Try from Hydra config first
    if hydra_config and "model" in hydra_config:
        model_cfg = hydra_config["model"]
        if isinstance(model_cfg, dict) and "encoder" in model_cfg:
            encoder_cfg = model_cfg["encoder"]
            if isinstance(encoder_cfg, dict):
                if "model_name" in encoder_cfg:
                    return str(encoder_cfg["model_name"])
                if "_target_" in encoder_cfg:
                    target = str(encoder_cfg["_target_"])
                    if "." in target:
                        return target.split(".")[-1].lower()

    # Try from state dict keys
    keys = list(state_dict.keys())

    # Common encoder patterns
    if any("resnet" in k.lower() for k in keys):
        if any("resnet18" in k.lower() for k in keys):
            return "resnet18"
        if any("resnet34" in k.lower() for k in keys):
            return "resnet34"
        if any("resnet50" in k.lower() for k in keys):
            return "resnet50"
        return "resnet"

    if any("mobilenet" in k.lower() for k in keys):
        if any("v3" in k.lower() for k in keys):
            if any("small" in k.lower() for k in keys):
                return "mobilenetv3_small"
            if any("large" in k.lower() for k in keys):
                return "mobilenetv3_large"
            return "mobilenetv3"
        return "mobilenet"

    return "unknown"


def extract_metrics_from_checkpoint(ckpt: dict[str, Any]) -> dict[str, Any]:
    """Extract metrics from checkpoint.

    Args:
        ckpt: Checkpoint dictionary

    Returns:
        Metrics dictionary
    """
    metrics = {
        "precision": None,
        "recall": None,
        "hmean": None,
        "validation_loss": None,
        "additional_metrics": {},
    }

    # Try to find metrics in checkpoint
    # Common locations:
    # - checkpoint["hyper_parameters"]["val_metrics"]
    # - checkpoint["callbacks"]["ModelCheckpoint"]["best_model_score"]
    # - checkpoint["val_metrics"]

    # Look in callbacks
    if "callbacks" in ckpt:
        callbacks = ckpt["callbacks"]
        if isinstance(callbacks, dict):
            for callback_name, callback_data in callbacks.items():
                if isinstance(callback_data, dict):
                    if "best_model_score" in callback_data:
                        score = callback_data["best_model_score"]
                        if isinstance(score, int | float):
                            metrics["hmean"] = float(score)

    # Look for CLEval metrics
    if "cleval_metrics" in ckpt:
        cleval = ckpt["cleval_metrics"]
        if isinstance(cleval, dict):
            if "precision" in cleval:
                metrics["precision"] = float(cleval["precision"])
            if "recall" in cleval:
                metrics["recall"] = float(cleval["recall"])
            if "hmean" in cleval or "f1" in cleval:
                metrics["hmean"] = float(cleval.get("hmean", cleval.get("f1")))

    # Look in hyper_parameters
    if "hyper_parameters" in ckpt:
        hp = ckpt["hyper_parameters"]
        if isinstance(hp, dict) and "val_metrics" in hp:
            val_metrics = hp["val_metrics"]
            if isinstance(val_metrics, dict):
                for key, value in val_metrics.items():
                    if isinstance(value, int | float):
                        metrics["additional_metrics"][key] = float(value)

    return metrics


def enrich_metadata_from_hydra(
    metadata: dict[str, Any],
    hydra_config: dict[str, Any],
) -> dict[str, Any]:
    """Enrich metadata with information from Hydra config.

    Args:
        metadata: Metadata dictionary to enrich
        hydra_config: Hydra configuration

    Returns:
        Enriched metadata
    """
    # Update experiment name if available
    if "exp_name" in hydra_config:
        metadata["exp_name"] = hydra_config["exp_name"]

    # Update model information
    if "model" in hydra_config:
        model_cfg = hydra_config["model"]
        if isinstance(model_cfg, dict):
            # Architecture
            if "architecture" in model_cfg:
                metadata["model"]["architecture"] = model_cfg["architecture"]

            # Encoder
            if "encoder" in model_cfg:
                encoder_cfg = model_cfg["encoder"]
                if isinstance(encoder_cfg, dict):
                    if "model_name" in encoder_cfg:
                        metadata["model"]["encoder"]["model_name"] = encoder_cfg["model_name"]
                    if "pretrained" in encoder_cfg:
                        metadata["model"]["encoder"]["pretrained"] = encoder_cfg["pretrained"]
                    if "frozen" in encoder_cfg:
                        metadata["model"]["encoder"]["frozen"] = encoder_cfg["frozen"]

            # Decoder
            if "decoder" in model_cfg:
                decoder_cfg = model_cfg["decoder"]
                if isinstance(decoder_cfg, dict):
                    if "name" in decoder_cfg or "_target_" in decoder_cfg:
                        name = decoder_cfg.get("name")
                        if not name and "_target_" in decoder_cfg:
                            target = decoder_cfg["_target_"]
                            name = target.split(".")[-1].lower() if "." in target else target
                        if name:
                            metadata["model"]["decoder"]["name"] = name

    # Update training info
    if "trainer" in hydra_config:
        trainer_cfg = hydra_config["trainer"]
        if isinstance(trainer_cfg, dict):
            if "max_epochs" in trainer_cfg:
                metadata["training"]["max_epochs"] = trainer_cfg["max_epochs"]

    # Update checkpointing config
    if "callbacks" in hydra_config:
        callbacks_cfg = hydra_config["callbacks"]
        if isinstance(callbacks_cfg, dict) and "model_checkpoint" in callbacks_cfg:
            ckpt_cfg = callbacks_cfg["model_checkpoint"]
            if isinstance(ckpt_cfg, dict):
                if "monitor" in ckpt_cfg:
                    metadata["checkpointing"]["monitor"] = ckpt_cfg["monitor"]
                if "mode" in ckpt_cfg:
                    metadata["checkpointing"]["mode"] = ckpt_cfg["mode"]
                if "save_top_k" in ckpt_cfg:
                    metadata["checkpointing"]["save_top_k"] = ckpt_cfg["save_top_k"]
                if "save_last" in ckpt_cfg:
                    metadata["checkpointing"]["save_last"] = ckpt_cfg["save_last"]

    return metadata


def save_metadata_yaml(metadata: dict[str, Any], ckpt_path: Path) -> Path:
    """Save metadata to YAML file.

    Args:
        metadata: Metadata dictionary
        ckpt_path: Checkpoint path

    Returns:
        Path to metadata file
    """
    metadata_path = ckpt_path.with_suffix(".ckpt.metadata.yaml")

    with metadata_path.open("w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    return metadata_path


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate metadata YAML files for existing checkpoints")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Outputs directory to scan for checkpoints (default: outputs/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be generated without creating files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    outputs_dir = args.outputs_dir.resolve()

    if not outputs_dir.exists():
        LOGGER.error("Outputs directory not found: %s", outputs_dir)
        return

    LOGGER.info("Scanning for checkpoints in: %s", outputs_dir)

    # Find checkpoints
    checkpoints = find_checkpoints(outputs_dir)

    if not checkpoints:
        LOGGER.info("No checkpoints found (or all already have metadata)")
        return

    LOGGER.info("Found %d checkpoints without metadata", len(checkpoints))

    # Process each checkpoint
    success_count = 0
    fail_count = 0

    for i, ckpt_path in enumerate(checkpoints, 1):
        LOGGER.info("[%d/%d] Processing: %s", i, len(checkpoints), ckpt_path)

        try:
            # Extract metadata
            metadata = extract_metadata_from_checkpoint(ckpt_path, outputs_dir)

            if metadata is None:
                LOGGER.warning("Failed to extract metadata from: %s", ckpt_path)
                fail_count += 1
                continue

            # Save metadata
            if args.dry_run:
                LOGGER.info("  [DRY RUN] Would create: %s", ckpt_path.with_suffix(".ckpt.metadata.yaml"))
                LOGGER.debug("  Metadata preview:\n%s", yaml.dump(metadata, default_flow_style=False))
            else:
                metadata_path = save_metadata_yaml(metadata, ckpt_path)
                LOGGER.info("  Created: %s", metadata_path)

            success_count += 1

        except Exception as exc:
            LOGGER.error("Failed to process %s: %s", ckpt_path, exc, exc_info=True)
            fail_count += 1

    # Summary
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("Summary:")
    LOGGER.info("  Total checkpoints: %d", len(checkpoints))
    LOGGER.info("  Successfully processed: %d", success_count)
    LOGGER.info("  Failed: %d", fail_count)
    LOGGER.info("=" * 60)

    if args.dry_run:
        LOGGER.info("\nThis was a dry run. Use without --dry-run to create files.")


if __name__ == "__main__":
    main()
