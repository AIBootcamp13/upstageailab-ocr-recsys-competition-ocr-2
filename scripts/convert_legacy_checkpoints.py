#!/usr/bin/env python
"""Convert legacy checkpoints to V2 metadata format.

This script generates .metadata.yaml files for existing checkpoints that don't
have them. This enables fast catalog building for legacy experiments.

Performance:
    - Conversion time: ~2-5 seconds per checkpoint (torch.load overhead)
    - One-time cost: After conversion, catalog builds are 40-100x faster

Usage:
    # Convert all checkpoints in outputs directory
    python scripts/convert_legacy_checkpoints.py --outputs-dir outputs/

    # Convert specific experiment
    python scripts/convert_legacy_checkpoints.py --exp-dir outputs/my_experiment/

    # Dry run (show what would be converted)
    python scripts/convert_legacy_checkpoints.py --outputs-dir outputs/ --dry-run

    # Force reconversion (overwrite existing metadata)
    python scripts/convert_legacy_checkpoints.py --outputs-dir outputs/ --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def _load_hydra_config(checkpoint_path: Path) -> dict[str, Any] | None:
    """Load Hydra config for a checkpoint if available.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Hydra config dict, or None if not found
    """
    try:
        # Typical structure: outputs/exp_name/checkpoints/checkpoint.ckpt
        exp_dir = checkpoint_path.parent.parent
        hydra_config_path = exp_dir / ".hydra" / "config.yaml"

        if not hydra_config_path.exists():
            LOGGER.debug("No Hydra config found at: %s", hydra_config_path)
            return None

        with hydra_config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        LOGGER.debug("Loaded Hydra config from: %s", hydra_config_path)
        return config

    except Exception as exc:
        LOGGER.debug("Failed to load Hydra config: %s", exc)
        return None


def extract_metadata_from_checkpoint(
    checkpoint_path: Path,
    checkpoint_data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Extract metadata from a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file
        checkpoint_data: Pre-loaded checkpoint data (optional, loads if None)

    Returns:
        Metadata dictionary conforming to CheckpointMetadataV1 schema,
        or None if extraction fails
    """
    try:
        # Import after checking checkpoint exists
        from ui.apps.inference.services.checkpoint.types import (
            CheckpointingConfig,
            CheckpointMetadataV1,
            DecoderInfo,
            EncoderInfo,
            HeadInfo,
            LossInfo,
            MetricsInfo,
            ModelInfo,
            TrainingInfo,
        )

        # Load checkpoint if not provided
        if checkpoint_data is None:
            LOGGER.debug("Loading checkpoint: %s", checkpoint_path)
            checkpoint_data = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

        # Extract hyper_parameters (contains model config)
        hyper_parameters = checkpoint_data.get("hyper_parameters", {})

        # Try to load config from Hydra if hyper_parameters is empty
        if not hyper_parameters:
            LOGGER.debug("No hyper_parameters in checkpoint, trying Hydra config")
            hydra_config = _load_hydra_config(checkpoint_path)
            if hydra_config:
                # Use Hydra config as source for model architecture
                hyper_parameters = {
                    "architecture_name": hydra_config.get("model", {}).get("architecture", "unknown"),
                    "encoder": hydra_config.get("model", {}).get("component_overrides", {}).get("encoder", {}),
                    "decoder": hydra_config.get("model", {}).get("component_overrides", {}).get("decoder", {}),
                    "head": hydra_config.get("model", {}).get("component_overrides", {}).get("head", {}),
                    "loss": hydra_config.get("model", {}).get("component_overrides", {}).get("loss", {}),
                    "max_epochs": hydra_config.get("trainer", {}).get("max_epochs"),
                }

        # Extract training epoch and step
        epoch = checkpoint_data.get("epoch", 0)
        global_step = checkpoint_data.get("global_step", 0)

        # Extract experiment name from path
        # Typical structure: outputs/exp_name/checkpoints/checkpoint.ckpt
        exp_name = "unknown_experiment"
        try:
            # Go up two levels from checkpoint: checkpoints/ -> exp_dir/ -> outputs/
            exp_name = checkpoint_path.parent.parent.name
        except (AttributeError, IndexError):
            pass

        # Extract model architecture info
        architecture = hyper_parameters.get("architecture_name", "unknown")

        # Extract encoder info
        encoder_config = hyper_parameters.get("encoder", {})
        encoder_info = EncoderInfo(
            model_name=encoder_config.get("model_name", "unknown"),
            pretrained=encoder_config.get("pretrained", True),
            frozen=encoder_config.get("frozen", False),
        )

        # Extract decoder info
        decoder_config = hyper_parameters.get("decoder", {})
        decoder_info = DecoderInfo(
            name=decoder_config.get("name", "unknown"),
            in_channels=decoder_config.get("in_channels", []),
            inner_channels=decoder_config.get("inner_channels"),
            output_channels=decoder_config.get("output_channels"),
            params=decoder_config.get("params", {}),
        )

        # Extract head info
        head_config = hyper_parameters.get("head", {})
        head_info = HeadInfo(
            name=head_config.get("name", "unknown"),
            in_channels=head_config.get("in_channels"),
            params=head_config.get("params", {}),
        )

        # Extract loss info
        loss_config = hyper_parameters.get("loss", {})
        loss_info = LossInfo(
            name=loss_config.get("name", "unknown"),
            params=loss_config.get("params", {}),
        )

        # Build model info
        model_info = ModelInfo(
            architecture=architecture,
            encoder=encoder_info,
            decoder=decoder_info,
            head=head_info,
            loss=loss_info,
        )

        # Extract metrics from multiple sources
        metrics_dict: dict[str, float] = {}

        # 1. Try to extract from cleval_metrics (most reliable for our checkpoints)
        if "cleval_metrics" in checkpoint_data:
            cleval = checkpoint_data["cleval_metrics"]
            if isinstance(cleval, dict):
                LOGGER.debug("Found cleval_metrics: %s", cleval)
                metrics_dict.update(cleval)

        # 2. Try to extract from callbacks
        callback_metrics = checkpoint_data.get("callbacks", {})
        for callback_name, callback_state in callback_metrics.items():
            if isinstance(callback_state, dict):
                # Look for best_model_score which often contains the monitored metric
                if "best_model_score" in callback_state:
                    best_score = callback_state["best_model_score"]
                    monitor = callback_state.get("monitor", "unknown_metric")
                    if isinstance(best_score, int | float):
                        metrics_dict[monitor] = float(best_score)

        # 3. Try to extract from hyper_parameters metrics if available
        if "metrics" in hyper_parameters:
            hp_metrics = hyper_parameters["metrics"]
            if isinstance(hp_metrics, dict):
                metrics_dict.update(hp_metrics)

        # Extract specific metrics (precision, recall, hmean)
        # Use flexible pattern matching
        precision = None
        recall = None
        hmean = None
        validation_loss = None

        for key, value in metrics_dict.items():
            key_lower = key.lower()

            # Match precision
            if precision is None and "precision" in key_lower:
                precision = float(value)

            # Match recall
            if recall is None and "recall" in key_lower:
                recall = float(value)

            # Match hmean/f1
            if hmean is None and ("hmean" in key_lower or "f1" in key_lower):
                hmean = float(value)

            # Match validation loss
            if validation_loss is None and "loss" in key_lower and "val" in key_lower:
                validation_loss = float(value)

        # Build metrics info
        metrics_info = MetricsInfo(
            precision=precision,
            recall=recall,
            hmean=hmean,
            validation_loss=validation_loss,
            additional_metrics=metrics_dict,
        )

        # Extract checkpointing config
        # Try to infer from callback state
        monitor = "val/hmean"  # Default
        mode = "max"  # Default
        save_top_k = 1  # Default
        save_last = True  # Default

        for callback_name, callback_state in callback_metrics.items():
            if "ModelCheckpoint" in callback_name and isinstance(callback_state, dict):
                monitor = callback_state.get("monitor", monitor)
                mode = callback_state.get("mode", mode)
                save_top_k = callback_state.get("save_top_k", save_top_k)
                save_last = callback_state.get("save_last", save_last)
                break

        checkpointing_config = CheckpointingConfig(
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
        )

        # Build training info
        training_info = TrainingInfo(
            epoch=epoch,
            global_step=global_step,
            training_phase="training",  # Default assumption
            max_epochs=hyper_parameters.get("max_epochs"),
        )

        # Resolve Hydra config path
        # Typical structure: outputs/exp_name/.hydra/config.yaml
        hydra_config_path = None
        try:
            exp_dir = checkpoint_path.parent.parent
            hydra_config = exp_dir / ".hydra" / "config.yaml"
            if hydra_config.exists():
                # Store relative path from outputs directory
                outputs_dir = exp_dir.parent
                hydra_config_path = str(hydra_config.relative_to(outputs_dir))
        except (ValueError, AttributeError):
            pass

        # Get checkpoint file modification time as created_at
        created_at = datetime.fromtimestamp(checkpoint_path.stat().st_mtime).isoformat()

        # Get relative checkpoint path
        checkpoint_path_str = str(checkpoint_path)
        try:
            outputs_dir = checkpoint_path.parent.parent.parent
            if outputs_dir.name == "outputs":
                checkpoint_path_str = str(checkpoint_path.relative_to(outputs_dir))
        except (ValueError, AttributeError):
            pass

        # Build metadata
        metadata = CheckpointMetadataV1(
            schema_version="1.0",
            checkpoint_path=checkpoint_path_str,
            exp_name=exp_name,
            created_at=created_at,
            training=training_info,
            model=model_info,
            metrics=metrics_info,
            checkpointing=checkpointing_config,
            hydra_config_path=hydra_config_path,
            wandb_run_id=None,  # Not available from checkpoint
        )

        return metadata.model_dump(mode="python", exclude_none=True)

    except Exception as exc:
        LOGGER.error(
            "Failed to extract metadata from %s: %s",
            checkpoint_path,
            exc,
            exc_info=True,
        )
        return None


def convert_checkpoint(
    checkpoint_path: Path,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """Convert a single checkpoint to V2 metadata format.

    Args:
        checkpoint_path: Path to checkpoint file
        force: Overwrite existing metadata file
        dry_run: Don't actually write metadata

    Returns:
        True if conversion succeeded, False otherwise
    """
    metadata_path = checkpoint_path.with_suffix(".metadata.yaml")

    # Check if metadata already exists
    if metadata_path.exists() and not force:
        LOGGER.debug("Metadata already exists (skipping): %s", metadata_path)
        return False

    if dry_run:
        LOGGER.info("[DRY RUN] Would convert: %s", checkpoint_path)
        return True

    # Extract metadata
    metadata_dict = extract_metadata_from_checkpoint(checkpoint_path)

    if metadata_dict is None:
        LOGGER.warning("Failed to extract metadata: %s", checkpoint_path)
        return False

    # Write metadata file
    try:
        with metadata_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                metadata_dict,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        LOGGER.info("Converted: %s -> %s", checkpoint_path.name, metadata_path.name)
        return True

    except Exception as exc:
        LOGGER.error(
            "Failed to write metadata for %s: %s",
            checkpoint_path,
            exc,
            exc_info=True,
        )
        return False


def convert_directory(
    directory: Path,
    force: bool = False,
    dry_run: bool = False,
    recursive: bool = True,
) -> dict[str, int]:
    """Convert all checkpoints in a directory.

    Args:
        directory: Directory to search for checkpoints
        force: Overwrite existing metadata files
        dry_run: Don't actually write metadata
        recursive: Recursively search subdirectories

    Returns:
        Statistics dict with counts
    """
    if not directory.exists():
        LOGGER.error("Directory does not exist: %s", directory)
        return {"converted": 0, "skipped": 0, "failed": 0, "total": 0}

    # Find all checkpoint files
    if recursive:
        checkpoint_pattern = "**/*.ckpt"
    else:
        checkpoint_pattern = "*.ckpt"

    checkpoint_paths = list(directory.glob(checkpoint_pattern))

    if not checkpoint_paths:
        LOGGER.warning("No checkpoint files found in: %s", directory)
        return {"converted": 0, "skipped": 0, "failed": 0, "total": 0}

    LOGGER.info("Found %d checkpoint files in: %s", len(checkpoint_paths), directory)

    # Convert each checkpoint
    converted = 0
    skipped = 0
    failed = 0

    for ckpt_path in checkpoint_paths:
        metadata_path = ckpt_path.with_suffix(".metadata.yaml")

        # Check if metadata exists
        if metadata_path.exists() and not force:
            skipped += 1
            continue

        # Convert
        success = convert_checkpoint(ckpt_path, force=force, dry_run=dry_run)

        if success:
            converted += 1
        else:
            # Check if it was skipped or failed
            if metadata_path.exists() and not force:
                skipped += 1
            else:
                failed += 1

    return {
        "converted": converted,
        "skipped": skipped,
        "failed": failed,
        "total": len(checkpoint_paths),
    }


def main() -> int:
    """Main entry point for conversion tool.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Convert legacy checkpoints to V2 metadata format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--outputs-dir",
        type=Path,
        help="Outputs directory to search for checkpoints (recursive)",
    )

    parser.add_argument(
        "--exp-dir",
        type=Path,
        help="Single experiment directory to convert",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Single checkpoint file to convert",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing metadata files",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without actually converting",
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't recursively search subdirectories",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not any([args.outputs_dir, args.exp_dir, args.checkpoint]):
        parser.error("Must specify one of: --outputs-dir, --exp-dir, or --checkpoint")
        return 1

    # Run conversion
    try:
        if args.checkpoint:
            # Convert single checkpoint
            if not args.checkpoint.exists():
                LOGGER.error("Checkpoint file does not exist: %s", args.checkpoint)
                return 1

            success = convert_checkpoint(
                args.checkpoint,
                force=args.force,
                dry_run=args.dry_run,
            )

            if success:
                LOGGER.info("Conversion complete!")
                return 0
            else:
                LOGGER.error("Conversion failed")
                return 1

        elif args.exp_dir or args.outputs_dir:
            # Convert directory
            target_dir = args.exp_dir or args.outputs_dir
            recursive = not args.no_recursive

            stats = convert_directory(
                target_dir,
                force=args.force,
                dry_run=args.dry_run,
                recursive=recursive,
            )

            # Print summary
            LOGGER.info("\n" + "=" * 60)
            LOGGER.info("Conversion Summary")
            LOGGER.info("=" * 60)
            LOGGER.info("Total checkpoints found: %d", stats["total"])
            LOGGER.info("Converted:               %d", stats["converted"])
            LOGGER.info("Skipped (existing):      %d", stats["skipped"])
            LOGGER.info("Failed:                  %d", stats["failed"])
            LOGGER.info("=" * 60)

            if stats["failed"] > 0:
                LOGGER.warning("Some conversions failed. Check logs for details.")
                return 1

            if stats["converted"] == 0 and stats["skipped"] == 0:
                LOGGER.warning("No checkpoints were converted.")
                return 1

            LOGGER.info("Conversion complete!")
            return 0

    except KeyboardInterrupt:
        LOGGER.warning("\nConversion interrupted by user")
        return 130

    except Exception as exc:
        LOGGER.error("Conversion failed: %s", exc, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
