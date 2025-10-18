#!/usr/bin/env python3
"""
Generate .config.json files for existing checkpoints that don't have them.

This script scans all checkpoints in the outputs directory and creates
.config.json files alongside checkpoints that are missing them. This
ensures backward compatibility with the new filesystem refactoring
that saves resolved configs alongside checkpoints.
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_checkpoints(outputs_dir: Path) -> list[Path]:
    """Find all .ckpt files in the outputs directory."""
    return list(outputs_dir.rglob("*.ckpt"))


def has_config_json(checkpoint_path: Path) -> bool:
    """Check if a checkpoint already has a .config.json file."""
    config_json_path = checkpoint_path.with_suffix(".config.json")
    return config_json_path.exists()


def find_config_for_checkpoint(checkpoint_path: Path) -> Path | None:
    """Find the appropriate config file for a checkpoint using wandb run data."""
    # Extract timestamp from checkpoint path to find corresponding wandb run
    # Checkpoint paths follow pattern: outputs/{exp_name}-{timestamp}/checkpoints/{checkpoint_file}
    # We need to extract the timestamp part

    path_str = str(checkpoint_path)
    # Look for timestamp pattern in the path
    import re

    timestamp_match = re.search(r"(\d{8}_\d{6})", path_str)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
        # Convert from YYYYMMDD_HHMMSS to YYYY-MM-DD-HH-MM-SS format for wandb
        if len(timestamp) == 15:  # YYYYMMDD_HHMMSS
            timestamp_formatted = f"{timestamp[:8]}_{timestamp[9:]}"

        # Look for wandb run with this timestamp
        wandb_dir = Path(__file__).parent.parent.parent / "wandb"
        if wandb_dir.exists():
            # Look for run directories with this timestamp
            for run_dir in wandb_dir.iterdir():
                if run_dir.is_dir() and timestamp_formatted in run_dir.name:
                    config_path = run_dir / "files" / "config.yaml"
                    if config_path.exists():
                        return config_path

    # Fallback: Check various locations for config files (legacy method)
    candidates = []

    # Check checkpoint's parent directory and up
    parent = checkpoint_path.parent
    candidates.extend(
        [
            parent / "config.yaml",
            parent / "hparams.yaml",
            parent / "train.yaml",
            parent / "predict.yaml",
            parent.parent / "config.yaml",
            parent.parent / "hparams.yaml",
            parent.parent / "train.yaml",
            parent.parent / "predict.yaml",
        ]
    )

    # Check .hydra directories
    hydra_candidates = [
        parent.parent / ".hydra" / "config.yaml",
        parent.parent.parent / ".hydra" / "config.yaml",
    ]
    candidates.extend(hydra_candidates)

    # Check project-level configs
    project_root = Path(__file__).parent.parent.parent  # scripts/agent_tools -> project root
    candidates.extend(
        [
            project_root / "configs" / "config.yaml",
            project_root / "configs" / "train.yaml",
            project_root / "configs" / "predict.yaml",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def extract_model_config(config_path: Path) -> dict | None:
    """Extract the model configuration from a config file."""
    try:
        with open(config_path, encoding="utf-8") as f:
            if config_path.suffix in [".yaml", ".yml"]:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

        # Handle wandb config format (contains resolved config)
        if isinstance(config, dict) and "_wandb" in config:
            # This is a wandb config - the actual config is stored elsewhere in wandb
            # We need to look for the resolved config in the wandb files
            config_dir = config_path.parent
            # Look for other config files that might contain the resolved config
            resolved_config_path = config_dir / "config.yaml.resolved"
            if resolved_config_path.exists():
                try:
                    with open(resolved_config_path, encoding="utf-8") as f:
                        resolved_config = yaml.safe_load(f)
                    if isinstance(resolved_config, dict):
                        model_config = resolved_config.get("model")
                        if model_config is not None:
                            return {"model": model_config}
                except Exception:
                    pass

            # If no resolved config, create minimal config from args
            wandb_value = config.get("_wandb", {}).get("value", {})
            e_dict = wandb_value.get("e", {})
            if e_dict:
                # Get the first (and typically only) experiment entry
                exp_key = next(iter(e_dict.keys()))
                args = e_dict[exp_key].get("args", [])

                # Parse command line arguments (format: key=value)
                parsed_args = {}
                for arg in args:
                    if "=" in arg:
                        key, value = arg.split("=", 1)
                        parsed_args[key] = value

                # Create minimal but valid model config
                model_config = {
                    "_target_": "ocr.models.architecture.OCRModel",
                    "architectures": parsed_args.get("model/architectures", "dbnet"),
                    "encoder": {
                        "model_name": parsed_args.get("model.encoder.model_name", "resnet18"),
                    },
                    "component_overrides": {
                        "decoder": {
                            "name": parsed_args.get("model.component_overrides.decoder.name", "fpn_decoder"),
                        },
                        "head": {
                            "name": parsed_args.get("model.component_overrides.head.name", "db_head"),
                        },
                        "loss": {
                            "name": parsed_args.get("model.component_overrides.loss.name", "db_loss"),
                        },
                    },
                }

                # Add component-specific parameters based on the component names
                decoder_name = parsed_args.get("model.component_overrides.decoder.name", "fpn_decoder")
                if decoder_name == "pan_decoder":
                    model_config["component_overrides"]["decoder"]["params"] = {
                        "inner_channels": 256,
                        "out_channels": 256,
                        "output_channels": 256,
                    }
                elif decoder_name == "unet":
                    model_config["component_overrides"]["decoder"]["params"] = {"inner_channels": 256, "output_channels": 256}

                head_name = parsed_args.get("model.component_overrides.head.name", "db_head")
                if head_name == "db_head":
                    model_config["component_overrides"]["head"]["params"] = {
                        "in_channels": 256,  # Match decoder output channels
                        "k": 50,
                        "postprocess": {"box_thresh": 0.3, "max_candidates": 300, "thresh": 0.2, "use_polygon": True},
                    }

                loss_name = parsed_args.get("model.component_overrides.loss.name", "db_loss")
                if loss_name == "db_loss":
                    model_config["component_overrides"]["loss"]["params"] = {
                        "binary_map_loss_weight": 1,
                        "negative_ratio": 3,
                        "prob_map_loss_weight": 5,
                        "thresh_map_loss_weight": 10,
                    }

                return {"model": model_config}  # Handle traditional config format
        if isinstance(config, dict):
            model_config = config.get("model")
            if model_config is not None:
                return {"model": model_config}
            else:
                # For configs without explicit model section, save key components
                config_to_save = {
                    key: value
                    for key, value in config.items()
                    if key in ["model", "architecture", "encoder", "decoder", "head", "backbone"]
                }
                if config_to_save:
                    return config_to_save

    except Exception as e:
        logger.warning(f"Failed to parse config {config_path}: {e}")

    return None


def generate_config_json(checkpoint_path: Path) -> bool:
    """Generate a .config.json file for a checkpoint."""
    config_path = find_config_for_checkpoint(checkpoint_path)
    if config_path is None:
        model_config = create_default_model_config()
    else:
        model_config = extract_model_config(config_path) or create_default_model_config()

    # Save the config
    config_json_path = checkpoint_path.with_suffix(".config.json")
    try:
        with open(config_json_path, "w") as f:
            json.dump(model_config, f, indent=2, default=str)
        logger.info(f"Created config file: {config_json_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save config file {config_json_path}: {e}")
        return False


def create_default_model_config() -> dict:
    """Create a default model config for checkpoints without wandb data."""
    return {
        "model": {
            "_target_": "ocr.models.architecture.OCRModel",
            "architectures": "dbnet",
            "encoder": {
                "model_name": "resnet18",
            },
            "component_overrides": {
                "decoder": {
                    "name": "fpn_decoder",
                },
                "head": {
                    "name": "db_head",
                },
                "loss": {
                    "name": "db_loss",
                },
            },
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Generate .config.json files for existing checkpoints")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"), help="Directory containing checkpoints (default: outputs)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")

    args = parser.parse_args()

    if not args.outputs_dir.exists():
        logger.error(f"Outputs directory does not exist: {args.outputs_dir}")
        return 1

    checkpoints = find_checkpoints(args.outputs_dir)
    logger.info(f"Found {len(checkpoints)} checkpoints")

    missing_configs = []
    for checkpoint in checkpoints:
        if not has_config_json(checkpoint):
            missing_configs.append(checkpoint)

    logger.info(f"Found {len(missing_configs)} checkpoints missing .config.json files")

    if args.dry_run:
        logger.info("DRY RUN - Would create config files for:")
        for checkpoint in missing_configs[:10]:  # Show first 10
            logger.info(f"  {checkpoint}")
        if len(missing_configs) > 10:
            logger.info(f"  ... and {len(missing_configs) - 10} more")
        return 0

    success_count = 0
    for checkpoint in missing_configs:
        if generate_config_json(checkpoint):
            success_count += 1

    logger.info(f"Successfully created {success_count}/{len(missing_configs)} config files")
    return 0 if success_count == len(missing_configs) else 1


if __name__ == "__main__":
    exit(main())
