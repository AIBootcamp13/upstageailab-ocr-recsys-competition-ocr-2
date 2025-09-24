"""
Configuration Parser for Streamlit UI

This module provides utilities to parse Hydra configurations and extract
available options for the Streamlit UI components.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from omegaconf import OmegaConf


class ConfigParser:
    """Parser for extracting configuration options from Hydra configs."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the config parser.

        Args:
            config_dir: Path to the configs directory. If None, uses default.
        """
        if config_dir is None:
            # Default to configs directory relative to project root
            project_root = Path(__file__).parent.parent.parent
            self.config_dir = project_root / "configs"
        else:
            self.config_dir = Path(config_dir)

        self._cache = {}

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available model components (encoders, decoders, heads, losses).

        Returns:
            Dictionary with component types as keys and lists of options as values.
        """
        if "models" in self._cache:
            return self._cache["models"]

        models = {"encoders": [], "decoders": [], "heads": [], "losses": []}

        # Scan model directories
        model_dirs = {
            "encoders": self.config_dir / "preset" / "models" / "encoder",
            "decoders": self.config_dir / "preset" / "models" / "decoder",
            "heads": self.config_dir / "preset" / "models" / "head",
            "losses": self.config_dir / "preset" / "models" / "loss",
        }

        for component_type, dir_path in model_dirs.items():
            if dir_path.exists():
                for yaml_file in dir_path.glob("*.yaml"):
                    models[component_type].append(yaml_file.stem)

        # Also check timm backbones for encoders
        encoder_config = (
            self.config_dir / "preset" / "models" / "encoder" / "timm_backbone.yaml"
        )
        if encoder_config.exists():
            try:
                with open(encoder_config, "r") as f:
                    config = yaml.safe_load(f)
                # Extract common backbone names (this would be expanded based on timm)
                models["backbones"] = [
                    "resnet18",
                    "resnet34",
                    "resnet50",
                    "mobilenet_v3_small",
                    "efficientnet_b0",
                ]
            except Exception:
                models["backbones"] = ["resnet18"]  # fallback

        self._cache["models"] = models
        return models

    def get_training_parameters(self) -> Dict[str, Any]:
        """Get available training parameters and their ranges.

        Returns:
            Dictionary with parameter info including defaults and ranges.
        """
        if "training" in self._cache:
            return self._cache["training"]

        # Parse from train.yaml and ablation configs
        train_config = self.config_dir / "train.yaml"
        params = {}

        if train_config.exists():
            try:
                with open(train_config, "r") as f:
                    config = yaml.safe_load(f)

                # Extract trainer parameters
                if "trainer" in config:
                    trainer = config["trainer"]
                    params.update(
                        {
                            "max_epochs": {
                                "default": trainer.get("max_epochs", 10),
                                "min": 1,
                                "max": 100,
                                "type": "int",
                            },
                            "batch_size": {
                                "default": 4,
                                "min": 1,
                                "max": 64,
                                "type": "int",
                            },
                        }
                    )

                # Extract other parameters
                params.update(
                    {
                        "learning_rate": {
                            "default": 0.001,
                            "min": 1e-6,
                            "max": 1e-2,
                            "type": "float",
                        },
                        "seed": {"default": config.get("seed", 42), "type": "int"},
                    }
                )

            except Exception as e:
                print(f"Error parsing train config: {e}")

        self._cache["training"] = params
        return params

    def get_available_datasets(self) -> List[str]:
        """Get available dataset configurations.

        Returns:
            List of available dataset config names.
        """
        if "datasets" in self._cache:
            return self._cache["datasets"]

        datasets = []
        dataset_dir = self.config_dir / "preset" / "datasets"

        if dataset_dir.exists():
            datasets = [yaml_file.stem for yaml_file in dataset_dir.glob("*.yaml")]

        self._cache["datasets"] = datasets
        return datasets

    def get_available_presets(self) -> List[str]:
        """Get available configuration presets.

        Returns:
            List of available preset names.
        """
        if "presets" in self._cache:
            return self._cache["presets"]

        presets = []
        preset_dir = self.config_dir / "preset"

        if preset_dir.exists():
            presets = [
                yaml_file.stem
                for yaml_file in preset_dir.glob("*.yaml")
                if not yaml_file.name.startswith("_")
            ]

        self._cache["presets"] = presets
        return presets

    def validate_config_combination(self, config: Dict[str, Any]) -> List[str]:
        """Validate a configuration combination.

        Args:
            config: Configuration dictionary to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Basic validation rules
        if config.get("max_epochs", 0) <= 0:
            errors.append("max_epochs must be positive")

        if not (1e-6 <= config.get("learning_rate", 0) <= 1e-2):
            errors.append("learning_rate must be between 1e-6 and 1e-2")

        if config.get("batch_size", 0) <= 0:
            errors.append("batch_size must be positive")

        return errors
