"""
Configuration Parser for Streamlit UI

This module provides utilities to parse Hydra configurations and extract
available options for the Streamlit UI components.
"""

from pathlib import Path
from typing import Any

import yaml

from ocr.models.core import registry


class ConfigParser:
    """Parser for extracting configuration options from Hydra configs."""

    def __init__(self, config_dir: str | None = None):
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

    def get_available_models(self) -> dict[str, list[str]]:
        """Get available model components (encoders, decoders, heads, losses).

        Returns:
            Dictionary with component types as keys and lists of options as values.
        """
        if "models" in self._cache:
            return self._cache["models"]

        models = {
            "encoders": [],
            "decoders": [],
            "heads": [],
            "losses": [],
            "optimizers": [],
        }

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

        # Optimizers from modular group
        optimizer_dir = self.config_dir / "model" / "optimizers"
        if optimizer_dir.exists():
            for yaml_file in optimizer_dir.glob("*.yaml"):
                models["optimizers"].append(yaml_file.stem)

        # Add registry-discovered components (if not already present)
        for enc in registry.list_encoders():
            if enc not in models["encoders"]:
                models["encoders"].append(enc)
        for dec in registry.list_decoders():
            if dec not in models["decoders"]:
                models["decoders"].append(dec)
        for head in registry.list_heads():
            if head not in models["heads"]:
                models["heads"].append(head)
        for loss in registry.list_losses():
            if loss not in models["losses"]:
                models["losses"].append(loss)

        # Also check timm backbones for encoders
        encoder_config = self.config_dir / "preset" / "models" / "encoder" / "timm_backbone.yaml"
        if encoder_config.exists():
            try:
                with open(encoder_config) as f:
                    yaml.safe_load(f)
                # Extract common backbone names (this would be expanded based on timm)
                models["backbones"] = [
                    "resnet18",
                    "resnet34",
                    "resnet50",
                    "mobilenetv3_small_050",
                    "efficientnet_b0",
                ]
            except Exception:
                models["backbones"] = ["resnet18"]  # fallback

        self._cache["models"] = models
        return models

    def get_available_architectures(self) -> list[str]:
        """Get available architecture presets from the registry for UI selection."""
        # Ensure architectures are registered by importing the module
        from ocr.models import architectures  # noqa: F401

        return registry.list_architectures()

    def get_architecture_metadata(self) -> dict[str, Any]:
        """Return UI-focused metadata for each registered architecture."""
        if "architecture_metadata" in self._cache:
            return self._cache["architecture_metadata"]

        metadata: dict[str, Any] = {}
        arch_dir = self.config_dir / "model" / "architectures"
        arch_ui_dir = self.config_dir / "ui_meta" / "architectures"
        if arch_dir.exists():
            for yaml_file in arch_dir.glob("*.yaml"):
                try:
                    with open(yaml_file) as f:
                        data = yaml.safe_load(f) or {}
                except yaml.YAMLError:
                    continue

                name = yaml_file.stem
                ui_metadata: dict[str, Any] = {}
                if arch_ui_dir.exists():
                    ui_meta_path = arch_ui_dir / yaml_file.name
                    if ui_meta_path.exists():
                        try:
                            with open(ui_meta_path) as meta_file:
                                meta_payload = yaml.safe_load(meta_file) or {}
                            ui_metadata = meta_payload.get("ui_metadata", meta_payload)
                        except yaml.YAMLError:
                            ui_metadata = {}

                # Ensure consistent keys
                ui_metadata.setdefault("components", {})
                ui_metadata.setdefault("compatible_backbones", [])
                ui_metadata.setdefault("recommended_optimizers", [])
                ui_metadata.setdefault("default_backbone", None)
                ui_metadata.setdefault("default_optimizer", None)
                ui_metadata.setdefault("default_learning_rate", None)
                ui_metadata.setdefault("recommended_data", None)
                ui_metadata.setdefault("default_decoder", None)
                ui_metadata.setdefault("compatible_decoders", [])
                ui_metadata.setdefault("default_head", None)
                ui_metadata.setdefault("default_loss", None)
                ui_metadata.setdefault("compatible_heads", [])
                ui_metadata.setdefault("compatible_losses", [])
                ui_metadata.setdefault("use_cases", [])

                metadata[name] = {
                    "ui_metadata": ui_metadata,
                    "component_overrides": data.get("component_overrides", {}),
                }

        self._cache["architecture_metadata"] = metadata
        return metadata

    def get_optimizer_metadata(self) -> dict[str, Any]:
        """Return UI metadata for optimizers."""
        if "optimizer_metadata" in self._cache:
            return self._cache["optimizer_metadata"]

        metadata: dict[str, Any] = {}
        optimizer_ui_dir = self.config_dir / "ui_meta" / "optimizers"
        if optimizer_ui_dir.exists():
            for yaml_file in optimizer_ui_dir.glob("*.yaml"):
                try:
                    with open(yaml_file) as f:
                        payload = yaml.safe_load(f) or {}
                except yaml.YAMLError:
                    continue

                name = yaml_file.stem
                ui_metadata = payload.get("ui_metadata", payload)
                learning_rate_meta = ui_metadata.get("learning_rate", {})
                if isinstance(learning_rate_meta, dict):
                    learning_rate_meta.setdefault("min", None)
                    learning_rate_meta.setdefault("max", None)
                    learning_rate_meta.setdefault("default", None)
                    learning_rate_meta.setdefault("step", None)
                    learning_rate_meta.setdefault("help", None)

                metadata[name] = {
                    "ui_metadata": ui_metadata,
                }

        self._cache["optimizer_metadata"] = metadata
        return metadata

    def get_preprocessing_profiles(self) -> dict[str, Any]:
        """Return metadata describing available preprocessing profiles for UI selection."""
        if "preprocessing_profiles" in self._cache:
            return self._cache["preprocessing_profiles"]

        metadata: dict[str, Any] = {}
        profiles_path = self.config_dir / "ui_meta" / "preprocessing_profiles.yaml"
        if profiles_path.exists():
            try:
                with open(profiles_path) as f:
                    payload = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                payload = {}
            profiles = payload.get("profiles", {})
            if isinstance(profiles, dict):
                for key, info in profiles.items():
                    if not isinstance(info, dict):
                        info = {}
                    metadata[key] = {
                        "label": info.get("label", key.replace("_", " ").title()),
                        "description": info.get("description", ""),
                        "overrides": info.get("overrides", []),
                    }

        # Always ensure "none" profile exists
        metadata.setdefault(
            "none",
            {
                "label": "No preprocessing (dataset defaults)",
                "description": "Use dataset-configured transforms without additional document cleanup.",
                "overrides": [],
            },
        )

        self._cache["preprocessing_profiles"] = metadata
        return metadata

    def get_training_parameters(self) -> dict[str, Any]:
        """Get available training parameters and their ranges from modular configs.

        Returns:
            Dictionary with parameter info including defaults and ranges.
        """
        if "training" in self._cache:
            return self._cache["training"]

        # Parse from modular configs
        params = {}

        # Trainer config
        trainer_config = self.config_dir / "trainer" / "default.yaml"
        if trainer_config.exists():
            try:
                with open(trainer_config) as f:
                    trainer = yaml.safe_load(f)
                params["max_epochs"] = {
                    "default": trainer.get("max_epochs", 10),
                    "min": 1,
                    "max": 200,
                    "type": "int",
                }
            except Exception as e:
                print(f"Error parsing trainer config: {e}")

        # Data config (batch_size)
        base_config = self.config_dir / "base.yaml"
        if base_config.exists():
            try:
                with open(base_config) as f:
                    base = yaml.safe_load(f)
                batch_size = base.get("data", {}).get("batch_size", 4)
                params["batch_size"] = {
                    "default": batch_size,
                    "min": 1,
                    "max": 64,
                    "type": "int",
                }
                params["seed"] = {"default": base.get("seed", 42), "type": "int"}
            except Exception as e:
                print(f"Error parsing base config: {e}")

        # Logger config (wandb, exp_version)
        logger_config = self.config_dir / "logger" / "default.yaml"
        if logger_config.exists():
            try:
                with open(logger_config) as f:
                    logger = yaml.safe_load(f)
                params["wandb"] = {"default": logger.get("wandb", False), "type": "bool"}
                params["exp_version"] = {"default": logger.get("exp_version", "v1.0"), "type": "str"}
            except Exception as e:
                print(f"Error parsing logger config: {e}")

        # Paths config (log_dir, checkpoint_dir)
        paths_config = self.config_dir / "paths" / "default.yaml"
        if paths_config.exists():
            try:
                with open(paths_config) as f:
                    paths = yaml.safe_load(f)
                params["log_dir"] = {"default": paths.get("log_dir", "outputs/exp/logs"), "type": "str"}
                params["checkpoint_dir"] = {"default": paths.get("checkpoint_dir", "outputs/exp/checkpoints"), "type": "str"}
            except Exception as e:
                print(f"Error parsing paths config: {e}")

        self._cache["training"] = params
        return params

    def get_available_datasets(self) -> list[str]:
        """Get available dataset configurations.

        Returns:
            List of available dataset configuration names.
        """
        if "datasets" in self._cache:
            return self._cache["datasets"]

        datasets = []
        dataset_dir = self.config_dir / "data"

        if dataset_dir.exists():
            datasets.extend(yaml_file.stem for yaml_file in dataset_dir.glob("*.yaml"))

        self._cache["datasets"] = datasets
        return datasets

    def get_available_presets(self) -> list[str]:
        """Get available configuration presets.

        Returns:
            List of available preset names.
        """
        if "presets" in self._cache:
            return self._cache["presets"]

        presets = []
        preset_dir = self.config_dir / "preset"

        if preset_dir.exists():
            presets = [yaml_file.stem for yaml_file in preset_dir.glob("*.yaml") if not yaml_file.name.startswith("_")]

        self._cache["presets"] = presets
        return presets

    def validate_config_combination(self, config: dict[str, Any]) -> list[str]:
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

        optimizer_name = config.get("optimizer")
        learning_rate = config.get("learning_rate", 0)
        if optimizer_name:
            optimizer_metadata = self.get_optimizer_metadata()
            optimizer_info = optimizer_metadata.get(optimizer_name, {})
            lr_meta = optimizer_info.get("ui_metadata", {}).get("learning_rate", {})
            lr_min = lr_meta.get("min", 1e-5)
            lr_max = lr_meta.get("max", 1e-2)
        else:
            lr_min, lr_max = 1e-5, 1e-2

        if not (lr_min <= learning_rate <= lr_max):
            errors.append(f"learning_rate must be between {lr_min} and {lr_max}")

        if config.get("batch_size", 0) <= 0:
            errors.append("batch_size must be positive")

        return errors

    def get_available_checkpoints(self) -> list[str]:
        """Get available checkpoint files for resuming training.

        Returns:
            List of available checkpoint file paths.
        """
        if "checkpoints" in self._cache:
            return self._cache["checkpoints"]

        checkpoints = []
        outputs_dir = self.config_dir.parent / "outputs"

        if outputs_dir.exists():
            # Look for checkpoint files in all experiment directories
            for exp_dir in outputs_dir.glob("*"):
                if exp_dir.is_dir():
                    checkpoint_dir = exp_dir / "checkpoints"
                    if checkpoint_dir.exists():
                        checkpoints.extend(
                            [str(ckpt_file.relative_to(self.config_dir.parent)) for ckpt_file in checkpoint_dir.glob("*.ckpt")]
                        )

        self._cache["checkpoints"] = checkpoints
        return checkpoints
