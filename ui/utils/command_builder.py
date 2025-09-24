"""
Command Builder for Streamlit UI

This module provides utilities to build and validate CLI commands
for training, testing, and prediction based on user selections.
"""

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class CommandBuilder:
    """Builder for generating CLI commands from UI selections."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize the command builder.

        Args:
            project_root: Path to the project root directory.
        """
        if project_root is None:
            # Default to project root relative to this file
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)

        self.runners_dir = self.project_root / "runners"

    def _add_config_overrides(self, cmd_parts: List[str], overrides: List[str]) -> None:
        """Add config path and overrides to command parts.

        Args:
            cmd_parts: Command parts list to modify.
            overrides: List of override strings.
        """
        cmd_parts.extend(["--config-path", str(self.project_root / "configs")])
        cmd_parts.extend(overrides)

    def build_train_command(self, config: Dict[str, Any]) -> str:
        """Build a training command from configuration.

        Args:
            config: Configuration dictionary from UI.

        Returns:
            Complete CLI command string.
        """
        cmd_parts = ["python", str(self.runners_dir / "train.py")]

        # Add overrides
        overrides = self._build_overrides(config)
        self._add_config_overrides(cmd_parts, overrides)

        return " ".join(cmd_parts)

    def build_test_command(self, config: Dict[str, Any]) -> str:
        """Build a testing command from configuration.

        Args:
            config: Configuration dictionary from UI.

        Returns:
            Complete CLI command string.
        """
        cmd_parts = ["python", str(self.runners_dir / "test.py")]

        # Add overrides
        overrides = self._build_test_overrides(config)
        self._add_config_overrides(cmd_parts, overrides)

        return " ".join(cmd_parts)

    def build_predict_command(self, config: Dict[str, Any]) -> str:
        """Build a prediction command from configuration.

        Args:
            config: Configuration dictionary from UI.

        Returns:
            Complete CLI command string.
        """
        cmd_parts = ["python", str(self.runners_dir / "predict.py")]

        # Add overrides
        overrides = self._build_predict_overrides(config)
        self._add_config_overrides(cmd_parts, overrides)

        return " ".join(cmd_parts)

    def _build_overrides(self, config: Dict[str, Any]) -> List[str]:
        """Build Hydra overrides from config dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            List of override strings.
        """
        overrides = []

        # Model configuration
        if "encoder" in config:
            overrides.append(f"models.encoder.model_name={config['encoder']}")

        if "decoder" in config:
            overrides.append(
                f"models.decoder._target=ocr.models.decoder.{config['decoder']}.UNet"
            )

        if "head" in config:
            overrides.append(
                f"models.head._target=ocr.models.head.{config['head']}.DBHead"
            )

        if "loss" in config:
            overrides.append(
                f"models.loss._target=ocr.models.loss.{config['loss']}.DBLoss"
            )

        # Training parameters
        if "learning_rate" in config:
            overrides.append(f"models.optimizer.lr={config['learning_rate']}")

        if "batch_size" in config:
            overrides.append("data.batch_size=" + str(config["batch_size"]))

        if "max_epochs" in config:
            overrides.append("trainer.max_epochs=" + str(config["max_epochs"]))

        if "seed" in config:
            overrides.append("seed=" + str(config["seed"]))

        # Experiment settings
        if "exp_name" in config:
            overrides.append(f"exp_name={config['exp_name']}")

        if "wandb" in config:
            overrides.append(f"wandb={str(config['wandb']).lower()}")

        if "resume" in config and config["resume"]:
            overrides.append(f"resume={config['resume']}")

        return overrides

    def _build_test_overrides(self, config: Dict[str, Any]) -> List[str]:
        """Build overrides for test command.

        Args:
            config: Configuration dictionary.

        Returns:
            List of override strings.
        """
        overrides = []

        if "checkpoint_path" in config:
            overrides.append(f"checkpoint_path={config['checkpoint_path']}")

        if "exp_name" in config:
            overrides.append(f"exp_name={config['exp_name']}")

        return overrides

    def _build_predict_overrides(self, config: Dict[str, Any]) -> List[str]:
        """Build overrides for predict command.

        Args:
            config: Configuration dictionary.

        Returns:
            List of override strings.
        """
        overrides = []

        if "checkpoint_path" in config:
            overrides.append(f"checkpoint_path={config['checkpoint_path']}")

        if "exp_name" in config:
            overrides.append(f"exp_name={config['exp_name']}")

        if "minified_json" in config:
            overrides.append(f"minified_json={str(config['minified_json']).lower()}")

        return overrides

    def validate_command(self, command: str) -> Tuple[bool, str]:
        """Validate that a command can be executed.

        Args:
            command: Command string to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            # Parse the command
            parts = command.split()
            if not parts:
                return False, "Empty command"

            script_path = parts[1] if len(parts) > 1 else ""
            if not Path(script_path).exists():
                return False, f"Script not found: {script_path}"

            return True, ""

        except Exception as e:
            return False, f"Command validation error: {str(e)}"

    def execute_command(
        self, command: str, cwd: Optional[str] = None
    ) -> Tuple[int, str, str]:
        """Execute a command and return the results.

        Args:
            command: Command string to execute.
            cwd: Working directory for execution.

        Returns:
            Tuple of (return_code, stdout, stderr).
        """
        if cwd is None:
            cwd = str(self.project_root)

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out after 5 minutes"
        except Exception as e:
            return -1, "", f"Execution error: {str(e)}"
