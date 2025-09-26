"""
Command Builder for Streamlit UI

This module provides utilities to build and validate CLI commands
for training, testing, and prediction based on user selections.
"""

import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class CommandBuilder:
    """Builder for generating CLI commands from UI selections."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize the command builder.

        Args:
            project_root: Path to the project root directory.
        """
        if project_root is None:
            # Default to project root relative to this file
            self.project_root = Path(__file__).resolve().parent.parent.parent
        else:
            self.project_root = Path(project_root)

        self.runners_dir = self.project_root / "runners"

    def build_command_from_overrides(
        self,
        script: str,
        overrides: List[str],
        constant_overrides: Optional[List[str]] = None,
    ) -> str:
        """Generic command builder for a given runner script using overrides.

        Args:
            script: Runner script filename, e.g., "train.py".
            overrides: Computed Hydra overrides from UI.
            constant_overrides: Constant overrides defined by schema.

        Returns:
            A complete CLI command string.
        """
        cmd_parts = ["uv", "run", "python", str(self.runners_dir / script)]
        all_overrides = list(constant_overrides or []) + list(overrides or [])
        self._add_config_overrides(cmd_parts, all_overrides)
        return " ".join(cmd_parts)

    def _add_config_overrides(self, cmd_parts: List[str], overrides: List[str]) -> None:
        """Add config path and overrides to command parts.

        Args:
            cmd_parts: Command parts list to modify.
            overrides: List of override strings.
        """
        # Do not pass --config-path; runners set config_path via @hydra.main
        # Ensure overrides are safe for Hydra CLI parsing
        safe_overrides = [self._quote_override(ov) for ov in overrides]
        cmd_parts.extend(safe_overrides)

    def _quote_override(self, ov: str) -> str:
        """Quote override values that contain special characters like '=' or spaces.

        Hydra treats '=' as the separator between key and value; if the value also contains
        '=', it must be quoted. We use single quotes and rely on shlex-aware splitting.
        """
        if "=" not in ov:
            return ov
        key, value = ov.split("=", 1)
        # Characters that warrant quoting
        if any(ch in value for ch in ["=", " ", "\t", '"']):
            # Basic single-quote wrapping; assume no single quotes in paths
            return f"{key}='{value}'"
        return ov

    def build_train_command(self, config: Dict[str, Any]) -> str:
        """Build a training command from configuration.

        Args:
            config: Configuration dictionary from UI.

        Returns:
            Complete CLI command string.
        """
        cmd_parts = ["uv", "run", "python", str(self.runners_dir / "train.py")]

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
        cmd_parts = ["uv", "run", "python", str(self.runners_dir / "test.py")]

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
        cmd_parts = ["uv", "run", "python", str(self.runners_dir / "predict.py")]

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
        # deprecated: do not force encoder_path

        # Model configuration
        if "encoder" in config:
            overrides.append(f"models.encoder.model_name={config['encoder']}")

        # Training parameters
        if "learning_rate" in config:
            overrides.append(f"models.optimizer.lr={config['learning_rate']}")

        if "batch_size" in config:
            overrides.append(f"data.batch_size={config['batch_size']}")

        if "max_epochs" in config:
            overrides.append(f"trainer.max_epochs={config['max_epochs']}")

        if "seed" in config:
            overrides.append(f"seed={config['seed']}")

        # Experiment settings
        if "exp_name" in config and config["exp_name"]:
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
        return self._extracted_from__build_predict_overrides_10(config)

    def _build_predict_overrides(self, config: Dict[str, Any]) -> List[str]:
        """Build overrides for predict command.

        Args:
            config: Configuration dictionary.

        Returns:
            List of override strings.
        """
        overrides = self._extracted_from__build_predict_overrides_10(config)
        if "minified_json" in config:
            overrides.append(f"minified_json={str(config['minified_json']).lower()}")

        return overrides

    # TODO Rename this here and in `_build_test_overrides` and `_build_predict_overrides`
    def _extracted_from__build_predict_overrides_10(self, config):
        result = ["models.encoder.model_name=resnet18", "models.optimizer.lr=0.001"]
        if "checkpoint_path" in config and config["checkpoint_path"]:
            result.append(f"checkpoint_path={config['checkpoint_path']}")
        if "exp_name" in config and config["exp_name"]:
            result.append(f"exp_name={config['exp_name']}")
        return result

    def validate_command(self, command: str) -> Tuple[bool, str]:
        """Validate that a command can be executed.

        Args:
            command: Command string to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            parts = command.split()
            if not parts:
                return False, "Empty command"

            # Check for 'uv run python' structure
            if parts[:3] == ["uv", "run", "python"]:
                script_path = Path(parts[3])
                if not script_path.exists():
                    return False, f"Script not found: {script_path}"
                return True, ""

            return False, "Command must start with 'uv run python'"

        except Exception as e:
            return False, f"Command validation error: {e}"

    def execute_command_streaming(
        self,
        command: str,
        cwd: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[int, str, str]:
        """Execute a command with streaming output and process group management.

        Args:
            command: Command string to execute.
            cwd: Working directory for execution.
            progress_callback: Optional callback function to handle output lines in real-time.

        Returns:
            Tuple of (return_code, stdout, stderr).
        """
        if cwd is None:
            cwd = str(self.project_root)

        try:
            # Use Popen with process group for better cleanup control
            # Use shell-aware splitting to preserve quoted arguments
            process = subprocess.Popen(
                shlex.split(command),
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
                preexec_fn=os.setsid,  # Create new process group
            )

            stdout_lines = []
            stderr_lines = []

            # Read output streams
            while True:
                # Check if process is still running
                if process.poll() is not None:
                    break

                # Read stdout
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        stdout_lines.append(line.rstrip())
                        if progress_callback:
                            progress_callback(f"OUT: {line.rstrip()}")

                # Read stderr
                if process.stderr:
                    line = process.stderr.readline()
                    if line:
                        stderr_lines.append(line.rstrip())
                        if progress_callback:
                            progress_callback(f"ERR: {line.rstrip()}")

                # Small delay to prevent busy waiting
                time.sleep(0.1)

            # Get any remaining output
            if process.stdout:
                remaining_stdout, remaining_stderr = process.communicate()
                if remaining_stdout:
                    for line in remaining_stdout.splitlines():
                        stdout_lines.append(line)
                        if progress_callback:
                            progress_callback(f"OUT: {line}")
                if remaining_stderr:
                    for line in remaining_stderr.splitlines():
                        stderr_lines.append(line)
                        if progress_callback:
                            progress_callback(f"ERR: {line}")

            return process.returncode, "\n".join(stdout_lines), "\n".join(stderr_lines)

        except FileNotFoundError:
            return (
                -1,
                "",
                "Execution error: 'uv' command not found. Is it installed and in your PATH?",
            )
        except Exception as e:
            return -1, "", f"An unexpected execution error occurred: {e}"

    def terminate_process_group(self, process: subprocess.Popen) -> bool:
        """Terminate a process group to ensure all child processes are killed.

        Args:
            process: The Popen process object.

        Returns:
            True if termination was successful, False otherwise.
        """
        try:
            if process.poll() is None:  # Process is still running
                # Send SIGTERM to the entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)

                # Wait a bit for graceful shutdown
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    # If still running, force kill
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait()
            return True
        except (ProcessLookupError, OSError):
            # Process might already be dead
            return False
