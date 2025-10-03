"""
Command Builder

Builds CLI commands from parameter models with proper Hydra override formatting.
"""

from pathlib import Path

from .models import CommandParams, PredictCommandParams, TestCommandParams, TrainCommandParams
from .quoting import quote_override


class CommandBuilder:
    """Builds CLI commands from parameter models."""

    def __init__(self, project_root: str | None = None):
        """Initialize with project root."""
        if project_root is None:
            # Default to project root relative to this file
            self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        else:
            self.project_root = Path(project_root)

        self.runners_dir = self.project_root / "runners"

    def build_command_from_overrides(
        self,
        script: str,
        overrides: list[str],
        constant_overrides: list[str] | None = None,
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

    def _add_config_overrides(self, cmd_parts: list[str], overrides: list[str]) -> None:
        """Add config path and overrides to command parts.

        Args:
            cmd_parts: Command parts list to modify.
            overrides: List of override strings.
        """
        # Do not pass --config-path; runners set config_path via @hydra.main
        # Ensure overrides are safe for Hydra CLI parsing
        safe_overrides = [quote_override(ov) for ov in overrides]
        cmd_parts.extend(safe_overrides)

    def build_train_command(self, params: TrainCommandParams) -> str:
        """Build a training command from parameters.

        Args:
            params: TrainCommandParams containing all training parameters.

        Returns:
            Complete CLI command string.
        """
        cmd_parts = ["uv", "run", "python", str(self.runners_dir / "train.py")]

        # Add overrides
        overrides = self._build_overrides_from_model(params)
        self._add_config_overrides(cmd_parts, overrides)

        return " ".join(cmd_parts)

    def build_test_command(self, params: TestCommandParams) -> str:
        """Build a testing command from parameters.

        Args:
            params: TestCommandParams containing all testing parameters.

        Returns:
            Complete CLI command string.
        """
        cmd_parts = ["uv", "run", "python", str(self.runners_dir / "test.py")]

        # Add overrides
        overrides = self._extracted_from_build_predict_overrides(params)
        self._add_config_overrides(cmd_parts, overrides)

        return " ".join(cmd_parts)

    def build_predict_command(self, params: PredictCommandParams) -> str:
        """Build a prediction command from parameters.

        Args:
            params: PredictCommandParams containing all prediction parameters.

        Returns:
            Complete CLI command string.
        """
        cmd_parts = ["uv", "run", "python", str(self.runners_dir / "predict.py")]

        # Add overrides
        overrides = self._extracted_from_build_predict_overrides(params)
        if params.minified_json:
            overrides.append(f"minified_json={str(params.minified_json).lower()}")

        self._add_config_overrides(cmd_parts, overrides)

        return " ".join(cmd_parts)

    def _build_overrides_from_model(self, params: TrainCommandParams) -> list[str]:
        """Convert parameter model to Hydra overrides.

        Args:
            params: TrainCommandParams containing training parameters.

        Returns:
            List of override strings.
        """
        overrides = []

        # Model configuration
        if params.encoder:
            overrides.append(f"model.encoder.model_name={params.encoder}")
        if params.decoder:
            overrides.append(f"model.component_overrides.decoder.name={params.decoder}")
        if params.head:
            overrides.append(f"model.component_overrides.head.name={params.head}")
        if params.loss:
            overrides.append(f"model.component_overrides.loss.name={params.loss}")
        if params.optimizer:
            overrides.append(f"model/optimizers={params.optimizer}")

        # Training parameters
        if params.learning_rate is not None:
            overrides.append(f"model.optimizer.lr={params.learning_rate}")
        if params.weight_decay is not None:
            overrides.append(f"model.optimizer.weight_decay={params.weight_decay}")

        if params.batch_size:
            overrides.append(f"data.batch_size={params.batch_size}")

        if params.max_epochs:
            overrides.append(f"trainer.max_epochs={params.max_epochs}")
        if params.accumulate_grad_batches:
            overrides.append(f"trainer.accumulate_grad_batches={params.accumulate_grad_batches}")
        if params.gradient_clip_val:
            overrides.append(f"trainer.gradient_clip_val={params.gradient_clip_val}")
        if params.precision:
            overrides.append(f"trainer.precision={params.precision}")

        if params.seed is not None:
            overrides.append(f"seed={params.seed}")

        # Experiment settings
        if params.exp_name:
            overrides.append(f"exp_name={params.exp_name}")

        if params.wandb:
            overrides.append(f"logger.wandb.enabled={str(params.wandb).lower()}")

        if params.resume:
            overrides.append(f"resume={params.resume}")

        return overrides

    def _extracted_from_build_predict_overrides(self, params: CommandParams) -> list[str]:
        """Extract common overrides for test and predict commands."""
        result = ["model.encoder.model_name=resnet18", "model.optimizer.lr=0.001"]
        if params.checkpoint_path:
            result.append(f"checkpoint_path={params.checkpoint_path}")
        if params.exp_name:
            result.append(f"exp_name={params.exp_name}")
        return result
