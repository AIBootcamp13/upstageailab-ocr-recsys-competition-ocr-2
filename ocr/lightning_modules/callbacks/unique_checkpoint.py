from __future__ import annotations

import os
from datetime import datetime

import torch
from lightning.pytorch.callbacks import ModelCheckpoint

import wandb


class UniqueModelCheckpoint(ModelCheckpoint):
    """
    Enhanced ModelCheckpoint with structured, hierarchical naming convention.

    Implements the naming scheme:
    <experiment_tag>-<model>_<phase>_<timestamp>/checkpoints/<type>-<epoch>_<step>_<metric>.ckpt

    This callback extends PyTorch Lightning's ModelCheckpoint to provide:
    - Clear, hierarchical checkpoint organization
    - Easy identification of experiments and training phases
    - Automatic metadata inclusion (model, epoch, step, metrics)
    - Prevention of overwrites through unique identifiers
    - Improved searchability and filtering

    Example checkpoint names:
    - Epoch checkpoint: epoch-03_step-000103.ckpt
    - Last checkpoint: last.ckpt
    - Best checkpoint: best-hmean-0.8920.ckpt
    """

    def __init__(
        self,
        *args,
        add_timestamp: bool = True,
        experiment_tag: str | None = None,
        training_phase: str = "training",
        **kwargs,
    ):
        """
        Initialize checkpoint callback with enhanced naming.

        Args:
            add_timestamp: Whether to add timestamp to directory structure
            experiment_tag: Unique identifier for the experiment (e.g., "ocr_pl_refactor_phase1")
            training_phase: Stage of the experiment (e.g., "training", "validation", "finetuning")
            **kwargs: Additional arguments passed to ModelCheckpoint
        """
        super().__init__(*args, **kwargs)
        self.add_timestamp = add_timestamp
        self.experiment_tag = experiment_tag
        self.training_phase = training_phase

        # Generate unique identifier once at initialization
        if self.add_timestamp:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def format_checkpoint_name(
        self,
        metrics: dict | None = None,
        filename: str | None = None,
        ver: int | None = None,
        prefix: str | None = None,
    ) -> str:
        """
        Formats checkpoint name using hierarchical naming convention.

        Implements the scheme:
        - Epoch checkpoint: epoch-<epoch>_step-<step>.ckpt
        - Last checkpoint: last.ckpt
        - Best checkpoint: best-<metric_name>-<value>.ckpt

        Args:
            metrics: Dictionary of metric values
            filename: Template filename from config (used for determining checkpoint type)
            ver: Version number (optional)
            prefix: Additional prefix (optional)

        Returns:
            Formatted checkpoint filename path
        """
        # Trainer might not be attached during initialization
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return super().format_checkpoint_name(metrics or {}, filename, ver, prefix)

        # 1. Get authoritative epoch and step directly from the trainer
        epoch = trainer.current_epoch
        step = trainer.global_step

        # 2. Determine checkpoint type and build the filename
        is_best_checkpoint = filename and "best" in filename.lower()
        is_last_checkpoint = filename and filename.lower() == "last"

        if is_last_checkpoint:
            # Last checkpoint: simply "last.ckpt"
            stem = "last"
        elif is_best_checkpoint:
            # Best checkpoint: best-<metric_name>-<value>.ckpt
            stem = "best"
            if self.monitor and metrics:
                metric_val = metrics.get(self.monitor)
                if isinstance(metric_val, torch.Tensor):
                    # Clean up the metric name (e.g., "val/hmean" -> "hmean")
                    metric_name_clean = self.monitor.split("/")[-1]
                    stem = f"best-{metric_name_clean}-{metric_val.item():.4f}"
        else:
            # Epoch checkpoint: epoch-<epoch>_step-<step>.ckpt
            stem = f"epoch-{epoch:02d}_step-{step:06d}"

            # Optionally add metric value for epoch checkpoints
            if self.auto_insert_metric_name and metrics and self.monitor:
                metric_val = metrics.get(self.monitor)
                if isinstance(metric_val, torch.Tensor):
                    metric_name_clean = self.monitor.split("/")[-1]
                    stem = f"{stem}_{metric_name_clean}-{metric_val.item():.4f}"

        # 3. Add prefix if provided
        if prefix:
            stem = f"{prefix}_{stem}"

        # 4. Add version if provided
        if ver is not None:
            stem = f"{stem}_v{ver}"

        # 5. Combine and return the final path
        dirpath = self.dirpath or "."
        final_name = f"{stem}{self.FILE_EXTENSION}"
        return os.path.join(dirpath, final_name)

    def _get_model_info(self) -> dict[str, str | None]:
        """
        Extract model architecture and encoder information.

        Returns:
            Dictionary with 'architecture' and 'encoder' keys
        """
        info: dict[str, str | None] = {"architecture": None, "encoder": None}

        try:
            trainer = getattr(self, "trainer", None)
            if trainer is not None and hasattr(trainer, "model"):
                model = trainer.model

                # Extract architecture
                if hasattr(model, "architecture_name"):
                    info["architecture"] = model.architecture_name
                else:
                    info["architecture"] = getattr(model, "_architecture_name", None)

                # Extract encoder information
                if hasattr(model, "encoder") and hasattr(model.encoder, "model_name"):
                    info["encoder"] = model.encoder.model_name
                elif hasattr(model, "component_overrides") and "encoder" in model.component_overrides:
                    encoder_override = model.component_overrides["encoder"]
                    if isinstance(encoder_override, dict) and "model_name" in encoder_override:
                        info["encoder"] = encoder_override["model_name"]

        except Exception:
            # If anything fails, return None values
            pass

        return info

    def _setup_dirpath(self) -> str:
        """
        Set up the directory path using the hierarchical naming scheme.

        Creates directory structure:
        outputs/<experiment_tag>-<model>_<phase>_<timestamp>/checkpoints/

        Returns:
            The formatted directory path
        """
        if not self.dirpath:
            return "."

        # Convert dirpath to string if it's a Path object
        dirpath_str = str(self.dirpath)

        # If add_timestamp is enabled, enhance the directory name
        if self.add_timestamp and self.experiment_tag:
            # Get model information
            model_info = self._get_model_info()
            architecture = model_info.get("architecture") or "unknown"
            encoder = model_info.get("encoder") or "unknown"

            # Clean names (remove special characters)
            import re

            def clean_name(name):
                if name is None or name == "unknown":
                    return name
                return re.sub(r"[^a-zA-Z0-9_-]", "_", str(name))

            arch_clean = clean_name(architecture)
            encoder_clean = clean_name(encoder)

            # Build model identifier
            if encoder_clean and encoder_clean != "unknown":
                model_identifier = f"{arch_clean}-{encoder_clean}"
            else:
                model_identifier = arch_clean

            # Build the enhanced directory name
            dir_name = f"{self.experiment_tag}-{model_identifier}_{self.training_phase}_{self.timestamp}"

            # Replace the exp_name portion in dirpath with our enhanced name
            # Assuming dirpath is like "outputs/{exp_name}/checkpoints"
            parts = dirpath_str.split(os.sep)
            if len(parts) >= 2:
                # Replace the exp_name part (typically second-to-last before /checkpoints)
                parts[-2] = dir_name
                enhanced_dirpath = os.sep.join(parts)
                return enhanced_dirpath

        return dirpath_str

    def setup(self, trainer, pl_module, stage: str | None = None) -> None:
        """
        Setup callback and configure directory structure.

        This is called before training starts, allowing us to configure
        the directory path with model information.
        """
        # Call parent setup with stage
        if stage is not None:
            super().setup(trainer, pl_module, stage)
        else:
            # Avoid the type issue by not passing None
            super().setup(trainer, pl_module, "fit")

        # Update dirpath with enhanced naming if enabled
        if self.add_timestamp and self.experiment_tag:
            enhanced_dirpath = self._setup_dirpath()
            self.dirpath = enhanced_dirpath  # type: ignore

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load state dict but handle dirpath mismatches gracefully.

        During prediction, the dirpath may differ from training, which causes
        warnings. We update the dirpath in the state_dict to match current
        dirpath to avoid warnings.
        """
        # Update dirpath in state_dict to match current dirpath
        if "dirpath" in state_dict and hasattr(self, "dirpath"):
            if state_dict["dirpath"] != self.dirpath:
                state_dict = state_dict.copy()
                state_dict["dirpath"] = self.dirpath

        super().load_state_dict(state_dict)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Log checkpoint directory to wandb when a checkpoint is saved."""
        if wandb.run and self.dirpath:
            wandb.log({"checkpoint_dir": self.dirpath})
        return checkpoint
