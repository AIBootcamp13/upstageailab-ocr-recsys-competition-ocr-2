from __future__ import annotations

import os
from datetime import datetime

import torch
from lightning.pytorch.callbacks import ModelCheckpoint

import wandb


class UniqueModelCheckpoint(ModelCheckpoint):
    """
    Enhanced ModelCheckpoint that prevents overwrites by adding unique identifiers.

    This callback extends PyTorch Lightning's ModelCheckpoint to ensure that
    checkpoints with the same epoch/step but different configurations don't
    overwrite each other.
    """

    def __init__(self, *args, add_timestamp: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_timestamp = add_timestamp

        # Generate unique identifier once at initialization
        if self.add_timestamp:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def format_checkpoint_name(
        self,
        metrics: dict | None = None,
        filename: str | None = None,
    ) -> str:
        """
        Formats the checkpoint name robustly using the trainer's state.

        This implementation manually constructs the filename to avoid issues with
        pre-formatted strings in the metrics dictionary. It directly accesses
        `trainer.current_epoch` and `trainer.global_step` for accurate, numerical values.
        """
        # Trainer might not be attached during initialization
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return super().format_checkpoint_name(metrics or {}, filename)

        # 1. Get authoritative epoch and step directly from the trainer
        epoch = trainer.current_epoch
        step = trainer.global_step

        # 2. Build the core filename string
        # We ignore the `filename` template from the config for epoch/step
        # as it's the source of the original bug.
        stem = f"epoch_{epoch:02d}_step_{step:06d}"

        # 3. Add the monitored metric value if enabled
        if self.auto_insert_metric_name and metrics and self.monitor:
            # Ensure monitor key exists and value is a tensor
            metric_val = metrics.get(self.monitor)
            if isinstance(metric_val, torch.Tensor):
                # Clean up the metric name for the filename (e.g., "val/hmean" -> "val_hmean")
                metric_name_clean = self.monitor.replace("/", "_")
                stem = f"{stem}_{metric_name_clean}_{metric_val.item():.4f}"

        # 4. Add unique identifiers (model info, timestamp)
        # This logic is preserved from your original implementation.
        dirpath = self.dirpath or "."
        is_best_checkpoint = "best" in (filename or "").lower()

        if is_best_checkpoint:
            stem = f"best_{stem}"  # Prepend "best" for clarity

        model_info = self._get_model_info()
        if model_info:
            stem = f"{stem}_{model_info}"

        if self.add_timestamp:
            stem = f"{stem}_{self.timestamp}"

        # 5. Combine and return the final path
        final_name = f"{stem}{self.FILE_EXTENSION}"
        return os.path.join(dirpath, final_name)

    def _get_model_info(self) -> str | None:
        """
        Extract model architecture and encoder information for filename.
        """
        try:
            # Access the trainer and model
            trainer = getattr(self, "trainer", None)
            if trainer is not None and hasattr(trainer, "model"):
                model = trainer.model
                if hasattr(model, "architecture_name"):
                    arch = model.architecture_name
                else:
                    arch = getattr(model, "_architecture_name", None) or "unknown"

                # Try to get encoder information
                encoder_name = None
                if hasattr(model, "encoder") and hasattr(model.encoder, "model_name"):
                    encoder_name = model.encoder.model_name
                elif hasattr(model, "component_overrides") and "encoder" in model.component_overrides:
                    encoder_override = model.component_overrides["encoder"]
                    if isinstance(encoder_override, dict) and "model_name" in encoder_override:
                        encoder_name = encoder_override["model_name"]

                # Clean the names (remove special characters)
                def clean_name(name):
                    if name is None:
                        return None
                    # Replace special characters with underscores
                    import re

                    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(name))

                arch_clean = clean_name(arch)
                encoder_clean = clean_name(encoder_name)

                if arch_clean and encoder_clean:
                    return f"{arch_clean}_{encoder_clean}"
                elif arch_clean:
                    return arch_clean
                elif encoder_clean:
                    return encoder_clean

        except Exception:
            # If anything fails, just return None - don't break checkpointing
            print("Failed to get model info for checkpoint naming.")
        return None

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
