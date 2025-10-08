from __future__ import annotations

import os
from datetime import datetime

import torch
from lightning.pytorch.callbacks import ModelCheckpoint


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
        metrics: dict[str, torch.Tensor] | None = None,
        filename: str | None = None,
    ) -> str:
        """
        Format checkpoint name with additional unique identifiers and model information.
        """
        metrics_dict: dict[str, torch.Tensor] = metrics if metrics is not None else {}
        base_path = super().format_checkpoint_name(metrics_dict, filename)

        dirpath, base_name = os.path.split(base_path)
        stem, ext = os.path.splitext(base_name)

        # Preserve Lightning's reserved "last" checkpoints so cleanup utilities keep working.
        reserved_name = base_name == self.CHECKPOINT_NAME_LAST

        if not reserved_name:
            stem = stem.replace("=", "_")
            model_info = self._get_model_info()
            if model_info:
                stem = f"{stem}_{model_info}"
            if self.add_timestamp:
                stem = f"{stem}_{self.timestamp}"

        final_name = f"{stem}{ext or self.FILE_EXTENSION}"
        return os.path.join(dirpath, final_name) if dirpath else final_name

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
