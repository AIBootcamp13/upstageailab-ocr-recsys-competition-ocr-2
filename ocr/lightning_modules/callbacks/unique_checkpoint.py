from __future__ import annotations

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

    def format_checkpoint_name(self, metrics: dict[str, torch.Tensor], *args, **kwargs) -> str:
        """
        Format checkpoint name with additional unique identifiers and model information.
        """
        # Extract metrics values for template formatting
        epoch = metrics.get("epoch", torch.tensor(0)).item()
        step = metrics.get("step", torch.tensor(0)).item()

        # Format the template manually since parent class doesn't do it
        if hasattr(self, "filename") and self.filename:
            template = self.filename
            try:
                formatted_name = template.format(epoch=epoch, step=step)
                # Remove .ckpt extension if present in template, we'll add it back
                if formatted_name.endswith(".ckpt"):
                    formatted_name = formatted_name[:-5]
            except (KeyError, ValueError):
                # Fallback if formatting fails
                formatted_name = f"epoch_{epoch:02d}_step_{step:06d}"
        else:
            formatted_name = f"epoch_{epoch:02d}_step_{step:06d}"

        # Add model information if available
        model_info = self._get_model_info()
        if model_info:
            formatted_name = f"{formatted_name}_{model_info}"

        # Add timestamp to prevent overwrites
        if self.add_timestamp:
            formatted_name = f"{formatted_name}_{self.timestamp}"

        # Add .ckpt extension
        if not formatted_name.endswith(".ckpt"):
            formatted_name += ".ckpt"

        return formatted_name

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
            pass

        return None
