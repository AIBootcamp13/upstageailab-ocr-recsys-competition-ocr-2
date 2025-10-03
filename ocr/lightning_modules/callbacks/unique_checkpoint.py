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
        Format checkpoint name with additional unique identifiers.
        """
        # Get the base filename from parent class
        base_filename = super().format_checkpoint_name(metrics, *args, **kwargs)

        # Add timestamp to prevent overwrites
        if self.add_timestamp and base_filename.endswith(".ckpt"):
            base_filename = base_filename.replace(".ckpt", f"_{self.timestamp}.ckpt")

        return base_filename
