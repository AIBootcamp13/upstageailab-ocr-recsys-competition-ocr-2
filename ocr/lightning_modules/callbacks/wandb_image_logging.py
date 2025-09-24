import lightning.pytorch as pl
import torch

from ocr.utils.wandb_utils import log_validation_images


class WandbImageLoggingCallback(pl.Callback):
    """Callback to log validation images with bounding boxes to Weights & Biases."""

    def __init__(self, log_every_n_epochs: int = 1, max_images: int = 8):
        self.log_every_n_epochs = log_every_n_epochs
        self.max_images = max_images

    def on_validation_epoch_end(self, trainer, pl_module):
        # Only log every N epochs to avoid too much data
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # For now, we'll skip the complex image logging and just log that validation completed
        # This is a placeholder - full image logging would require more integration work
        # to properly collect images, ground truth, and predictions during validation

        # Log a simple metric to show the callback is working
        if hasattr(trainer.logger, "experiment"):
            try:
                import wandb

                if wandb.run:
                    wandb.log(
                        {
                            "validation_epoch": trainer.current_epoch,
                            "validation_logged": True,
                        }
                    )
            except ImportError:
                pass
