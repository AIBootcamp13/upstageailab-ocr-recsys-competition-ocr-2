import os
import sys

import hydra
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor  # noqa
from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.append(os.getcwd())
from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402

CONFIG_DIR = os.environ.get("OP_CONFIG_DIR") or "../configs"


@hydra.main(config_path=CONFIG_DIR, config_name="train", version_base="1.2")
def train(config):
    """
    Train a OCR model using the provided configuration.

    Args:
        `config` (dict): A dictionary containing configuration settings for training.
    """
    pl.seed_everything(config.get("seed", 42), workers=True)

    model_module, data_module = get_pl_modules_by_cfg(config)

    if config.get("wandb"):
        from lightning.pytorch.loggers import WandbLogger as Logger  # noqa: E402

        from ocr.utils.wandb_utils import generate_run_name  # noqa: E402

        run_name = generate_run_name(config)
        logger = Logger(
            run_name,
            project=config.project_name,
            config=dict(config),
        )
    else:
        from lightning.pytorch.loggers.tensorboard import (  # noqa: E402
            TensorBoardLogger,
        )

        logger = TensorBoardLogger(
            save_dir=config.log_dir,
            name=config.exp_name,
            version=config.exp_version,
            default_hp_metric=False,
        )

    checkpoint_path = config.checkpoint_dir

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=checkpoint_path, save_top_k=3, monitor="val/loss", mode="min"
        ),
    ]

    # Add wandb image logging callback if wandb is enabled
    if config.get("wandb"):
        from ocr.lightning_modules.callbacks.wandb_image_logging import (  # noqa: E402
            WandbImageLoggingCallback,
        )

        callbacks.append(
            WandbImageLoggingCallback(log_every_n_epochs=5)
        )  # Log every 5 epochs

    trainer = pl.Trainer(**config.trainer, logger=logger, callbacks=callbacks)

    trainer.fit(
        model_module,
        data_module,
        ckpt_path=config.get("resume", None),
    )
    trainer.test(
        model_module,
        data_module,
    )

    # Finalize wandb run if wandb was used
    if config.get("wandb"):
        from ocr.utils.wandb_utils import finalize_run  # noqa: E402

        # Get final validation loss as a simple metric for finalization
        final_loss = trainer.callback_metrics.get("val/loss", 0.0)
        try:
            final_loss = float(
                final_loss.item() if hasattr(final_loss, "item") else final_loss
            )
        except:
            final_loss = 0.0
        finalize_run(final_loss)


if __name__ == "__main__":
    train()
