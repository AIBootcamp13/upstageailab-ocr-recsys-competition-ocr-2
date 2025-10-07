import hydra
import lightning.pytorch as pl

# Setup project paths automatically
from ocr.utils.path_utils import get_path_resolver, setup_project_paths

setup_project_paths()

from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402


@hydra.main(config_path=str(get_path_resolver().config.config_dir), config_name="test", version_base="1.2")
def test(config):
    """
    Train a OCR model using the provided configuration.

    Args:
        `config` (dict): A dictionary containing configuration settings for test.
    """
    pl.seed_everything(config.get("seed", 42), workers=True)

    model_module, data_module = get_pl_modules_by_cfg(config)

    # --- Callback Configuration ---
    # This is the new, Hydra-native way to handle callbacks.
    # It iterates through the 'callbacks' config group and instantiates each one.
    callbacks = []
    if config.get("callbacks"):
        from omegaconf import DictConfig  # noqa: E402

        for _, cb_conf in config.callbacks.items():
            if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
                print(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    if config.logger.wandb:
        from lightning.pytorch.loggers import WandbLogger as Logger  # noqa: E402
        from omegaconf import OmegaConf  # noqa: E402

        # Properly serialize config for wandb, handling hydra interpolations
        try:
            # Try to resolve interpolations for cleaner config
            wandb_config = OmegaConf.to_container(config, resolve=True)
        except Exception:
            # Fall back to unresolved config if resolution fails
            wandb_config = OmegaConf.to_container(config, resolve=False)

        logger = Logger(config.exp_name, project=config.logger.project_name, config=wandb_config)
    else:
        from lightning.pytorch.loggers.tensorboard import TensorBoardLogger  # noqa: E402

        logger = TensorBoardLogger(
            save_dir=config.paths.log_dir,
            name=config.exp_name,
            version=config.logger.exp_version,
            default_hp_metric=False,
        )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
    )

    ckpt_path = config.get("checkpoint_path")
    assert ckpt_path is not None, "checkpoint_path must be provided for test"

    trainer.test(
        model_module,
        data_module,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    test()
