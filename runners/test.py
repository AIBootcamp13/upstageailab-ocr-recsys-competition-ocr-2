import hydra
import lightning.pytorch as pl

# Setup project paths automatically
from ocr.utils.path_utils import get_path_resolver, setup_paths

setup_paths()

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

    if config.logger.wandb:
        from lightning.pytorch.loggers import WandbLogger as Logger  # noqa: E402

        logger = Logger(config.exp_name, project=config.logger.project_name, config=dict(config))
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
