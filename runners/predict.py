import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

# Setup project paths automatically
from ocr.utils.path_utils import get_path_resolver, setup_project_paths

setup_project_paths()

from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402


@hydra.main(config_path=str(get_path_resolver().config.config_dir), config_name="predict", version_base="1.2")
def predict(config: DictConfig):
    """
    Train a OCR model using the provided configuration.

    Args:
        `config` (dict): A dictionary containing configuration settings for predict.
    """
    pl.seed_everything(config.get("seed", 42), workers=True)

    model_module, data_module = get_pl_modules_by_cfg(config)

    # --- Callback Configuration ---
    # Instantiate callbacks from config to match the checkpoint
    callbacks = []
    if config.get("callbacks"):
        for _, cb_conf in config.callbacks.items():
            if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
                print(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    trainer = pl.Trainer(logger=False, callbacks=callbacks)

    ckpt_path = config.get("checkpoint_path")
    assert ckpt_path, "checkpoint_path must be provided for prediction"

    trainer.predict(
        model_module,
        data_module,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    predict()
