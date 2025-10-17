import warnings

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

# Setup project paths automatically
from ocr.utils.path_utils import get_path_resolver, setup_project_paths

setup_project_paths()

# Suppress known wandb Pydantic compatibility warnings
# This is a known issue where wandb uses incorrect Field() syntax in Annotated types
# The warnings come from Pydantic when processing wandb's type annotations
warnings.filterwarnings("ignore", message=r"The '(repr|frozen)' attribute.*Field.*function.*no effect", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*(repr|frozen).*Field.*function.*no effect", category=UserWarning)

# Also suppress by category for more reliable filtering
try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning

    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except ImportError:
    # Pydantic v1 doesn't have this warning class
    pass

from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402
from ocr.lightning_modules.utils.model_utils import load_state_dict_with_fallback  # noqa: E402


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

    # Load checkpoint manually to handle PyTorch 2.6 compatibility issues
    import torch

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        # Fallback for PyTorch 2.6+ compatibility with OmegaConf objects
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Load the state dict manually
    if "state_dict" in checkpoint:
        missing, unexpected = load_state_dict_with_fallback(model_module, checkpoint["state_dict"])
        print(f"Loaded state dict from checkpoint (missing: {len(missing)}, unexpected: {len(unexpected)})")
    else:
        print("Warning: No state_dict found in checkpoint")

    trainer.predict(
        model_module,
        data_module,
        # Don't pass ckpt_path since we already loaded the checkpoint manually
    )


if __name__ == "__main__":
    predict()
