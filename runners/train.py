import math
import os
import signal
import sys

# Setup project paths automatically
import hydra
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from omegaconf import DictConfig

import wandb

# Clean up any lingering W&B session to prevent warnings
wandb.finish()

from ocr.utils.path_utils import get_path_resolver, setup_project_paths

setup_project_paths()

from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402

_shutdown_in_progress = False
trainer = None
data_module = None


def signal_handler(signum, frame):
    """Handle interrupt signals to ensure graceful shutdown without recursion."""
    global _shutdown_in_progress
    if _shutdown_in_progress:
        return
    _shutdown_in_progress = True

    print(f"\nReceived signal {signum}. Shutting down gracefully...")

    try:
        if trainer is not None:
            print("Stopping trainer...")
            # Lightning handles SIGINT/SIGTERM for graceful shutdown
    except Exception as e:
        print(f"Error during trainer shutdown: {e}")

    try:
        if data_module is not None:
            print("Cleaning up data module...")
            # DataLoader workers will be cleaned up by process shutdown
    except Exception as e:
        print(f"Error during data module cleanup: {e}")

    # Do not send SIGTERM to our own process group to avoid recursive signals
    print("Shutdown complete.")
    sys.exit(1)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Avoid creating a new process group here; the caller (UI) manages process groups


@hydra.main(config_path=str(get_path_resolver().config.config_dir), config_name="train", version_base="1.2")
def train(config: DictConfig):
    """
    Train a OCR model using the provided configuration.

    Args:
        `config` (DictConfig): A dictionary containing configuration settings for training.
    """
    global trainer, data_module

    pl.seed_everything(config.get("seed", 42), workers=True)

    # Enable Tensor Core utilization for better GPU performance
    import torch

    torch.set_float32_matmul_precision("high")

    runtime_cfg = config.get("runtime") or {}
    auto_gpu_devices = runtime_cfg.get("auto_gpu_devices", True)
    preferred_strategy = runtime_cfg.get("ddp_strategy", "ddp_find_unused_parameters_false")
    min_auto_devices = runtime_cfg.get("min_auto_devices", 2)

    def _normalize_device_request(value):
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return value
        return value

    if auto_gpu_devices and config.trainer.get("accelerator", "cpu") == "gpu" and torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        requested_devices = _normalize_device_request(config.trainer.get("devices"))
        if available_gpus >= max(1, min_auto_devices):
            if requested_devices in (None, 1):
                config.trainer.devices = available_gpus
                strategy_cfg = config.trainer.get("strategy")
                if strategy_cfg in (None, "auto"):
                    config.trainer.strategy = preferred_strategy
                print(f"[AutoParallel] Scaling to {available_gpus} GPUs with strategy='{config.trainer.strategy}'.")
            elif isinstance(requested_devices, int) and requested_devices > available_gpus:
                config.trainer.devices = available_gpus
                print(
                    f"[AutoParallel] Requested {requested_devices} GPUs, but only {available_gpus} detected. "
                    f"Falling back to {available_gpus}."
                )

    model_module, data_module = get_pl_modules_by_cfg(config)

    # Ensure key output directories exist before creating callbacks
    try:
        os.makedirs(config.paths.log_dir, exist_ok=True)
        os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
        # Some workflows also expect a submission dir
        if hasattr(config.paths, "submission_dir"):
            os.makedirs(config.paths.submission_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: failed to ensure output directories exist: {e}")

    if config.logger.wandb:
        from lightning.pytorch.loggers import WandbLogger as Logger  # noqa: E402
        from omegaconf import OmegaConf  # noqa: E402

        from ocr.utils.wandb_utils import generate_run_name, load_env_variables  # noqa: E402

        # Load environment variables from .env.local/.env
        load_env_variables()

        run_name = generate_run_name(config)

        # Properly serialize config for wandb, handling hydra interpolations
        try:
            # Try to resolve interpolations for cleaner config
            wandb_config = OmegaConf.to_container(config, resolve=True)
        except Exception:
            # Fall back to unresolved config if resolution fails
            wandb_config = OmegaConf.to_container(config, resolve=False)

        logger = Logger(
            run_name,
            project=config.logger.project_name,
            config=wandb_config,
        )
    else:
        from lightning.pytorch.loggers.tensorboard import TensorBoardLogger  # noqa: E402

        logger = TensorBoardLogger(
            save_dir=config.paths.log_dir,
            name=config.exp_name,
            version=config.logger.exp_version,
            default_hp_metric=False,
        )

    # --- Callback Configuration ---
    # This is the new, Hydra-native way to handle callbacks.
    # It iterates through the 'callbacks' config group and instantiates each one.
    callbacks = []
    if config.get("callbacks"):
        for _, cb_conf in config.callbacks.items():
            if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
                print(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Always add LearningRateMonitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Preprocess trainer config for PyTorch Lightning compatibility
    trainer_config = dict(config.trainer)
    if trainer_config.get("max_steps") is None:
        trainer_config["max_steps"] = -1  # PyTorch Lightning expects -1 for unlimited steps

    trainer = pl.Trainer(**trainer_config, logger=logger, callbacks=callbacks)

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
    if config.logger.wandb:
        from ocr.utils.wandb_utils import finalize_run  # noqa: E402

        metrics: dict[str, float] = {}

        def _to_float(value) -> float | None:
            try:
                if isinstance(value, torch.Tensor):
                    return float(value.detach().cpu().item())
                if hasattr(value, "item"):
                    item_val = value.item()
                    return float(item_val)
                return float(value)
            except (TypeError, ValueError):
                return None

        for key, value in trainer.callback_metrics.items():
            cast_value = _to_float(value)
            if cast_value is not None and math.isfinite(cast_value):
                metrics[key] = cast_value

        finalize_run(metrics)


if __name__ == "__main__":
    train()
