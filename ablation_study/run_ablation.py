#!/usr/bin/env python3
"""
Ablation Study Runner

This script provides a systematic way to run ablation studies by leveraging Hydra's
multirun capabilities and wandb for experiment tracking.

Usage:
    # Run learning rate ablation
    python run_ablation.py +ablation=learning_rate

    # Run batch size ablation
    python run_ablation.py +ablation=batch_size

    # Run custom ablation with specific overrides
    python run_ablation.py training.learning_rate=1e-3,5e-4,1e-4

    # Run model architecture ablation
    python run_ablation.py +ablation=model_architecture

Example:
    python run_ablation.py +ablation=learning_rate experiment_tag=lr_ablation
"""

import sys
from typing import Any

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

import wandb
from ocr.lightning_modules import get_pl_modules_by_cfg
from ocr.utils.path_utils import get_path_resolver


def _normalize_multirun_flag() -> None:
    """Ensure the Hydra multirun flag precedes overrides for backward compatibility."""
    argv = sys.argv
    # Handle short and long forms.
    for flag in ("-m", "--multirun"):
        if flag in argv:
            idx = argv.index(flag)
            # Reorder only when the flag isn't already the first argument after the filename.
            if idx > 1:
                pre = argv[1:idx]
                post = argv[idx + 1 :]
                sys.argv = [argv[0], flag, *pre, *post]
            break


_normalize_multirun_flag()


def run_single_experiment(cfg: DictConfig) -> dict:
    """
    Run a single experiment with the given configuration.

    Args:
        cfg: Hydra configuration object

    Returns:
        Dictionary with experiment results
    """
    # Set seed for reproducibility
    pl.seed_everything(cfg.get("seed", 42), workers=True)

    # Initialize wandb if enabled
    if cfg.get("wandb", False):
        project_name = cfg.get("project_name")
        if not project_name and cfg.get("logger"):
            project_name = cfg.logger.get("project_name")
        project_name = project_name or "OCR_Ablation"

        exp_name = cfg.get("exp_name", "ablation_run")
        experiment_tag = (
            cfg.get("experiment_tag") or (cfg.get("ablation", {}).get("experiment_tag") if cfg.get("ablation") else None) or exp_name
        )

        wandb.init(
            project=project_name,
            name=exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
            tags=[experiment_tag],
        )

    try:
        # Get model and data modules
        model_module, data_module = get_pl_modules_by_cfg(cfg)

        # Setup trainer
        trainer = pl.Trainer(
            max_epochs=cfg.trainer.get("max_epochs", 10),
            log_every_n_steps=cfg.trainer.get("log_every_n_steps", 50),
            check_val_every_n_epoch=cfg.trainer.get("check_val_every_n_epoch", 1),
            enable_progress_bar=cfg.trainer.get("enable_progress_bar", True),
            logger=pl.loggers.WandbLogger() if cfg.get("wandb", False) else True,
        )

        # Train the model
        trainer.fit(model_module, data_module)

        # Get final metrics
        final_metrics = {}
        if hasattr(trainer, "callback_metrics"):
            final_metrics = dict(trainer.callback_metrics)

        # Test if test data is available
        if data_module.test_dataloader() is not None:
            test_results = trainer.test(model_module, data_module)
            final_metrics.update({"test_" + k: v for k, v in test_results[0].items()})

        return {
            "status": "success",
            "metrics": final_metrics,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
    finally:
        if wandb.run:
            wandb.finish()


@hydra.main(config_path=str(get_path_resolver().config.config_dir), config_name="train", version_base="1.2")
def main(cfg: DictConfig) -> dict[str, Any]:
    """Main function for running ablation studies."""
    experiment_tag = (
        cfg.get("experiment_tag") or (cfg.get("ablation", {}).get("experiment_tag") if cfg.get("ablation") else None) or "unnamed"
    )
    print(f"Running ablation study with config: {experiment_tag}")

    # Run the experiment
    result = run_single_experiment(cfg)

    # Print results
    if result["status"] == "success":
        print("Experiment completed successfully!")
        print("Final metrics:")
        for key, value in result["metrics"].items():
            print(f"  {key}: {value}")
    else:
        print(f"Experiment failed: {result['error']}")

    return result


if __name__ == "__main__":
    main()
