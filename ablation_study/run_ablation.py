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

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

import wandb
from ocr.lightning_modules import get_pl_modules_by_cfg


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
        wandb.init(
            project=cfg.get("project_name", "OCR_Ablation"),
            name=cfg.get("exp_name", "ablation_run"),
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=[cfg.get("experiment_tag", "ablation")],
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


@hydra.main(config_path="../configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Main function for running ablation studies."""
    print(f"Running ablation study with config: {cfg.get('experiment_tag', 'unnamed')}")

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
