from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import wandb
from lightning.pytorch.callbacks import Callback


class WandbCompletionCallback(Callback):
    """Signal run completion to W&B and the filesystem."""

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.fast_dev_run:  # type: ignore[attr-defined]
            print("Skipping completion signal for fast_dev_run.")
            return

        current_run = getattr(wandb, "run", None)
        if current_run:
            try:
                current_run.tags = current_run.tags + ("status:completed",)
                current_run.summary["final_status"] = "success"
                print("Successfully tagged W&B run as completed.")
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: Failed to update W&B run: {exc}")

        try:
            checkpoint_callback = next(
                (cb for cb in trainer.callbacks if isinstance(cb, pl.callbacks.ModelCheckpoint)),  # type: ignore[attr-defined]
                None,
            )
            if checkpoint_callback and checkpoint_callback.dirpath:
                output_dir = Path(checkpoint_callback.dirpath)
                output_dir.mkdir(parents=True, exist_ok=True)
                success_file_path = output_dir / ".SUCCESS"
                success_file_path.touch()
                print(f"Created success sentinel file at: {success_file_path}")
            else:
                print("Warning: Could not find ModelCheckpoint callback to determine output directory for sentinel file.")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Failed to create local sentinel file: {exc}")

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        current_run = getattr(wandb, "run", None)
        if current_run:
            current_run.tags = current_run.tags + ("status:failed",)
            current_run.summary["final_status"] = "failed"

        checkpoint_callback = next(
            (cb for cb in trainer.callbacks if isinstance(cb, pl.callbacks.ModelCheckpoint)),  # type: ignore[attr-defined]
            None,
        )
        if checkpoint_callback and checkpoint_callback.dirpath:
            failure_file_path = Path(checkpoint_callback.dirpath) / ".FAILURE"
            with open(failure_file_path, "w", encoding="utf-8") as handle:
                handle.write(str(exception))
            print(f"Created failure sentinel file at: {failure_file_path}")
