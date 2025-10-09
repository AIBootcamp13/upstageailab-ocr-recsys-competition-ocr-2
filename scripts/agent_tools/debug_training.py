#!/usr/bin/env python3
"""
Debug script to test if training and validation work with minimal setup.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import hydra

# Albumentations imports
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader

from ocr.datasets.base import OCRDataset

# Import project modules
from ocr.lightning_modules.ocr_pl import OCRLightningModule


class DebugCallback(Callback):
    """Callback to debug training and validation."""

    def on_train_start(self, trainer, pl_module):
        print("🚀 Training started")

    def on_validation_start(self, trainer, pl_module):
        print("🔍 Validation started")

    def on_validation_end(self, trainer, pl_module):
        print("✅ Validation completed")
        if hasattr(pl_module, "logged_metrics"):
            metrics = dict(pl_module.logged_metrics)
            val_metrics = {k: v for k, v in metrics.items() if k.startswith("val/")}
            if val_metrics:
                print(f"📊 Validation metrics: {val_metrics}")
            else:
                print("⚠️  No validation metrics found")

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"📈 Epoch {trainer.current_epoch} completed")

    def on_exception(self, trainer, pl_module, exception):
        print(f"💥 Exception during training: {exception}")
        import traceback

        traceback.print_exc()


def create_minimal_dataloaders():
    """Create minimal dataloaders for testing."""
    # Use existing data paths
    train_images = Path("data/datasets/images/train")
    train_annotations = Path("data/datasets/jsons/train.json")
    val_images = Path("data/datasets/images/val")
    val_annotations = Path("data/datasets/jsons/val.json")

    if not all(p.exists() for p in [train_images, train_annotations, val_images, val_annotations]):
        print("❌ Required data files not found")
        return None, None

    # Minimal transforms (just normalization)

    transform = Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])

    try:
        train_dataset = OCRDataset(image_path=str(train_images), annotation_path=str(train_annotations), transform=transform)

        val_dataset = OCRDataset(image_path=str(val_images), annotation_path=str(val_annotations), transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

        return train_loader, val_loader

    except Exception as e:
        print(f"❌ Failed to create dataloaders: {e}")
        return None, None


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    print("🔧 Starting debug training...")

    # Create minimal dataloaders
    train_loader, val_loader = create_minimal_dataloaders()
    if train_loader is None or val_loader is None:
        return

    # Create model
    try:
        model = OCRLightningModule(cfg.model, cfg.training)
        print("✅ Model created")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return

    # Create callbacks
    debug_callback = DebugCallback()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./debug_checkpoints",
        filename="debug-{epoch:02d}-{val/hmean:.3f}",
        monitor="val/hmean",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    # Create trainer with minimal settings
    trainer = Trainer(
        max_epochs=1,  # Just 1 epoch for testing
        limit_train_batches=2,  # Just 2 batches for testing
        limit_val_batches=2,  # Just 2 validation batches for testing
        callbacks=[debug_callback, checkpoint_callback],
        logger=False,  # Disable logging for simplicity
        enable_progress_bar=True,
        accelerator="cpu",  # Use CPU for testing
    )

    print("🏃 Starting training...")
    try:
        trainer.fit(model, train_loader, val_loader)
        print("🎉 Training completed successfully!")
    except Exception as e:
        print(f"💥 Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
