from collections import OrderedDict, defaultdict
from typing import Any

import lightning.pytorch as pl
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader

from ocr.evaluation import CLEvalEvaluator
from ocr.lightning_modules.loggers import WandbProblemLogger
from ocr.lightning_modules.utils import CheckpointHandler, extract_metric_kwargs, extract_normalize_stats
from ocr.lightning_modules.utils.model_utils import load_state_dict_with_fallback
from ocr.metrics import CLEvalMetric
from ocr.utils.orientation import remap_polygons
from ocr.utils.submission import SubmissionWriter


class OCRPLModule(pl.LightningModule):
    def __init__(self, model, dataset, config, metric_cfg: DictConfig | None = None):
        super().__init__()
        self.model = model
        # Compile the model for better performance
        if hasattr(config, "compile_model") and config.compile_model:
            # Configure torch.compile to handle scalar outputs better
            import torch._dynamo

            torch._dynamo.config.capture_scalar_outputs = True
            self.model = torch.compile(self.model, mode="default")
        self.dataset = dataset
        self.metric_cfg = metric_cfg
        self.metric_kwargs = extract_metric_kwargs(metric_cfg)
        self.metric = instantiate(metric_cfg) if metric_cfg is not None else CLEvalMetric(**self.metric_kwargs)
        self.config = config
        self.lr_scheduler = None
        self._normalize_mean, self._normalize_std = extract_normalize_stats(config)

        self.valid_evaluator = CLEvalEvaluator(self.dataset["val"], self.metric_kwargs, mode="val") if "val" in self.dataset else None
        self.test_evaluator = CLEvalEvaluator(self.dataset["test"], self.metric_kwargs, mode="test") if "test" in self.dataset else None
        self.predict_step_outputs: OrderedDict[str, Any] = OrderedDict()
        self.validation_step_outputs: OrderedDict[str, Any] = OrderedDict()

        # Initialize helper classes
        self.wandb_logger = WandbProblemLogger(config, self._normalize_mean, self._normalize_std)
        self.submission_writer = SubmissionWriter(config)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state dict with fallback handling for different checkpoint formats."""
        return load_state_dict_with_fallback(self, state_dict, strict=strict)

    def forward(self, x):
        return self.model(return_loss=False, **x)

    def training_step(self, batch, batch_idx):
        pred = self.model(**batch)
        self.log("train/loss", pred["loss"], batch_size=batch["images"].shape[0])
        for key, value in pred["loss_dict"].items():
            self.log(f"train/{key}", value, batch_size=batch["images"].shape[0])
        return pred["loss"]

    def validation_step(self, batch, batch_idx):
        pred = self.model(**batch)
        self.log("val_loss", pred["loss"], batch_size=batch["images"].shape[0])
        for key, value in pred["loss_dict"].items():
            self.log(f"val_{key}", value, batch_size=batch["images"].shape[0])

        boxes_batch, _ = self.model.get_polygons_from_maps(batch, pred)
        predictions: list[dict[str, Any]] = []
        for idx, boxes in enumerate(boxes_batch):
            normalized_boxes = [np.asarray(box, dtype=np.float32).reshape(-1, 2) for box in boxes]
            prediction_entry = {
                "boxes": normalized_boxes,
                "orientation": batch.get("orientation", [1])[idx] if "orientation" in batch else 1,
                "raw_size": tuple(batch.get("raw_size", [(0, 0)])[idx]) if "raw_size" in batch else None,
                "canonical_size": tuple(batch.get("canonical_size", [None])[idx])
                if "canonical_size" in batch
                else None,  # BUG REPORTED Error: `TypeError: 'int' object is not iterable` in canonical_size handling
                "image_path": batch.get("image_path", [None])[idx] if "image_path" in batch else None,
            }
            predictions.append(prediction_entry)

        # Store predictions for wandb image logging callback
        for idx, prediction_entry in enumerate(predictions):
            filename = batch["image_filename"][idx]
            self.validation_step_outputs[filename] = prediction_entry

        if self.valid_evaluator is not None:
            self.valid_evaluator.update(batch["image_filename"], predictions)

        # Compute per-batch validation metrics
        batch_metrics = self._compute_batch_metrics(batch, predictions)
        self.log(f"batch_{batch_idx}/recall", batch_metrics["recall"], batch_size=batch["images"].shape[0])
        self.log(f"batch_{batch_idx}/precision", batch_metrics["precision"], batch_size=batch["images"].shape[0])
        self.log(f"batch_{batch_idx}/hmean", batch_metrics["hmean"], batch_size=batch["images"].shape[0])

        # Log problematic batch images
        self.wandb_logger.log_if_needed(batch, predictions, batch_metrics, batch_idx)

        return pred["loss"]

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs.clear()
        self.wandb_logger.reset_epoch_counter()

    def _compute_batch_metrics(self, batch, predictions: list[dict[str, Any]]):
        """Compute validation metrics for a batch of images."""
        cleval_metrics = defaultdict(list)

        # Get the underlying dataset (handle Subset case)
        val_dataset = self.dataset["val"]
        if hasattr(val_dataset, "dataset"):
            # It's a Subset, get the underlying dataset
            val_dataset = val_dataset.dataset

        for idx, prediction_entry in enumerate(predictions):
            filename = batch["image_filename"][idx]
            if filename not in val_dataset.anns:
                continue
            gt_words = val_dataset.anns[filename]

            orientation = prediction_entry.get("orientation", 1)
            if "orientation" in batch:
                orientation = prediction_entry.get("orientation", batch["orientation"][idx])

            raw_size = prediction_entry.get("raw_size")
            if raw_size is None and "raw_size" in batch:
                raw_size = batch["raw_size"][idx]

            image_path = prediction_entry.get("image_path")
            if image_path is None and "image_path" in batch:
                image_path = batch["image_path"][idx]
            if image_path is None and hasattr(self.dataset["val"], "image_path"):
                image_path = self.dataset["val"].image_path / filename  # type: ignore[attr-defined]

            raw_width, raw_height = 0, 0
            if raw_size is not None and all(dim for dim in raw_size):
                raw_width, raw_height = map(int, raw_size)
            else:
                try:
                    with Image.open(image_path) as pil_image:  # type: ignore[arg-type]
                        raw_width, raw_height = pil_image.size
                except Exception:
                    raw_width, raw_height = 0, 0

            det_polygons = [np.asarray(polygon, dtype=np.float32) for polygon in prediction_entry.get("boxes", []) if polygon is not None]
            det_quads = [polygon.reshape(-1).tolist() for polygon in det_polygons if polygon.size > 0]

            # Filter and clip detection polygons to image bounds
            filtered_det_quads = []
            for quad in det_quads:
                if len(quad) < 8:  # Need at least 4 points (8 values)
                    continue
                coords = np.array(quad).reshape(-1, 2)
                x_coords = coords[:, 0]
                y_coords = coords[:, 1]
                # Skip polygons completely outside image bounds
                if x_coords.max() < 0 or x_coords.min() > raw_width or y_coords.max() < 0 or y_coords.min() > raw_height:
                    continue
                # Clip coordinates to image bounds
                clipped_coords = []
                for x, y in coords:
                    clipped_x = max(0, min(x, raw_width))
                    clipped_y = max(0, min(y, raw_height))
                    clipped_coords.extend([clipped_x, clipped_y])
                filtered_det_quads.append(clipped_coords)
            det_quads = filtered_det_quads

            canonical_gt = []
            if gt_words is not None and len(gt_words) > 0:
                if raw_width > 0 and raw_height > 0:
                    canonical_gt = remap_polygons(gt_words, raw_width, raw_height, orientation)
                else:
                    canonical_gt = [np.asarray(poly, dtype=np.float32) for poly in gt_words]

            gt_quads = [np.asarray(poly, dtype=np.float32).reshape(-1).tolist() for poly in canonical_gt if np.asarray(poly).size > 0]

            metric = CLEvalMetric(**self.metric_kwargs)  # Create new instance
            metric.reset()
            metric(det_quads, gt_quads)
            result = metric.compute()

            cleval_metrics["recall"].append(result["recall"].item())
            cleval_metrics["precision"].append(result["precision"].item())
            cleval_metrics["hmean"].append(result["f1"].item())

        recall = float(np.mean(cleval_metrics["recall"])) if cleval_metrics["recall"] else 0.0
        precision = float(np.mean(cleval_metrics["precision"])) if cleval_metrics["precision"] else 0.0
        hmean = float(np.mean(cleval_metrics["hmean"])) if cleval_metrics["hmean"] else 0.0

        return {"recall": recall, "precision": precision, "hmean": hmean}

    def on_validation_epoch_end(self):
        if self.valid_evaluator is None:
            return

        metrics = self.valid_evaluator.compute()
        for key, value in metrics.items():
            self.log(key, value, on_epoch=True, prog_bar=True)

        self._checkpoint_metrics = {
            "recall": metrics.get("val/recall", 0.0),
            "precision": metrics.get("val/precision", 0.0),
            "hmean": metrics.get("val/hmean", 0.0),
        }

        self.valid_evaluator.reset()

    def on_save_checkpoint(self, checkpoint):
        """Save additional metrics in the checkpoint."""
        return CheckpointHandler.on_save_checkpoint(self, checkpoint)

    def on_load_checkpoint(self, checkpoint):
        """Restore metrics from checkpoint (optional)."""
        CheckpointHandler.on_load_checkpoint(self, checkpoint)

    def test_step(self, batch):
        pred = self.model(return_loss=False, **batch)

        boxes_batch, _ = self.model.get_polygons_from_maps(batch, pred)
        predictions: list[dict[str, Any]] = []
        for idx, boxes in enumerate(boxes_batch):
            normalized_boxes = [np.asarray(box, dtype=np.float32).reshape(-1, 2) for box in boxes]
            predictions.append(
                {
                    "boxes": normalized_boxes,
                    "orientation": batch.get("orientation", [1])[idx] if "orientation" in batch else 1,
                    "raw_size": tuple(batch.get("raw_size", [(0, 0)])[idx]) if "raw_size" in batch else None,
                    "canonical_size": tuple(batch.get("canonical_size", [None])[idx]) if "canonical_size" in batch else None,
                    "image_path": batch.get("image_path", [None])[idx] if "image_path" in batch else None,
                }
            )

        if self.test_evaluator is not None:
            self.test_evaluator.update(batch["image_filename"], predictions)
        return pred

    def on_test_epoch_end(self):
        if self.test_evaluator is None:
            return

        metrics = self.test_evaluator.compute()
        for key, value in metrics.items():
            self.log(key, value, on_epoch=True, prog_bar=True)

        self.test_evaluator.reset()

    def predict_step(self, batch):
        pred = self.model(return_loss=False, **batch)
        boxes_batch, scores_batch = self.model.get_polygons_from_maps(batch, pred)

        include_confidence = getattr(self.config, "include_confidence", False)

        for idx, (boxes, scores) in enumerate(zip(boxes_batch, scores_batch, strict=True)):
            normalized_boxes = [np.asarray(box, dtype=np.float32).reshape(-1, 2) for box in boxes]
            if include_confidence:
                self.predict_step_outputs[batch["image_filename"][idx]] = {"boxes": normalized_boxes, "scores": scores}
            else:
                self.predict_step_outputs[batch["image_filename"][idx]] = normalized_boxes
        return pred

    def on_predict_epoch_end(self):
        self.submission_writer.save(self.predict_step_outputs)
        self.predict_step_outputs.clear()

    def configure_optimizers(self):
        optimizers, schedulers = self.model.get_optimizers()
        optimizer_list = optimizers if isinstance(optimizers, list) else [optimizers]

        if isinstance(schedulers, list):
            self.lr_scheduler = schedulers[0] if schedulers else None
        elif schedulers is None:
            self.lr_scheduler = None
        else:
            self.lr_scheduler = schedulers

        return optimizer_list

    def on_train_epoch_end(self):
        # Log cache statistics from datasets if caching is enabled
        if hasattr(self, "train_dataloader"):
            try:
                train_loader = self.trainer.train_dataloader
                if train_loader and hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "log_cache_statistics"):
                    train_loader.dataset.log_cache_statistics()
            except Exception:
                pass  # Silently skip if dataset doesn't support cache statistics

        if self.lr_scheduler is None:
            return

        if self.trainer is not None and self.trainer.sanity_checking:
            return

        optimizer = getattr(self.lr_scheduler, "optimizer", None)
        if optimizer is None:
            return
        step_count = getattr(optimizer, "_step_count", 0)
        if step_count > 0:
            self.lr_scheduler.step()


class OCRDataPLModule(pl.LightningDataModule):
    def __init__(self, dataset, config):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.dataloaders_cfg = self.config.dataloaders
        self.collate_cfg = self.config.collate_fn

    def _build_collate_fn(self, *, inference_mode: bool) -> Any:
        # Create collate function (no longer using cache - using pre-processed maps instead)
        collate_fn = instantiate(self.collate_cfg)
        if hasattr(collate_fn, "inference_mode"):
            collate_fn.inference_mode = inference_mode
        return collate_fn

    def train_dataloader(self):
        train_loader_config = self.dataloaders_cfg.train_dataloader
        # Filter out multiprocessing-only parameters when num_workers == 0
        if train_loader_config.get("num_workers", 0) == 0:
            train_loader_config = {k: v for k, v in train_loader_config.items() if k not in ["prefetch_factor", "persistent_workers"]}
        collate_fn = self._build_collate_fn(inference_mode=False)
        return DataLoader(self.dataset["train"], collate_fn=collate_fn, **train_loader_config)

    def val_dataloader(self):
        val_loader_config = self.dataloaders_cfg.val_dataloader
        # Filter out multiprocessing-only parameters when num_workers == 0
        if val_loader_config.get("num_workers", 0) == 0:
            val_loader_config = {k: v for k, v in val_loader_config.items() if k not in ["prefetch_factor", "persistent_workers"]}
        collate_fn = self._build_collate_fn(inference_mode=False)
        return DataLoader(self.dataset["val"], collate_fn=collate_fn, **val_loader_config)

    def test_dataloader(self):
        test_loader_config = self.dataloaders_cfg.test_dataloader
        # Filter out multiprocessing-only parameters when num_workers == 0
        if test_loader_config.get("num_workers", 0) == 0:
            test_loader_config = {k: v for k, v in test_loader_config.items() if k not in ["prefetch_factor", "persistent_workers"]}
        collate_fn = self._build_collate_fn(inference_mode=False)
        return DataLoader(self.dataset["test"], collate_fn=collate_fn, **test_loader_config)

    def predict_dataloader(self):
        predict_loader_config = self.dataloaders_cfg.predict_dataloader
        # Filter out multiprocessing-only parameters when num_workers == 0
        if predict_loader_config.get("num_workers", 0) == 0:
            predict_loader_config = {k: v for k, v in predict_loader_config.items() if k not in ["prefetch_factor", "persistent_workers"]}
        collate_fn = self._build_collate_fn(inference_mode=True)
        return DataLoader(self.dataset["predict"], collate_fn=collate_fn, **predict_loader_config)
