import json
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from ocr.metrics import CLEvalMetric


class OCRPLModule(pl.LightningModule):
    def __init__(self, model, dataset, config, metric_cfg: DictConfig | None = None):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.metric_cfg = metric_cfg
        self.metric_kwargs = self._extract_metric_kwargs(metric_cfg)
        self.metric = instantiate(metric_cfg) if metric_cfg is not None else CLEvalMetric(**self.metric_kwargs)
        self.config = config
        self.lr_scheduler = None

        self.validation_step_outputs: OrderedDict[str, Any] = OrderedDict()
        self.test_step_outputs: OrderedDict[str, Any] = OrderedDict()
        self.predict_step_outputs: OrderedDict[str, Any] = OrderedDict()

    @staticmethod
    def _extract_metric_kwargs(metric_cfg: DictConfig | None) -> dict:
        if metric_cfg is None:
            return {}

        cfg_dict = OmegaConf.to_container(metric_cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            return {}

        cfg_dict.pop("_target_", None)
        return cfg_dict

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
        for idx, boxes in enumerate(boxes_batch):
            self.validation_step_outputs[batch["image_filename"][idx]] = boxes

        # Compute per-batch validation metrics
        batch_metrics = self._compute_batch_metrics(batch, boxes_batch)
        self.log(f"batch_{batch_idx}/recall", batch_metrics["recall"], batch_size=batch["images"].shape[0])
        self.log(f"batch_{batch_idx}/precision", batch_metrics["precision"], batch_size=batch["images"].shape[0])
        self.log(f"batch_{batch_idx}/hmean", batch_metrics["hmean"], batch_size=batch["images"].shape[0])

        # Log problematic batch image paths
        if batch_metrics["recall"] < 0.8:
            image_paths = [str(path) for path in batch["image_path"]]
            # Use wandb.log if available for image paths
            try:
                import wandb

                wandb.log({f"problematic_batch_{batch_idx}": image_paths})
            except ImportError:
                pass  # wandb not available

        return pred["loss"]

    def _compute_batch_metrics(self, batch, boxes_batch):
        """Compute validation metrics for a batch of images."""
        cleval_metrics = defaultdict(list)

        for idx, boxes in enumerate(boxes_batch):
            filename = batch["image_filename"][idx]
            if filename not in self.dataset["val"].anns:
                continue
            gt_words = self.dataset["val"].anns[filename]

            det_quads = [[point for coord in polygons for point in coord] for polygons in boxes]
            gt_quads = [item.squeeze().reshape(-1) for item in gt_words]

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
        cleval_metrics = defaultdict(list)

        for gt_filename, gt_words in tqdm(self.dataset["val"].anns.items(), desc="Evaluation"):
            if gt_filename not in self.validation_step_outputs:
                import logging

                logging.warning(
                    f"Missing predictions for ground truth file '{gt_filename}' during validation epoch end. "
                    "This may indicate a data loading or prediction issue."
                )
                cleval_metrics["recall"].append(0.0)
                cleval_metrics["precision"].append(0.0)
                cleval_metrics["hmean"].append(0.0)
                continue

            pred = self.validation_step_outputs[gt_filename]
            det_quads = [[point for coord in polygons for point in coord] for polygons in pred]
            gt_quads = [item.squeeze().reshape(-1) for item in gt_words]

            metric = self.metric
            metric.reset()
            metric(det_quads, gt_quads)
            result = metric.compute()

            cleval_metrics["recall"].append(result["recall"].item())
            cleval_metrics["precision"].append(result["precision"].item())
            cleval_metrics["hmean"].append(result["f1"].item())
            metric.reset()

        recall = float(np.mean(cleval_metrics["recall"])) if cleval_metrics["recall"] else 0.0
        precision = float(np.mean(cleval_metrics["precision"])) if cleval_metrics["precision"] else 0.0
        hmean = float(np.mean(cleval_metrics["hmean"])) if cleval_metrics["hmean"] else 0.0

        self.log("val/recall", recall, on_epoch=True, prog_bar=True)
        self.log("val/precision", precision, on_epoch=True, prog_bar=True)
        self.log("val/hmean", hmean, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch):
        pred = self.model(return_loss=False, **batch)

        boxes_batch, _ = self.model.get_polygons_from_maps(batch, pred)
        for idx, boxes in enumerate(boxes_batch):
            self.test_step_outputs[batch["image_filename"][idx]] = boxes
        return pred

    def on_test_epoch_end(self):
        cleval_metrics = defaultdict(list)

        for gt_filename, gt_words in tqdm(self.dataset["test"].anns.items(), desc="Evaluation"):
            if gt_filename not in self.test_step_outputs:
                import logging

                logging.warning(
                    f"Missing predictions for ground truth file '{gt_filename}' during test epoch end. "
                    "This may indicate a data loading or prediction issue."
                )
                cleval_metrics["recall"].append(0.0)
                cleval_metrics["precision"].append(0.0)
                cleval_metrics["hmean"].append(0.0)
                continue

            pred = self.test_step_outputs[gt_filename]
            det_quads = [[point for coord in polygons for point in coord] for polygons in pred]
            gt_quads = [item.squeeze().reshape(-1) for item in gt_words]

            metric = self.metric
            metric.reset()
            metric(det_quads, gt_quads)
            result = metric.compute()

            cleval_metrics["recall"].append(result["recall"].item())
            cleval_metrics["precision"].append(result["precision"].item())
            cleval_metrics["hmean"].append(result["f1"].item())
            metric.reset()

        recall = float(np.mean(cleval_metrics["recall"])) if cleval_metrics["recall"] else 0.0
        precision = float(np.mean(cleval_metrics["precision"])) if cleval_metrics["precision"] else 0.0
        hmean = float(np.mean(cleval_metrics["hmean"])) if cleval_metrics["hmean"] else 0.0

        self.log("test/recall", recall, on_epoch=True, prog_bar=True)
        self.log("test/precision", precision, on_epoch=True, prog_bar=True)
        self.log("test/hmean", hmean, on_epoch=True, prog_bar=True)

        self.test_step_outputs.clear()

    def predict_step(self, batch):
        pred = self.model(return_loss=False, **batch)
        boxes_batch, scores_batch = self.model.get_polygons_from_maps(batch, pred)

        include_confidence = getattr(self.config, "include_confidence", False)

        for idx, (boxes, scores) in enumerate(zip(boxes_batch, scores_batch, strict=True)):
            if include_confidence:
                self.predict_step_outputs[batch["image_filename"][idx]] = {"boxes": boxes, "scores": scores}
            else:
                self.predict_step_outputs[batch["image_filename"][idx]] = boxes
        return pred

    def on_predict_epoch_end(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_file = Path(f"{self.config.paths.submission_dir}") / f"{timestamp}.json"
        submission_file.parent.mkdir(parents=True, exist_ok=True)

        submission = OrderedDict(images=OrderedDict())
        include_confidence = getattr(self.config, "include_confidence", False)

        for filename, pred_data in self.predict_step_outputs.items():
            if include_confidence:
                boxes = pred_data["boxes"]
                scores = pred_data["scores"]
            else:
                boxes = pred_data
                scores = None

            # Separate box
            words = OrderedDict()
            for idx, box in enumerate(boxes):
                word_data = OrderedDict(points=box)
                if include_confidence and scores is not None:
                    word_data["confidence"] = float(scores[idx])
                words[f"{idx + 1:04}"] = word_data

            # Append box
            submission["images"][filename] = OrderedDict(words=words)  # Export submission
        with submission_file.open("w") as fp:
            if self.config.minified_json:
                json.dump(submission, fp, indent=None, separators=(",", ":"))
            else:
                json.dump(submission, fp, indent=4)

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
        self.collate_fn = instantiate(self.collate_cfg)

    def train_dataloader(self):
        train_loader_config = self.dataloaders_cfg.train_dataloader
        self.collate_fn.inference_mode = False
        return DataLoader(self.dataset["train"], collate_fn=self.collate_fn, **train_loader_config)

    def val_dataloader(self):
        val_loader_config = self.dataloaders_cfg.val_dataloader
        self.collate_fn.inference_mode = False
        return DataLoader(self.dataset["val"], collate_fn=self.collate_fn, **val_loader_config)

    def test_dataloader(self):
        test_loader_config = self.dataloaders_cfg.test_dataloader
        self.collate_fn.inference_mode = False
        return DataLoader(self.dataset["test"], collate_fn=self.collate_fn, **test_loader_config)

    def predict_dataloader(self):
        predict_loader_config = self.dataloaders_cfg.predict_dataloader
        self.collate_fn.inference_mode = True
        return DataLoader(self.dataset["predict"], collate_fn=self.collate_fn, **predict_loader_config)
