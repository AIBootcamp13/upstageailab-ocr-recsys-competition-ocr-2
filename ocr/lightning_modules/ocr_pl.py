import json
import multiprocessing as mp
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from ocr.metrics import CLEvalMetric


def evaluate_single_sample(det_quads, gt_quads, metric_kwargs=None):
    """Helper function for parallel evaluation of a single sample."""
    metric_kwargs = metric_kwargs or {}
    metric = CLEvalMetric(**metric_kwargs)
    metric(det_quads, gt_quads)
    result = metric.compute()
    return result["recall"].item(), result["precision"].item(), result["f1"].item()


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
        self.log("train/loss", pred["loss"], batch_size=len(batch))
        for key, value in pred["loss_dict"].items():
            self.log(f"train/{key}", value, batch_size=len(batch))
        return pred

    def validation_step(self, batch, batch_idx):
        pred = self.model(**batch)
        self.log("val/loss", pred["loss"], batch_size=len(batch))
        for key, value in pred["loss_dict"].items():
            self.log(f"val/{key}", value, batch_size=len(batch))

        boxes_batch, _ = self.model.get_polygons_from_maps(batch, pred)
        for idx, boxes in enumerate(boxes_batch):
            self.validation_step_outputs[batch["image_filename"][idx]] = boxes
        return pred

    def on_validation_epoch_end(self):
        cleval_metrics = defaultdict(list)

        # Prepare evaluation tasks
        eval_tasks = []
        for gt_filename, gt_words in self.dataset["val"].anns.items():
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
            eval_tasks.append((det_quads, gt_quads))

        # Parallel evaluation
        if eval_tasks:
            num_workers = min(mp.cpu_count(), len(eval_tasks))
            worker = partial(evaluate_single_sample, metric_kwargs=self.metric_kwargs)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(worker, det_quads, gt_quads) for det_quads, gt_quads in eval_tasks]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Parallel Evaluation",
                ):
                    recall, precision, hmean = future.result()
                    cleval_metrics["recall"].append(recall)
                    cleval_metrics["precision"].append(precision)
                    cleval_metrics["hmean"].append(hmean)

        recall = float(np.mean(cleval_metrics["recall"]))
        precision = float(np.mean(cleval_metrics["precision"]))
        hmean = float(np.mean(cleval_metrics["hmean"]))

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

        # Prepare evaluation tasks
        eval_tasks = []
        for gt_filename, gt_words in self.dataset["test"].anns.items():
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
            eval_tasks.append((det_quads, gt_quads))

        # Parallel evaluation
        if eval_tasks:
            num_workers = min(mp.cpu_count(), len(eval_tasks))
            worker = partial(evaluate_single_sample, metric_kwargs=self.metric_kwargs)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(worker, det_quads, gt_quads) for det_quads, gt_quads in eval_tasks]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Parallel Evaluation",
                ):
                    recall, precision, hmean = future.result()
                    cleval_metrics["recall"].append(recall)
                    cleval_metrics["precision"].append(precision)
                    cleval_metrics["hmean"].append(hmean)

        recall = float(np.mean(cleval_metrics["recall"]))
        precision = float(np.mean(cleval_metrics["precision"]))
        hmean = float(np.mean(cleval_metrics["hmean"]))

        self.log("test/recall", recall, on_epoch=True, prog_bar=True)
        self.log("test/precision", precision, on_epoch=True, prog_bar=True)
        self.log("test/hmean", hmean, on_epoch=True, prog_bar=True)

        self.test_step_outputs.clear()

    def predict_step(self, batch):
        pred = self.model(return_loss=False, **batch)
        boxes_batch, _ = self.model.get_polygons_from_maps(batch, pred)

        for idx, boxes in enumerate(boxes_batch):
            self.predict_step_outputs[batch["image_filename"][idx]] = boxes
        return pred

    def on_predict_epoch_end(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_file = Path(f"{self.config.paths.submission_dir}") / f"{timestamp}.json"
        submission_file.parent.mkdir(parents=True, exist_ok=True)

        submission = OrderedDict(images=OrderedDict())
        for filename, pred_boxes in self.predict_step_outputs.items():
            # Separate box
            boxes = OrderedDict()
            for idx, box in enumerate(pred_boxes):
                boxes[f"{idx + 1:04}"] = OrderedDict(points=box)

            # Append box
            submission["images"][filename] = OrderedDict(words=boxes)

        # Export submission
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
