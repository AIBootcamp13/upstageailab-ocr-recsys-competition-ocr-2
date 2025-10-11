"""Dedicated evaluator used by the OCR Lightning module."""

from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from ocr.metrics import CLEvalMetric
from ocr.utils.orientation import remap_polygons
from ocr.validation.models import LightningStepPrediction, validate_predictions

try:  # Rich is optional – fall back to tqdm when unavailable
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - development environments without rich
    RICH_AVAILABLE = False


class CLEvalEvaluator:
    """Accumulates predictions and computes CLEval metrics for a dataset split."""

    def __init__(
        self,
        dataset: Any,
        metric_cfg: dict[str, Any] | None = None,
        mode: str = "val",
        enable_validation: bool = True,
    ) -> None:
        self.dataset = dataset
        self.mode = mode
        self.metric_cfg = metric_cfg or {}
        self.metric = CLEvalMetric(**self.metric_cfg)
        self.enable_validation = enable_validation
        self.predictions: OrderedDict[str, LightningStepPrediction] = OrderedDict()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def update(self, filenames: Sequence[str], predictions: Sequence[dict[str, Any]]) -> None:
        """Store the predictions generated for a single batch.

        Args:
            filenames: Image filenames associated with the predictions.
            predictions: Raw prediction dictionaries produced by the model.
        """

        validated = (
            validate_predictions(filenames, predictions)
            if self.enable_validation
            else [LightningStepPrediction(**pred) for pred in predictions]
        )
        for name, prediction in zip(filenames, validated, strict=True):
            self.predictions[name] = prediction

    def compute(self) -> dict[str, float]:
        """Compute CLEval metrics for all accumulated predictions."""

        dataset, filenames_to_check = self._resolve_dataset_and_filenames()
        if hasattr(dataset, "log_cache_statistics"):
            dataset.log_cache_statistics()

        processed_filenames = [name for name in filenames_to_check if name in self.predictions]
        if not processed_filenames:
            logging.warning("No %s predictions found. This may indicate a data loading or prediction issue.", self.mode)
            return {
                f"{self.mode}/recall": 0.0,
                f"{self.mode}/precision": 0.0,
                f"{self.mode}/hmean": 0.0,
            }

        metrics = defaultdict(list)
        iterator: Iterable[str]
        if RICH_AVAILABLE:
            iterator = self._rich_iterator(processed_filenames, description="Evaluation")
        else:
            iterator = tqdm(
                processed_filenames,
                desc="Evaluation",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                colour="red",
            )

        for filename in iterator:
            gt_words = self._get_annotations(dataset, filename)
            prediction = self.predictions[filename]
            det_quads = [polygon.reshape(-1).tolist() for polygon in prediction.boxes if polygon.size > 0]

            raw_width, raw_height = self._resolve_raw_size(prediction, dataset, filename)
            canonical_gt = self._remap_ground_truth(gt_words, raw_width, raw_height, prediction.orientation)
            gt_quads = [poly.reshape(-1).tolist() for poly in canonical_gt if poly.size > 0]

            self.metric.reset()
            self.metric(det_quads, gt_quads)
            result = self.metric.compute()

            metrics["recall"].append(result["recall"].item())
            metrics["precision"].append(result["precision"].item())
            metrics["hmean"].append(result["f1"].item())

        recall = float(np.mean(metrics["recall"])) if metrics["recall"] else 0.0
        precision = float(np.mean(metrics["precision"])) if metrics["precision"] else 0.0
        hmean = float(np.mean(metrics["hmean"])) if metrics["hmean"] else 0.0

        return {
            f"{self.mode}/recall": recall,
            f"{self.mode}/precision": precision,
            f"{self.mode}/hmean": hmean,
        }

    def reset(self) -> None:
        """Clear all accumulated predictions."""

        self.predictions.clear()

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _resolve_dataset_and_filenames(self) -> tuple[Any, list[str]]:
        dataset = self.dataset
        if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
            base = dataset.dataset
            annotations = list(getattr(base, "anns", {}).keys())
            filenames = [annotations[idx] for idx in dataset.indices]
            return base, filenames
        annotations = list(getattr(dataset, "anns", {}).keys())
        return dataset, annotations

    def _get_annotations(self, dataset: Any, filename: str) -> Any:
        if hasattr(dataset, "anns"):
            return dataset.anns.get(filename)
        if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "anns"):
            return dataset.dataset.anns.get(filename)
        return None

    def _resolve_raw_size(self, prediction: LightningStepPrediction, dataset: Any, filename: str) -> tuple[int, int]:
        if prediction.raw_size is not None:
            return prediction.raw_size
        image_path = prediction.image_path
        if image_path is None:
            base_path = getattr(dataset, "image_path", None)
            if base_path is not None:
                image_path = str(Path(base_path) / filename)
        try:
            with Image.open(image_path) as image:  # type: ignore[arg-type]
                return image.size
        except Exception:  # noqa: BLE001
            return 0, 0

    @staticmethod
    def _remap_ground_truth(gt_words: Any, raw_width: int, raw_height: int, orientation: int) -> list[np.ndarray]:
        if gt_words is None or len(gt_words) == 0:
            return []
        if raw_width > 0 and raw_height > 0:
            return remap_polygons(gt_words, raw_width, raw_height, orientation)
        return [np.asarray(poly, dtype=np.float32) for poly in gt_words]

    @staticmethod
    def _rich_iterator(filenames: Sequence[str], description: str) -> Iterable[str]:
        console = None
        try:
            from rich.console import Console

            console = Console()
        except ImportError:  # pragma: no cover - executed when rich console unavailable
            pass

        if console is None:
            return filenames

        progress = Progress(
            TextColumn("[bold red]{task.description}"),
            BarColumn(bar_width=50, style="red"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[progress.completed]{task.completed}/{task.total}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=2,
        )
        task_id = progress.add_task(description, total=len(filenames))

        def iterator() -> Iterable[str]:
            with progress:
                for name in filenames:
                    yield name
                    progress.advance(task_id)

        return iterator()
