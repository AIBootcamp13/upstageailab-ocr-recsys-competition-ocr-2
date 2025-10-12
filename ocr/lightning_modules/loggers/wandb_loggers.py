"""W&B logging utilities for the OCR Lightning module."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from ocr.lightning_modules.processors import ImageProcessor
from ocr.metrics import CLEvalMetric
from ocr.utils.orientation import remap_polygons

if TYPE_CHECKING:
    import wandb
else:
    try:
        import wandb
    except ImportError:
        wandb = None


class WandbProblemLogger:
    """Handles logging of problematic validation images to Weights & Biases.

    This class encapsulates the complex logic for determining when to log images
    based on recall thresholds and managing the W&B image upload process.
    """

    def __init__(
        self,
        config: Any,
        normalize_mean: Any = None,
        normalize_std: Any = None,
        val_dataset: Any | None = None,
        metric_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the logger with configuration.

        Args:
            config: The full configuration object
            normalize_mean: Mean values for image denormalization
            normalize_std: Standard deviation values for image denormalization
            val_dataset: Validation dataset reference for metric computation
            metric_kwargs: Keyword arguments used when instantiating CLEvalMetric
        """
        self.config = config
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self._logged_batches = 0
        self.val_dataset = val_dataset
        self.metric_kwargs = metric_kwargs or {}

    def reset_epoch_counter(self) -> None:
        """Reset the counter for logged batches at the start of each epoch."""
        self._logged_batches = 0

    def log_if_needed(self, batch: dict[str, Any], predictions: list[dict[str, Any]], batch_idx: int) -> dict[str, float]:
        """Compute metrics and log problematic images to W&B when appropriate.

        Args:
            batch: The batch data from the dataloader
            predictions: List of prediction dictionaries
            batch_idx: Index of the batch in the epoch

        Returns:
            The metric summary for the batch.
        """
        batch_metrics = self._compute_batch_metrics(batch, predictions)

        per_batch_cfg = getattr(self.config.logger, "per_batch_image_logging", None)
        if per_batch_cfg is None or not per_batch_cfg.enabled:
            return batch_metrics

        # Check if we should log this batch
        recall_threshold = getattr(per_batch_cfg, "recall_threshold", 1.0)
        if batch_metrics["recall"] >= recall_threshold:
            return batch_metrics

        max_batches_per_epoch = getattr(per_batch_cfg, "max_batches_per_epoch", None)
        if max_batches_per_epoch is not None and max_batches_per_epoch > 0 and self._logged_batches >= max_batches_per_epoch:
            return batch_metrics

        # Log the problematic images
        self._log_problematic_batch(batch, predictions, batch_metrics, batch_idx, per_batch_cfg)
        self._logged_batches += 1
        return batch_metrics

    def _log_problematic_batch(
        self, batch: dict[str, Any], predictions: list[dict[str, Any]], batch_metrics: dict[str, float], batch_idx: int, per_batch_cfg: Any
    ) -> None:
        """Log a batch of problematic images to W&B."""
        if wandb is None:
            return  # wandb not available

        max_images = getattr(per_batch_cfg, "max_images_per_batch", len(batch["image_path"]))
        if max_images <= 0:
            max_images = len(batch["image_path"])

        use_transformed_batch = bool(getattr(per_batch_cfg, "use_transformed_batch", False))
        image_format = str(getattr(per_batch_cfg, "image_format", "")).lower()
        max_image_side = getattr(per_batch_cfg, "max_image_side", None)

        batch_images_tensor = batch.get("images") if use_transformed_batch else None
        if batch_images_tensor is not None:
            batch_images_tensor = batch_images_tensor.detach().cpu()

        wandb_images: list[Any] = []
        pil_images_to_close: list[Image.Image] = []

        for local_idx, path in enumerate(batch["image_path"]):
            if len(wandb_images) >= max_images:
                break

            pil_image = self._prepare_image_for_logging(path, batch_images_tensor, local_idx, use_transformed_batch)
            if pil_image is None:
                continue

            processed_image = ImageProcessor.prepare_wandb_image(pil_image, max_image_side)

            filename = path.name if hasattr(path, "name") else str(path).split("/")[-1]
            caption = f"Problematic batch {batch_idx} - {filename} (recall: {batch_metrics['recall']:.3f})"

            wandb_kwargs: dict[str, Any] = {}
            if image_format in {"jpeg", "jpg"}:
                wandb_kwargs["file_type"] = "jpg"
            elif image_format == "png":
                wandb_kwargs["file_type"] = "png"

            wandb_images.append(wandb.Image(processed_image, caption=caption, **wandb_kwargs))  # type: ignore

            if processed_image is not pil_image:
                pil_images_to_close.append(processed_image)
            pil_images_to_close.append(pil_image)

        if wandb_images:
            wandb.log(  # type: ignore
                {
                    f"problematic_batch_{batch_idx}_images": wandb_images,
                    f"problematic_batch_{batch_idx}_count": len(wandb_images),
                    f"problematic_batch_{batch_idx}_recall": batch_metrics["recall"],
                    f"problematic_batch_{batch_idx}_precision": batch_metrics["precision"],
                    f"problematic_batch_{batch_idx}_hmean": batch_metrics["hmean"],
                }
            )

        # Clean up PIL images
        for img in pil_images_to_close:
            try:
                img.close()
            except Exception:
                pass

    def _compute_batch_metrics(self, batch: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, float]:
        """Compute CLEval metrics for a batch using dataset annotations."""
        if not predictions:
            return {"recall": 0.0, "precision": 0.0, "hmean": 0.0}

        val_dataset = self.val_dataset
        if val_dataset is None:
            return {"recall": 0.0, "precision": 0.0, "hmean": 0.0}
        if hasattr(val_dataset, "dataset"):
            val_dataset = val_dataset.dataset
        anns = getattr(val_dataset, "anns", None)
        if not anns:
            return {"recall": 0.0, "precision": 0.0, "hmean": 0.0}

        cleval_metrics = defaultdict(list)
        dataset_image_root = getattr(val_dataset, "image_path", getattr(self.val_dataset, "image_path", None))

        for idx, prediction_entry in enumerate(predictions):
            filename = batch["image_filename"][idx]
            gt_words = anns.get(filename)
            if gt_words is None:
                continue

            orientation = prediction_entry.get("orientation", 1)
            if "orientation" in batch:
                orientation = prediction_entry.get("orientation", batch["orientation"][idx])

            raw_size = prediction_entry.get("raw_size")
            if raw_size is None and "raw_size" in batch:
                raw_size = batch["raw_size"][idx]

            image_path = prediction_entry.get("image_path")
            if image_path is None and "image_path" in batch:
                image_path = batch["image_path"][idx]
            if image_path is None and dataset_image_root is not None:
                image_path = Path(dataset_image_root) / filename

            raw_width, raw_height = 0, 0
            if raw_size is not None and all(dim for dim in raw_size):
                raw_width, raw_height = map(int, raw_size)
            elif image_path is not None:
                try:
                    with Image.open(image_path) as pil_image:
                        raw_width, raw_height = pil_image.size
                except Exception:
                    raw_width, raw_height = 0, 0

            det_polygons = [np.asarray(polygon, dtype=np.float32) for polygon in prediction_entry.get("boxes", []) if polygon is not None]
            det_quads = [polygon.reshape(-1).tolist() for polygon in det_polygons if polygon.size > 0]

            filtered_det_quads = []
            for quad in det_quads:
                if len(quad) < 8:
                    continue
                coords = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
                x_coords = coords[:, 0]
                y_coords = coords[:, 1]
                if raw_width and raw_height:
                    if x_coords.max() < 0 or x_coords.min() > raw_width or y_coords.max() < 0 or y_coords.min() > raw_height:
                        continue
                clipped_coords: list[float] = []
                for x, y in coords:
                    clipped_x = max(0.0, min(float(x), float(raw_width))) if raw_width else float(x)
                    clipped_y = max(0.0, min(float(y), float(raw_height))) if raw_height else float(y)
                    clipped_coords.extend([clipped_x, clipped_y])
                filtered_det_quads.append(clipped_coords)
            det_quads = filtered_det_quads

            canonical_gt = []
            if gt_words:
                if raw_width > 0 and raw_height > 0:
                    canonical_gt = remap_polygons(gt_words, raw_width, raw_height, orientation)
                else:
                    canonical_gt = [np.asarray(poly, dtype=np.float32) for poly in gt_words]

            gt_quads = [np.asarray(poly, dtype=np.float32).reshape(-1).tolist() for poly in canonical_gt if np.asarray(poly).size > 0]

            metric = CLEvalMetric(**self.metric_kwargs)
            metric.reset()
            metric(det_quads, gt_quads)
            result = metric.compute()

            cleval_metrics["recall"].append(result["recall"].item())
            cleval_metrics["precision"].append(result["precision"].item())
            cleval_metrics["hmean"].append(result["f1"].item())

        if not cleval_metrics["recall"]:
            return {"recall": 0.0, "precision": 0.0, "hmean": 0.0}

        recall = float(np.mean(cleval_metrics["recall"])) if cleval_metrics["recall"] else 0.0
        precision = float(np.mean(cleval_metrics["precision"])) if cleval_metrics["precision"] else 0.0
        hmean = float(np.mean(cleval_metrics["hmean"])) if cleval_metrics["hmean"] else 0.0

        return {"recall": recall, "precision": precision, "hmean": hmean}

    def _prepare_image_for_logging(
        self, path: str, batch_images_tensor: Any, local_idx: int, use_transformed_batch: bool
    ) -> Image.Image | None:
        """Prepare a single image for W&B logging."""
        pil_image = None

        if use_transformed_batch and batch_images_tensor is not None:
            try:
                pil_image = ImageProcessor.tensor_to_pil_image(
                    batch_images_tensor[local_idx],
                    mean=self.normalize_mean,
                    std=self.normalize_std,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: Failed to convert transformed image for wandb logging: {exc}")

        if pil_image is None:
            try:
                pil_image = Image.open(path)
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
            except Exception as e:  # noqa: BLE001
                print(f"Warning: Failed to load image {path} for wandb logging: {e}")
                return None

        return pil_image
