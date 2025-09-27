from __future__ import annotations

from collections.abc import Iterable, Sequence

import lightning.pytorch as pl
import numpy as np
from PIL import Image

from ocr.datasets.base import OCRDataset
from ocr.utils.wandb_utils import log_validation_images

EXIF_ORIENTATION = 274  # Orientation Information: 274


class WandbImageLoggingCallback(pl.Callback):
    """Callback to log validation images with bounding boxes to Weights & Biases."""

    def __init__(self, log_every_n_epochs: int = 1, max_images: int = 8):
        self.log_every_n_epochs = log_every_n_epochs
        self.max_images = max_images

    def on_validation_epoch_end(self, trainer, pl_module):
        # Only log every N epochs to avoid too much data
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Collect validation images, ground truth, and predictions for logging
        if not hasattr(pl_module, "validation_step_outputs") or not pl_module.validation_step_outputs:
            return

        # Get the validation dataset for ground truth
        if not hasattr(pl_module, "dataset") or not isinstance(pl_module.dataset, dict) or "val" not in pl_module.dataset:
            return

        val_dataset = pl_module.dataset["val"]  # type: ignore

        # Prepare data for logging
        images = []
        gt_bboxes = []
        pred_bboxes = []
        filenames = []

        # Collect up to max_images samples
        count = 0
        for filename, pred_boxes in list(pl_module.validation_step_outputs.items())[: self.max_images]:  # type: ignore
            if not hasattr(val_dataset, "anns") or filename not in val_dataset.anns:  # type: ignore
                continue

            # Get ground truth boxes
            gt_boxes = val_dataset.anns[filename]  # type: ignore
            gt_quads = self._normalise_polygons(gt_boxes)
            pred_quads = self._normalise_polygons(pred_boxes)

            # Get image directly from filesystem (similar to dataset loading)
            try:
                image_path = val_dataset.image_path / filename  # type: ignore
                image = Image.open(image_path).convert("RGB")

                # Apply EXIF rotation to match dataset preprocessing
                exif = image.getexif()
                original_size = image.size
                if exif and EXIF_ORIENTATION in exif:
                    orientation = exif[EXIF_ORIENTATION]
                    image = OCRDataset.rotate_image(image, orientation)
                    # Transform polygons to match rotated image
                    gt_quads = OCRDataset.transform_polygons_for_exif(gt_quads, orientation, original_size)
                    pred_quads = OCRDataset.transform_polygons_for_exif(pred_quads, orientation, original_size)

                gt_quads = self._postprocess_polygons(gt_quads, image.size)
                pred_quads = self._postprocess_polygons(pred_quads, image.size)

                images.append(image)
                gt_bboxes.append(gt_quads)
                pred_bboxes.append(pred_quads)
                filenames.append((filename, image.size[0], image.size[1]))  # (filename, width, height)
                count += 1

                if count >= self.max_images:
                    break
            except Exception as e:
                # If we can't get the image, skip this sample
                print(f"Warning: Failed to load image {filename}: {e}")
                continue

        # Log images with bounding boxes if we have data
        if images and gt_bboxes and pred_bboxes:
            try:
                log_validation_images(
                    images=images,
                    gt_bboxes=gt_bboxes,
                    pred_bboxes=pred_bboxes,
                    epoch=trainer.current_epoch,
                    limit=self.max_images,
                    filenames=filenames,
                )
            except Exception as e:
                # Log error but don't crash training
                print(f"Warning: Failed to log validation images to WandB: {e}")

    @staticmethod
    def _normalise_polygons(polygons: Sequence | Iterable | None) -> list[np.ndarray]:
        if not polygons:
            return []

        normalised: list[np.ndarray] = []
        for polygon in polygons:  # type: ignore[arg-type]
            try:
                polygon_array = OCRDataset._ensure_polygon_array(polygon)  # type: ignore[arg-type]
            except ValueError as exc:
                print(f"Warning: Skipping polygon due to shape error: {exc}")
                continue

            if polygon_array is None or polygon_array.size == 0:
                continue

            normalised.append(np.array(polygon_array, copy=True))

        return normalised

    @staticmethod
    def _postprocess_polygons(polygons: Sequence[np.ndarray] | None, image_size: tuple[int, int]) -> list[np.ndarray]:
        if not polygons:
            return []

        processed = [np.array(polygon, copy=True) for polygon in polygons if polygon.size > 0]
        if not processed:
            return []

        width, height = image_size
        OCRDataset._clip_polygons_in_place(processed, width, height)
        return OCRDataset._filter_degenerate_polygons(processed)
