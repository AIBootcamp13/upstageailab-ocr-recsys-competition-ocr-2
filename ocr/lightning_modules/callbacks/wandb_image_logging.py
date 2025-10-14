from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
from PIL import Image

from ocr.utils.orientation import normalize_pil_image, remap_polygons
from ocr.utils.polygon_utils import ensure_polygon_array
from ocr.utils.wandb_utils import log_validation_images


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

        # Handle Subset datasets - get the underlying dataset
        if hasattr(val_dataset, "dataset"):
            val_dataset = val_dataset.dataset  # type: ignore

        # Prepare data for logging
        images = []
        gt_bboxes = []
        pred_bboxes = []
        filenames = []

        # Collect up to max_images samples
        count = 0
        for filename, pred_data in list(pl_module.validation_step_outputs.items())[: self.max_images]:  # type: ignore
            entry = pred_data if isinstance(pred_data, dict) else {"boxes": pred_data}
            pred_boxes = entry.get("boxes", [])
            orientation_hint = entry.get("orientation", 1)
            raw_size_hint = entry.get("raw_size")
            metadata = self._normalize_metadata(entry.get("metadata"))

            # Get ground truth boxes
            gt_boxes = val_dataset.anns[filename]  # type: ignore
            gt_quads = self._normalise_polygons(gt_boxes)
            pred_quads = self._normalise_polygons(pred_boxes)

            # Check if we have the transformed image stored
            transformed_image = entry.get("transformed_image")
            if transformed_image is not None:
                # Use the exact transformed image from training
                try:
                    # Convert tensor back to PIL Image
                    pil_image = self._tensor_to_pil(transformed_image)
                    image = pil_image
                    raw_width, raw_height = pil_image.size
                    normalized_image = pil_image  # Already transformed
                except Exception as e:
                    print(f"Warning: Failed to convert transformed image for {filename}: {e}")
                    continue
            else:
                # Fallback to original method (load from disk and transform)
                if metadata:
                    if "orientation" in metadata and metadata["orientation"] is not None:
                        orientation_hint = int(metadata["orientation"])
                    if "raw_size" in metadata and metadata["raw_size"] is not None:
                        raw_size_hint = metadata["raw_size"]
                if not hasattr(val_dataset, "anns") or filename not in val_dataset.anns:  # type: ignore
                    continue

                # Get image directly from filesystem (similar to dataset loading)
                try:
                    image_path = self._resolve_image_path(entry, metadata, val_dataset, filename)
                    pil_image = Image.open(image_path)
                    raw_width, raw_height = pil_image.size
                    normalized_image, orientation = normalize_pil_image(pil_image)

                    if normalized_image.mode != "RGB":
                        image = normalized_image.convert("RGB")
                    else:
                        image = normalized_image.copy()

                    if normalized_image is not image:
                        normalized_image.close()
                    pil_image.close()
                except Exception as e:
                    # If we can't get the image, skip this sample
                    print(f"Warning: Failed to load image {filename}: {e}")
                    continue

                polygon_frame = metadata.get("polygon_frame") if metadata else None
                if gt_quads:
                    if polygon_frame == "canonical":
                        pass
                    elif orientation != 1:
                        gt_quads = remap_polygons(gt_quads, raw_width, raw_height, orientation)
                    elif orientation_hint != 1:
                        hint_width, hint_height = raw_size_hint or (raw_width, raw_height)
                        gt_quads = remap_polygons(gt_quads, hint_width, hint_height, orientation_hint)

                gt_quads = self._postprocess_polygons(gt_quads, image.size)
                pred_quads = self._postprocess_polygons(pred_quads, image.size)

                images.append(image)
                gt_bboxes.append(gt_quads)
                pred_bboxes.append(pred_quads)
                filenames.append((filename, image.size[0], image.size[1]))  # (filename, width, height)
                count += 1

                if count >= self.max_images:
                    break

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
                polygon_array = ensure_polygon_array(np.asarray(polygon))  # type: ignore[arg-type]
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

        def _is_degenerate_polygon(polygon: np.ndarray) -> bool:
            """Check if a polygon is degenerate (has duplicate consecutive points or all points same)."""
            if polygon.size == 0:
                return True

            # Flatten to (N, 2) shape
            poly_2d = polygon.reshape(-1, 2)

            # Check if all points are the same
            if np.allclose(poly_2d[0], poly_2d):
                return True

            # Check for duplicate consecutive points
            for i in range(len(poly_2d)):
                if np.allclose(poly_2d[i], poly_2d[(i + 1) % len(poly_2d)]):
                    return True

            return False

        processed = [np.array(polygon, copy=True) for polygon in polygons if not _is_degenerate_polygon(polygon)]

        width, height = image_size
        return processed

    @staticmethod
    def _normalize_metadata(metadata: Any) -> dict[str, Any] | None:
        if metadata is None:
            return None
        if hasattr(metadata, "model_dump"):
            metadata = metadata.model_dump()
        elif not isinstance(metadata, dict):
            try:
                metadata = dict(metadata)
            except Exception:  # noqa: BLE001
                return None

        normalized: dict[str, Any] = {}
        for key, value in metadata.items():
            if key == "path" and value is not None:
                normalized[key] = Path(value)
            elif key in {"raw_size", "canonical_size"} and value is not None:
                normalized[key] = WandbImageLoggingCallback._ensure_size_tuple(value)
            elif key == "orientation" and value is not None:
                try:
                    normalized[key] = int(value)
                except (TypeError, ValueError):  # noqa: BLE001
                    continue
            else:
                normalized[key] = value

        return normalized

    @staticmethod
    def _ensure_size_tuple(value: Any) -> tuple[int, int] | None:
        if value is None:
            return None
        if isinstance(value, tuple) and len(value) == 2:
            return int(value[0]), int(value[1])
        if isinstance(value, list) and len(value) == 2:
            return int(value[0]), int(value[1])
        try:
            width, height = value  # type: ignore[misc]
            return int(width), int(height)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _tensor_to_pil(tensor):
        """Convert a normalized tensor (C, H, W) back to PIL Image."""
        import numpy as np
        import torch
        from PIL import Image

        # Convert to numpy and transpose to HWC
        if torch.is_tensor(tensor):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.array(tensor)

        # Handle different tensor shapes
        if arr.shape[0] == 3:  # CHW format
            arr = np.transpose(arr, (1, 2, 0))  # HWC
        elif arr.shape[-1] == 3:  # HWC format already
            pass
        else:  # Grayscale or other
            arr = arr.squeeze()

        # Un-normalize from [-1, 1] or [0, 1] to [0, 255]
        if arr.max() <= 1.0:
            if arr.min() < 0:  # Likely in [-1, 1] range
                arr = (arr + 1) * 127.5
            else:  # Likely in [0, 1] range
                arr = arr * 255.0

        arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Convert to PIL Image
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return Image.fromarray(arr, mode="RGB")
        elif arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        else:
            raise ValueError(f"Unsupported tensor shape: {arr.shape}")

    @staticmethod
    def _resolve_image_path(entry: dict[str, Any], metadata: dict[str, Any] | None, dataset: Any, filename: str) -> Path:
        candidates: list[Any] = [entry.get("image_path")]
        if metadata is not None:
            candidates.append(metadata.get("path"))

        for candidate in candidates:
            if candidate is None:
                continue
            candidate_path = Path(candidate)
            if candidate_path.is_absolute():
                return candidate_path
            if hasattr(dataset, "image_path"):
                base_path = dataset.image_path  # type: ignore[attr-defined]
                return Path(base_path) / candidate_path

        # Fallback to dataset root
        if hasattr(dataset, "image_path"):
            return Path(dataset.image_path) / filename  # type: ignore[attr-defined]

        return Path(filename)
