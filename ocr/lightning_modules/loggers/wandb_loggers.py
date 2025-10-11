"""W&B logging utilities for the OCR Lightning module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PIL import Image

from ocr.lightning_modules.processors import ImageProcessor

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

    def __init__(self, config: Any, normalize_mean: Any = None, normalize_std: Any = None):
        """Initialize the logger with configuration.

        Args:
            config: The full configuration object
            normalize_mean: Mean values for image denormalization
            normalize_std: Standard deviation values for image denormalization
        """
        self.config = config
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self._logged_batches = 0

    def reset_epoch_counter(self) -> None:
        """Reset the counter for logged batches at the start of each epoch."""
        self._logged_batches = 0

    def log_if_needed(
        self, batch: dict[str, Any], predictions: list[dict[str, Any]], batch_metrics: dict[str, float], batch_idx: int
    ) -> None:
        """Log problematic images to W&B if conditions are met.

        Args:
            batch: The batch data from the dataloader
            predictions: List of prediction dictionaries
            batch_metrics: Computed metrics for the batch
            batch_idx: Index of the batch in the epoch
        """
        per_batch_cfg = getattr(self.config.logger, "per_batch_image_logging", None)
        if per_batch_cfg is None or not per_batch_cfg.enabled:
            return

        # Check if we should log this batch
        recall_threshold = getattr(per_batch_cfg, "recall_threshold", 1.0)
        if batch_metrics["recall"] >= recall_threshold:
            return

        max_batches_per_epoch = getattr(per_batch_cfg, "max_batches_per_epoch", None)
        if max_batches_per_epoch is not None and max_batches_per_epoch > 0 and self._logged_batches >= max_batches_per_epoch:
            return

        # Log the problematic images
        self._log_problematic_batch(batch, predictions, batch_metrics, batch_idx, per_batch_cfg)
        self._logged_batches += 1

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
