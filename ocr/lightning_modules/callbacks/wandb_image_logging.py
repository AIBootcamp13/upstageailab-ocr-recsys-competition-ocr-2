import lightning.pytorch as pl
import numpy as np
from PIL import Image

from ocr.utils.wandb_utils import log_validation_images

EXIF_ORIENTATION = 274  # Orientation Information: 274


class WandbImageLoggingCallback(pl.Callback):
    """Callback to log validation images with bounding boxes to Weights & Biases."""

    def __init__(self, log_every_n_epochs: int = 1, max_images: int = 8):
        self.log_every_n_epochs = log_every_n_epochs
        self.max_images = max_images

    @staticmethod
    def rotate_image(image, orientation):
        """
        Rotate image based on EXIF orientation.
        Handles orientations 1-8 according to EXIF standard.
        Orientation 1 (normal) requires no rotation.
        """
        if orientation == 2:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            image = image.rotate(180)
        elif orientation == 4:
            image = image.rotate(180).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            image = image.rotate(-90, expand=True).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            image = image.rotate(-90, expand=True)
        elif orientation == 7:
            image = image.rotate(90, expand=True).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            image = image.rotate(90, expand=True)
        # Orientation 1 (normal) and any other values: no rotation needed
        return image

    @staticmethod
    def transform_polygons_for_exif(polygons, orientation, image_size):
        """
        Transform polygons to match EXIF-rotated image coordinates.

        Args:
            polygons: List of 1D numpy arrays, each containing flattened coordinates [x1,y1,x2,y2,...]
            orientation: EXIF orientation value
            image_size: (width, height) of original image

        Returns:
            Transformed polygons in the same format
        """
        if not polygons or orientation == 1:  # No rotation needed
            return polygons

        width, height = image_size
        transformed_polygons = []

        for polygon_coords in polygons:
            # polygon_coords is a 1D array like [x1,y1,x2,y2,x3,y3,x4,y4]
            if not isinstance(polygon_coords, np.ndarray) or len(polygon_coords) % 2 != 0:
                # Skip invalid polygons
                transformed_polygons.append(polygon_coords)
                continue

            # Reshape to (N, 2) where N is number of points
            points = polygon_coords.reshape(-1, 2)
            transformed_points = []

            for x, y in points:
                if orientation == 2:  # Flip left-right
                    new_x, new_y = width - x, y
                elif orientation == 3:  # 180 degrees
                    new_x, new_y = width - x, height - y
                elif orientation == 4:  # 180 + flip left-right
                    new_x, new_y = x, height - y
                elif orientation == 5:  # -90 + flip left-right
                    # -90 degrees counter-clockwise, then flip left-right
                    new_x, new_y = height - y, width - x
                    new_x, new_y = width - new_x, new_y  # flip left-right
                elif orientation == 6:  # -90 degrees (90 clockwise)
                    # Rotate 90 degrees clockwise: (x,y) -> (y, width - x)
                    new_x, new_y = y, width - x
                elif orientation == 7:  # 90 + flip left-right
                    # 90 degrees counter-clockwise, then flip left-right
                    new_x, new_y = height - y, x
                    new_x, new_y = width - new_x, new_y  # flip left-right
                elif orientation == 8:  # 90 degrees (90 counter-clockwise)
                    # Rotate 90 degrees counter-clockwise: (x,y) -> (height - y, x)
                    new_x, new_y = height - y, x
                else:
                    new_x, new_y = x, y

                transformed_points.extend([new_x, new_y])

            transformed_polygons.append(np.array(transformed_points))

        return transformed_polygons

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
            if gt_boxes is None:
                gt_quads = []
            elif isinstance(gt_boxes, (list, tuple)):
                try:
                    gt_quads = [item.squeeze().reshape(-1) for item in gt_boxes]
                except Exception as e:
                    print(f"Warning: Failed to process gt_boxes for {filename}: {e}, gt_boxes={gt_boxes}")
                    gt_quads = []
            else:
                print(f"Warning: Unexpected gt_boxes type for {filename}: {type(gt_boxes)}, value={gt_boxes}")
                gt_quads = []

            # Get image directly from filesystem (similar to dataset loading)
            try:
                image_path = val_dataset.image_path / filename  # type: ignore
                image = Image.open(image_path).convert("RGB")

                # Apply EXIF rotation to match dataset preprocessing
                exif = image.getexif()
                original_size = image.size
                if exif and EXIF_ORIENTATION in exif:
                    orientation = exif[EXIF_ORIENTATION]
                    image = self.rotate_image(image, orientation)
                    # Transform ground truth polygons to match rotated image
                    gt_quads = self.transform_polygons_for_exif(gt_quads, orientation, original_size)
                    # Transform prediction polygons to match rotated image
                    pred_boxes = self.transform_polygons_for_exif(pred_boxes, orientation, original_size)

                images.append(image)
                gt_bboxes.append(gt_quads)
                pred_bboxes.append(pred_boxes)
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
