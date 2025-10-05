#!/usr/bin/env python3
"""Visualize model predictions on images with Streamlit-compatible scaling."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont

# Prefer installing the project in editable mode, but preserve the original behaviour for now.
sys.path.append("/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2")

from ocr.utils.orientation import normalize_pil_image, remap_polygons
from ocr.utils.path_utils import setup_paths
from ui.utils.inference import InferenceEngine

setup_paths()


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _clip_polygons_in_place(polygons: Sequence[np.ndarray], width: int, height: int) -> None:
    for polygon in polygons:
        if polygon is None:
            continue
        polygon[..., 0] = np.clip(polygon[..., 0], 0, max(width - 1, 0))
        polygon[..., 1] = np.clip(polygon[..., 1], 0, max(height - 1, 0))


def _filter_degenerate_polygons(polygons: Sequence[np.ndarray], min_side: float = 1.0) -> list[np.ndarray]:
    filtered: list[np.ndarray] = []
    for polygon in polygons:
        if polygon is None or polygon.size == 0:
            continue
        reshaped = polygon.reshape(-1, 2)
        if reshaped.shape[0] < 3:
            continue
        width_span = float(reshaped[:, 0].max() - reshaped[:, 0].min())
        height_span = float(reshaped[:, 1].max() - reshaped[:, 1].min())
        if width_span < min_side or height_span < min_side:
            continue
        filtered.append(polygon)
    return filtered


def _postprocess_polygons(polygons: Sequence[np.ndarray], image_size: tuple[int, int]) -> list[np.ndarray]:
    if not polygons:
        return []
    processed = [np.array(polygon, copy=True) for polygon in polygons if len(polygon) > 0]
    if not processed:
        return []
    width, height = image_size
    _clip_polygons_in_place(processed, width, height)
    return _filter_degenerate_polygons(processed)


def _parse_polygon_string(polygon_str: str) -> list[np.ndarray]:
    if not polygon_str:
        return []
    polygons: list[np.ndarray] = []
    for raw_polygon in polygon_str.split("|"):
        raw_polygon = raw_polygon.strip()
        if not raw_polygon:
            continue
        coords = [coord for coord in raw_polygon.split(",") if coord]
        if len(coords) < 8 or len(coords) % 2:
            continue
        try:
            points = [(float(coords[i]), float(coords[i + 1])) for i in range(0, len(coords), 2)]
        except ValueError:
            continue
        polygons.append(np.array(points, dtype=np.float32))
    return polygons


def _pair_predictions(
    polygons: Sequence[np.ndarray],
    confidences: Sequence[float] | None,
    score_threshold: float | None,
) -> list[tuple[np.ndarray, float | None]]:
    if not polygons:
        return []

    pairs: list[tuple[np.ndarray, float | None]] = []
    if confidences:
        limit = min(len(polygons), len(confidences))
        for idx in range(limit):
            score = float(confidences[idx])
            if score_threshold is not None and score < score_threshold:
                continue
            pairs.append((polygons[idx], score))
        pairs.extend((polygons[idx], None) for idx in range(limit, len(polygons)))
    else:
        pairs.extend((polygon, None) for polygon in polygons)

    return pairs


class StreamlitInferenceAdapter:
    """Adaptor that reuses the Streamlit inference engine for batch visualization."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.engine = InferenceEngine()

        checkpoint_path = getattr(cfg, "checkpoint_path", None)
        if not checkpoint_path:
            raise ValueError("checkpoint_path must be provided for visualization.")

        if not self.engine.load_model(checkpoint_path):
            raise RuntimeError(f"Failed to load checkpoint: {checkpoint_path}")

        self.bin_thresh = _maybe_float(cfg.get("postprocess_thresh", None))
        self.box_thresh = _maybe_float(cfg.get("postprocess_box_thresh", None))
        self.max_candidates = _maybe_int(cfg.get("postprocess_max_candidates", None))
        self.min_detection_size = _maybe_int(cfg.get("postprocess_min_detection_size", None))
        self.score_threshold = _maybe_float(cfg.get("score_threshold", None))
        self.use_polygon = _maybe_bool(cfg.get("postprocess_use_polygon", None))

        self.engine.update_postprocessor_params(
            binarization_thresh=self.bin_thresh,
            box_thresh=self.box_thresh,
            max_candidates=self.max_candidates,
            min_detection_size=self.min_detection_size,
        )

        if self.use_polygon is not None:
            head = getattr(self.engine.model, "head", None)
            postprocess = getattr(head, "postprocess", None) if head else None
            if postprocess is not None:
                postprocess.use_polygon = self.use_polygon

    def predict(self, image_paths: Iterable[str]) -> dict[str, dict[str, list[Any]]]:
        predictions: dict[str, dict[str, list[Any]]] = {}

        for image_path in image_paths:
            result = self.engine.predict_image(
                image_path=image_path,
                binarization_thresh=self.bin_thresh,
                box_thresh=self.box_thresh,
                max_candidates=self.max_candidates,
                min_detection_size=self.min_detection_size,
            )

            filename = Path(image_path).name
            polygons: list[np.ndarray] = []
            scores: list[Any] = []

            if result:
                polygons = _parse_polygon_string(result.get("polygons", ""))
                confidences = result.get("confidences")
                paired = _pair_predictions(polygons, confidences, self.score_threshold)
                polygons = [poly for poly, _ in paired]
                scores = [score for _, score in paired]
                print(f"DEBUG: {filename} - boxes: {len(polygons)}")
            else:
                print(f"DEBUG: {filename} - inference returned no predictions.")

            predictions[filename] = {"boxes": polygons, "scores": scores}

        return predictions


def draw_predictions_on_image(image_path: str, predictions: dict[str, Any]) -> Image.Image:
    pil_image = Image.open(image_path)
    raw_width, raw_height = pil_image.size
    normalized_image, orientation = normalize_pil_image(pil_image)

    if normalized_image.mode != "RGB":
        image = normalized_image.convert("RGB")
    else:
        image = normalized_image.copy()

    if normalized_image is not pil_image and normalized_image is not image:
        normalized_image.close()
    pil_image.close()

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    filename = Path(image_path).name
    pred_data = predictions.get(filename)

    if not pred_data:
        print(f"No predictions found for {filename}")
        return image

    raw_boxes = pred_data.get("boxes", [])

    if orientation != 1 and raw_boxes:
        raw_boxes = remap_polygons(raw_boxes, raw_width, raw_height, orientation)

    draw = ImageDraw.Draw(image)
    boxes = _postprocess_polygons(raw_boxes, image.size)
    scores = pred_data.get("scores", []) or [None] * len(boxes)

    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
    for index, box in enumerate(boxes):
        if len(box) < 4:
            continue
        color = colors[index % len(colors)]
        draw.polygon([(float(x), float(y)) for x, y in box], outline=color, width=3)

        score = scores[index] if index < len(scores) else None
        if score is not None:
            label = f"Score: {score:.2f}"
            text_position = (min(point[0] for point in box), min(point[1] for point in box) - 20)
            draw.text(text_position, label, fill=color, font=font)

    print(f"Drew {len(boxes)} boxes on {filename}")
    return image


@hydra.main(config_path="../configs", config_name="predict", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

    image_dir = Path(cfg.image_dir)
    image_paths = sorted([str(path) for ext in ("*.jpg", "*.jpeg", "*.png") for path in image_dir.glob(ext)])
    image_paths = image_paths[: cfg.max_images]

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_paths)} images to process.")

    adapter = StreamlitInferenceAdapter(cfg)
    predictions = adapter.predict(image_paths)

    num_images = len(image_paths)
    fig, axes = plt.subplots(1, num_images, figsize=(6 * num_images, 8))
    if num_images == 1:
        axes = [axes]

    for axis, image_path in zip(axes, image_paths, strict=True):
        path = Path(image_path)
        print(f"Visualizing {path.name}...")
        annotated = draw_predictions_on_image(str(path), predictions)
        axis.imshow(annotated)
        axis.set_title(path.name)
        axis.axis("off")

    plt.tight_layout()

    if cfg.save_dir:
        save_path = Path(cfg.save_dir) / "prediction_visualization.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
