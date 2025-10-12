"""Utility helpers for formatting OCR model predictions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


def format_predictions(batch: dict[str, Any], boxes_batch: Sequence[Iterable[np.ndarray]]) -> list[dict[str, Any]]:
    """Normalize model polygon outputs into the Lightning data contract shape.

    Args:
        batch: Mini-batch produced by the collate function.
        boxes_batch: Sequence of polygon collections corresponding to each item in the batch.

    Returns:
        List of dictionaries matching the prediction contract consumed by evaluators/loggers.
    """
    predictions: list[dict[str, Any]] = []
    for idx, boxes in enumerate(boxes_batch):
        normalized_boxes = [np.asarray(box, dtype=np.float32).reshape(-1, 2) for box in boxes]

        metadata_entry: dict[str, Any] | None = None
        metadata_batch = batch.get("metadata")
        if metadata_batch is not None and idx < len(metadata_batch):
            raw_metadata = metadata_batch[idx]
            if raw_metadata is not None:
                if hasattr(raw_metadata, "model_dump"):
                    metadata_entry = raw_metadata.model_dump()  # type: ignore[assignment]
                elif isinstance(raw_metadata, dict):
                    metadata_entry = dict(raw_metadata)
                else:
                    metadata_entry = {"value": raw_metadata}

                if metadata_entry is not None and "path" in metadata_entry:
                    metadata_entry["path"] = str(metadata_entry["path"])

        predictions.append(
            {
                "boxes": normalized_boxes,
                "orientation": batch.get("orientation", [1])[idx] if "orientation" in batch else 1,
                "metadata": metadata_entry,
                "raw_size": tuple(batch.get("raw_size", [(0, 0)])[idx]) if "raw_size" in batch else None,
                "canonical_size": tuple(batch.get("canonical_size", [None])[idx]) if "canonical_size" in batch else None,
                "image_path": batch.get("image_path", [None])[idx] if "image_path" in batch else None,
            }
        )
    return predictions
