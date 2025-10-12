#!/usr/bin/env python
"""CLI for priming dataset caches and running preprocessing transforms.

This script replaces the legacy OCRDataset-based preprocessing entrypoint with a
Hydra-driven workflow. It instantiates a ``ValidatedOCRDataset`` directly from a
``DatasetConfig`` structure, ensuring parity with the runtime configuration used
by the training and inference pipelines.

Usage example (processing the training split):

    uv run python scripts/preprocess_data.py dataset_key=train_dataset limit=100

The script will iterate through the requested dataset, materialising any cache
artifacts configured in ``DatasetConfig.cache_config`` (e.g., tensor cache,
preloaded images) while exercising the full transform stack.
"""

from __future__ import annotations

import logging
from typing import Any

import hydra
from omegaconf import DictConfig

from ocr.datasets import ValidatedOCRDataset

log = logging.getLogger(__name__)


def _materialise_dataset(dataset: ValidatedOCRDataset, *, limit: int | None) -> int:
    """Iterate the dataset to trigger preprocessing and caching side-effects."""
    processed = 0
    total = len(dataset)
    target = min(limit, total) if limit is not None else total

    for index in range(target):
        try:
            dataset[index]
        except Exception as exc:  # noqa: BLE001 - emit detailed context for CLI users
            raise RuntimeError(f"Dataset preprocessing failed at index {index}: {exc}") from exc
        processed += 1

    return processed


def _resolve_dataset_cfg(cfg: DictConfig, dataset_key: str) -> DictConfig:
    datasets = cfg.get("datasets")
    if datasets is None:
        raise KeyError("'datasets' section is missing from the loaded Hydra config")

    if dataset_key not in datasets:
        available = ", ".join(sorted(datasets.keys()))
        raise KeyError(f"Dataset '{dataset_key}' not found. Available keys: {available}")

    dataset_cfg = datasets[dataset_key]
    if "config" not in dataset_cfg:
        raise KeyError(f"Dataset '{dataset_key}' is missing the 'config' block required for instantiation")

    return dataset_cfg


@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point executed by Hydra."""
    logging.basicConfig(level=logging.INFO)

    dataset_key: str = cfg.get("dataset_key", "train_dataset")
    limit_value: Any = cfg.get("limit")
    limit: int | None = int(limit_value) if limit_value is not None else None

    dataset_cfg = _resolve_dataset_cfg(cfg, dataset_key)
    log.info("Instantiating dataset '%s' via Hydra", dataset_key)

    dataset: ValidatedOCRDataset = hydra.utils.instantiate(dataset_cfg)

    log.info("Dataset ready (size=%d, cache tensors=%s)", len(dataset), dataset.cache_transformed_tensors)

    processed = _materialise_dataset(dataset, limit=limit)

    log.info("Preprocessing finished: %d samples processed", processed)


if __name__ == "__main__":
    main()
