#!/usr/bin/env python3
"""Utility for benchmarking decoder variants via Hydra configurations."""

from __future__ import annotations

import csv
import sys
import time
from collections.abc import Iterable
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

try:
    import torch
except ImportError:  # pragma: no cover - torch is an optional runtime dependency in some environments
    torch = None  # type: ignore[assignment]

import hydra
import lightning.pytorch as pl
from hydra.utils import get_original_cwd
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, OmegaConf

from ocr.lightning_modules import get_pl_modules_by_cfg
from ocr.utils.path_utils import get_path_resolver


@dataclass(slots=True)
class DecoderSpec:
    """Configuration for a single decoder benchmark run."""

    key: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    checkpoint: str | None = None
    trainer_overrides: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None


@dataclass(slots=True)
class BenchmarkSettings:
    """Top-level benchmark settings extracted from Hydra config."""

    base_exp_name: str
    output_dir: Path
    results_filename: str
    metric_key: str
    evaluate_split: str
    run_test: bool
    skip_training: bool
    limit_train_batches: float | int | None
    limit_val_batches: float | int | None
    limit_test_batches: float | int | None
    resume_from_checkpoint: str | None
    trainer_overrides: dict[str, Any]
    logger_overrides: dict[str, Any]


def _convert_decoder_specs(config: Iterable[DictConfig | dict[str, Any]]) -> list[DecoderSpec]:
    specs: list[DecoderSpec] = []
    for entry in config:
        raw_entry = OmegaConf.to_container(entry, resolve=True) if isinstance(entry, DictConfig) else entry
        if raw_entry is None:
            raise ValueError("Decoder specification cannot be None.")
        if not isinstance(raw_entry, dict):
            raise TypeError(f"Decoder specification must be a mapping, received {type(raw_entry)!r}.")
        raw = dict(cast(dict[str, Any], raw_entry))
        specs.append(
            DecoderSpec(
                key=str(raw.get("key")),
                name=str(raw.get("name")),
                params=dict(raw.get("params", {})),
                checkpoint=raw.get("checkpoint"),
                trainer_overrides=dict(raw.get("trainer_overrides", {})),
                notes=raw.get("notes"),
            )
        )
    return specs


def _safe_to_dict(data: DictConfig | dict[str, Any] | None) -> dict[str, Any]:
    if data is None:
        return {}
    if isinstance(data, DictConfig):
        converted = OmegaConf.to_container(data, resolve=True)
        if converted is None:
            return {}
        if not isinstance(converted, dict):
            raise TypeError("Expected mapping when converting configuration to dict.")
        return dict(cast(dict[str, Any], converted))
    return dict(data)


def _prepare_settings(cfg: DictConfig, root_dir: Path) -> BenchmarkSettings:
    benchmark_cfg = cfg.benchmark
    raw_container = OmegaConf.to_container(benchmark_cfg, resolve=True)
    if raw_container is None:
        raw: dict[str, Any] = {}
    elif not isinstance(raw_container, dict):
        raise TypeError("Benchmark configuration must resolve to a mapping.")
    else:
        raw = cast(dict[str, Any], raw_container)

    output_dir = Path(raw.get("output_dir", "outputs/decoder_benchmark"))
    if not output_dir.is_absolute():
        output_dir = (root_dir / output_dir).resolve()

    resume_path = raw.get("resume_from_checkpoint")
    if resume_path is not None:
        resume_path_obj = Path(resume_path)
        if not resume_path_obj.is_absolute():
            resume_path_obj = (root_dir / resume_path_obj).resolve()
        resume_path = str(resume_path_obj)

    return BenchmarkSettings(
        base_exp_name=str(raw.get("base_exp_name", cfg.exp_name)),
        output_dir=output_dir,
        results_filename=str(raw.get("results_filename", "decoder_benchmark.csv")),
        metric_key=str(raw.get("metric_key", "val/hmean")),
        evaluate_split=str(raw.get("evaluate_split", "val")),
        run_test=bool(raw.get("run_test", False)),
        skip_training=bool(raw.get("skip_training", False)),
        limit_train_batches=raw.get("limit_train_batches"),
        limit_val_batches=raw.get("limit_val_batches"),
        limit_test_batches=raw.get("limit_test_batches"),
        resume_from_checkpoint=resume_path,
        trainer_overrides=_safe_to_dict(raw.get("trainer_overrides")),
        logger_overrides=_safe_to_dict(raw.get("logger_overrides")),
    )


def _base_template(cfg: DictConfig) -> dict[str, Any]:
    container_obj = OmegaConf.to_container(cfg, resolve=True)
    if container_obj is None:
        raise ValueError("Base configuration could not be materialised into a mapping.")
    if not isinstance(container_obj, dict):
        raise TypeError("Expected root configuration to resolve to a mapping.")
    container = dict(cast(dict[str, Any], container_obj))
    container.pop("benchmark", None)
    container.pop("hydra", None)
    return container


def _make_absolute_paths(paths: dict[str, Any], root_dir: Path) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, value in paths.items():
        path_obj = Path(str(value))
        if not path_obj.is_absolute():
            path_obj = (root_dir / path_obj).resolve()
        resolved[key] = str(path_obj)
    return resolved


def _build_run_config(
    template: dict[str, Any],
    decoder: DecoderSpec,
    settings: BenchmarkSettings,
    root_dir: Path,
) -> DictConfig:
    run_container = deepcopy(template)
    exp_name = f"{settings.base_exp_name}_{decoder.key}"
    run_container["exp_name"] = exp_name

    paths = run_container.get("paths", {})
    run_container["paths"] = _make_absolute_paths(paths, root_dir)

    # Apply trainer overrides (global first, then per-decoder)
    trainer_cfg = run_container.get("trainer", {})
    trainer_cfg.update(settings.trainer_overrides)
    trainer_cfg.update(decoder.trainer_overrides)

    for attr, limit_value in (
        ("limit_train_batches", settings.limit_train_batches),
        ("limit_val_batches", settings.limit_val_batches),
        ("limit_test_batches", settings.limit_test_batches),
    ):
        if limit_value is not None:
            trainer_cfg[attr] = limit_value

    trainer_cfg.setdefault("logger", False)
    trainer_cfg.setdefault("enable_checkpointing", False)
    trainer_cfg.setdefault("enable_model_summary", False)
    run_container["trainer"] = trainer_cfg

    # Ensure logger configuration is explicit to avoid accidental wandb usage
    logger_cfg = run_container.get("logger", {})
    if settings.logger_overrides:
        logger_cfg.update(settings.logger_overrides)
    logger_cfg.setdefault("wandb", False)
    run_container["logger"] = logger_cfg

    # Component override for decoder
    model_cfg = run_container.setdefault("model", {})
    overrides = model_cfg.setdefault("component_overrides", {})
    overrides["decoder"] = {"name": decoder.name, "params": dict(decoder.params)}
    model_cfg["component_overrides"] = overrides
    run_container["model"] = model_cfg

    # Handle checkpoint usage
    checkpoint_path = decoder.checkpoint or settings.resume_from_checkpoint
    run_container["resume"] = checkpoint_path

    return OmegaConf.create(run_container)


def _flatten_metrics(prefix: str, metrics: dict[str, float | int | None]) -> dict[str, float | int | None]:
    return {f"{prefix}.{key}": metrics[key] for key in metrics}


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(output_file: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_file.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _is_wandb_enabled(logger_cfg: dict[str, Any]) -> bool:
    wandb_cfg = logger_cfg.get("wandb") if logger_cfg else None
    if isinstance(wandb_cfg, dict):
        return bool(wandb_cfg.get("enabled", True))
    return bool(wandb_cfg)


def _initialise_wandb_logger(
    run_cfg: DictConfig,
    decoder: DecoderSpec,
    logger_cfg: dict[str, Any],
) -> tuple[Any | None, list[Callback]]:
    if not _is_wandb_enabled(logger_cfg):
        return None, []

    try:
        from lightning.pytorch.loggers import WandbLogger
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "W&B logging requested but the 'wandb' package is not installed. "
            "Install wandb to enable logging or disable it in the benchmark configuration."
        ) from exc

    from ocr.utils.wandb_utils import generate_run_name, load_env_variables

    load_env_variables()

    tags = logger_cfg.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]

    config_payload = OmegaConf.to_container(run_cfg, resolve=True)

    run_name = logger_cfg.get("run_name")
    if not run_name:
        try:
            run_name = generate_run_name(run_cfg)
        except Exception:
            run_name = str(run_cfg.get("exp_name", "decoder_benchmark"))
    if decoder.key:
        run_name = f"{run_name}_{decoder.key}"

    wandb_logger_kwargs: dict[str, Any] = {
        "name": run_name,
        "project": logger_cfg.get("project_name", "decoder-benchmark"),
        "entity": logger_cfg.get("entity"),
        "group": logger_cfg.get("group"),
        "tags": tags,
        "job_type": logger_cfg.get("job_type", "decoder-benchmark"),
        "offline": bool(logger_cfg.get("offline", False)),
        "log_model": bool(logger_cfg.get("log_model", False)),
        "config": config_payload,
    }

    save_dir_value = logger_cfg.get("save_dir")
    if save_dir_value is not None:
        wandb_logger_kwargs["save_dir"] = save_dir_value

    wandb_logger = WandbLogger(**wandb_logger_kwargs)

    callbacks: list[Callback] = []
    image_log_every = int(logger_cfg.get("image_log_every_n_epochs", 5))
    max_images = int(logger_cfg.get("image_log_max_images", 8))
    if image_log_every > 0 and max_images > 0:
        try:
            from ocr.lightning_modules.callbacks.wandb_image_logging import (
                WandbImageLoggingCallback,
            )

            callbacks.append(
                WandbImageLoggingCallback(
                    log_every_n_epochs=image_log_every,
                    max_images=max_images,
                )
            )
        except Exception as exc:  # pragma: no cover - non-critical logging utility
            print(f"Warning: Failed to initialise W&B image logging callback: {exc}")

    return wandb_logger, callbacks


def _extract_metric(metrics: dict[str, Any], metric_key: str) -> float | None:
    value = metrics.get(metric_key)
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_single_decoder(
    run_cfg: DictConfig,
    decoder: DecoderSpec,
    settings: BenchmarkSettings,
) -> dict[str, Any]:
    pl.seed_everything(run_cfg.get("seed", 42), workers=True)

    model_module, data_module = get_pl_modules_by_cfg(run_cfg)

    trainer_kwargs = dict(run_cfg.get("trainer", {}))
    callbacks: list[Callback] = []

    logger_cfg = _safe_to_dict(run_cfg.get("logger"))
    wandb_logger, wandb_callbacks = _initialise_wandb_logger(run_cfg, decoder, logger_cfg)
    callbacks.extend(wandb_callbacks)

    existing_callbacks = trainer_kwargs.pop("callbacks", None)
    if isinstance(existing_callbacks, list):
        callbacks.extend(existing_callbacks)
    elif existing_callbacks is not None:
        callbacks.append(existing_callbacks)

    logger_setting = trainer_kwargs.pop("logger", None)

    trainer_init_kwargs = dict(trainer_kwargs)
    if callbacks:
        trainer_init_kwargs["callbacks"] = callbacks

    if wandb_logger is not None:
        trainer_init_kwargs["logger"] = wandb_logger
    elif logger_setting is not None:
        trainer_init_kwargs["logger"] = logger_setting

    trainer = pl.Trainer(**trainer_init_kwargs)

    metrics_summary: dict[str, Any] = {
        "decoder": decoder.key,
        "decoder_name": decoder.name,
        "status": "success",
        "metric_key": settings.metric_key,
        "metric_value": None,
        "train_time_sec": 0.0,
        "notes": decoder.notes,
    }

    start_time = time.perf_counter()
    try:
        resume_path = run_cfg.get("resume")
        if not settings.skip_training:
            trainer.fit(model_module, data_module, ckpt_path=resume_path)

        val_metrics: dict[str, Any] = {}
        test_metrics: dict[str, Any] = {}

        evaluate_split = settings.evaluate_split.lower()
        if evaluate_split in {"val", "both"}:
            val_ckpt = resume_path if settings.skip_training else None
            if val_outputs := trainer.validate(
                model_module,
                datamodule=data_module,
                ckpt_path=val_ckpt,
                verbose=False,
            ):
                val_metrics = {k: float(v) for k, v in val_outputs[0].items()}
                metrics_summary.update(_flatten_metrics("val", val_metrics))

        if settings.run_test or evaluate_split in {"test", "both"}:
            test_ckpt = resume_path if settings.skip_training else None
            if test_outputs := trainer.test(
                model_module,
                datamodule=data_module,
                ckpt_path=test_ckpt,
                verbose=False,
            ):
                test_metrics = {k: float(v) for k, v in test_outputs[0].items()}
                metrics_summary.update(_flatten_metrics("test", test_metrics))

        if metrics_summary["metric_value"] is None:
            combined_metrics = val_metrics | test_metrics
            metrics_summary["metric_value"] = _extract_metric(combined_metrics, settings.metric_key)

    except Exception as exc:  # noqa: BLE001
        metrics_summary["status"] = "failed"
        metrics_summary["error"] = str(exc)
    finally:
        end_time = time.perf_counter()
        metrics_summary["train_time_sec"] = round(end_time - start_time, 2)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if wandb_logger is not None:
        try:
            wandb_logger.experiment.summary.update(  # type: ignore[call-arg]
                {key: value for key, value in metrics_summary.items() if isinstance(value, str | int | float)}
            )
        except Exception as exc:  # pragma: no cover - logging robustness
            print(f"Warning: Failed to update W&B summary for {decoder.key}: {exc}")
        finally:
            with suppress(Exception):
                wandb_logger.finalize(metrics_summary.get("status", "success"))

    return metrics_summary


@hydra.main(config_path=str(get_path_resolver().config.config_dir), config_name="benchmark/decoder", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Benchmark multiple decoder configurations sequentially."""

    root_dir = Path(get_original_cwd())
    settings = _prepare_settings(cfg, root_dir)
    decoder_specs = _convert_decoder_specs(cfg.benchmark.decoders)

    if not decoder_specs:
        print("No decoders provided for benchmarking.")
        sys.exit(1)

    template = _base_template(cfg)

    results: list[dict[str, Any]] = []
    failures = 0

    for spec in decoder_specs:
        run_cfg = _build_run_config(template, spec, settings, root_dir)
        result = _run_single_decoder(run_cfg, spec, settings)
        if result.get("status") != "success":
            failures += 1
        results.append(result)
        metric_value = result.get("metric_value")
        metric_display = f"{metric_value:.4f}" if isinstance(metric_value, float) else "n/a"
        print(f"[{result['status']}] {spec.key} → {settings.metric_key}: {metric_display}")

    _ensure_output_dir(settings.output_dir)
    output_file = settings.output_dir / settings.results_filename
    _write_csv(output_file, results)

    print(f"Results saved to: {output_file}")

    if failures:
        print(f"{failures} decoder runs failed. See CSV for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
