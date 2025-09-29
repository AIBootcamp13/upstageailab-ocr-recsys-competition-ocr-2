from __future__ import annotations

"""Compatibility schema loading for model catalogues.

Keep this in sync with ``configs/schemas/ui_inference_compat.yaml`` and the
protocols in ``docs/ai_handbook/02_protocols/``. Do not hand-edit mappings
without updating the schema files and, when necessary, ``configs/schemas/default_model.yaml``.
If you need broader context, read the maintenance and refactor protocols before
changing the logic here.
"""

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..models.checkpoint import CheckpointMetadata

LOGGER = logging.getLogger(__name__)

DEFAULT_SCHEMA_RELATIVE_PATH = Path("configs") / "schemas" / "ui_inference_compat.yaml"


@dataclass(slots=True)
class ModelFamily:
    id: str
    encoder_model_names: tuple[str, ...]
    decoder_class: str | None
    decoder_output_channels: int | None
    decoder_inner_channels: int | None
    decoder_in_channels: tuple[int, ...]
    head_class: str | None
    head_in_channels: int | None
    description: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ModelFamily:
        encoder_cfg = _normalize_mapping(data.get("encoder"))
        decoder_cfg = _normalize_mapping(data.get("decoder"))
        head_cfg = _normalize_mapping(data.get("head"))
        return cls(
            id=str(data.get("id")),
            encoder_model_names=tuple(str(name) for name in encoder_cfg.get("model_names", [])),
            decoder_class=decoder_cfg.get("class"),
            decoder_output_channels=_maybe_int(decoder_cfg.get("output_channels")),
            decoder_inner_channels=_maybe_int(decoder_cfg.get("inner_channels")),
            decoder_in_channels=tuple(int(value) for value in decoder_cfg.get("in_channels", [])),
            head_class=head_cfg.get("class"),
            head_in_channels=_maybe_int(head_cfg.get("in_channels")),
            description=_maybe_str(data.get("description")),
        )


class ModelCompatibilitySchema:
    def __init__(self, families: Iterable[ModelFamily]):
        self._families: list[ModelFamily] = list(families)
        self._families_by_encoder: dict[str, list[ModelFamily]] = {}
        for family in self._families:
            for encoder_name in family.encoder_model_names:
                self._families_by_encoder.setdefault(encoder_name, []).append(family)

    def validate(self, metadata: CheckpointMetadata) -> CheckpointMetadata:
        encoder_name = metadata.encoder_name
        candidates = self._families_by_encoder.get(encoder_name or "", [])
        if not candidates:
            metadata.issues.append(
                f"No compatibility schema found for encoder '{encoder_name}'. Add an entry to ui_inference_compat.yaml to enable this checkpoint."
            )
            return metadata

        evaluations = [(family, self._collect_issues(metadata, family)) for family in candidates]
        evaluations.sort(
            key=lambda item: (
                len(item[1]),
                item[0].decoder_output_channels if item[0].decoder_output_channels is not None else float("inf"),
                item[0].id,
            )
        )

        family, issues = evaluations[0]
        metadata.schema_family_id = family.id
        metadata.issues.extend(issues)
        return metadata

    def _collect_issues(self, metadata: CheckpointMetadata, family: ModelFamily) -> list[str]:
        issues: list[str] = []

        def _check(value: int | None, expected: int | None, label: str) -> None:
            if expected is None or value is None:
                return
            if int(value) != int(expected):
                issues.append(f"{label} mismatch (checkpoint={value}, expected={expected}) for family '{family.id}'.")

        _check(metadata.decoder.output_channels, family.decoder_output_channels, "Decoder output channels")
        _check(metadata.decoder.inner_channels, family.decoder_inner_channels, "Decoder inner channels")
        _check(metadata.head.in_channels, family.head_in_channels, "Head input channels")

        if family.decoder_in_channels:
            checkpoint_channels = tuple(metadata.decoder.in_channels)
            if checkpoint_channels and checkpoint_channels != family.decoder_in_channels:
                issues.append(f"Decoder in_channels mismatch (checkpoint={checkpoint_channels}, expected={family.decoder_in_channels})")

        return issues


def _discover_default_schema_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / DEFAULT_SCHEMA_RELATIVE_PATH
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Compatibility schema not found relative to {__file__}")


def load_schema(schema_path: Path | None = None) -> ModelCompatibilitySchema:
    if schema_path is None:
        schema_path = _discover_default_schema_path()

    if not schema_path.exists():
        raise FileNotFoundError(f"Compatibility schema not found at {schema_path}")

    with schema_path.open("r", encoding="utf-8") as fp:
        raw_schema = yaml.safe_load(fp) or {}

    families = [ModelFamily.from_mapping(entry) for entry in raw_schema.get("model_families", [])]
    if not families:
        LOGGER.warning("No model families defined in compatibility schema %s", schema_path)

    return ModelCompatibilitySchema(families)


def _maybe_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _maybe_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _normalize_mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}
