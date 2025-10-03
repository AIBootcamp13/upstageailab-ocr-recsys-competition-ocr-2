from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DecoderSignature:
    in_channels: list[int] = field(default_factory=list)
    inner_channels: int | None = None
    output_channels: int | None = None


@dataclass(slots=True)
class HeadSignature:
    in_channels: int | None = None


@dataclass(slots=True)
class CheckpointMetadata:
    checkpoint_path: Path
    config_path: Path | None = None
    display_name: str = ""
    architecture: str = "unknown"
    backbone: str = "unknown"
    epochs: int | None = None
    exp_name: str | None = None
    decoder: DecoderSignature = field(default_factory=DecoderSignature)
    head: HeadSignature = field(default_factory=HeadSignature)
    encoder_name: str | None = None
    schema_family_id: str | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def short_label(self) -> str:
        return self.display_name or self.checkpoint_path.stem

    def to_display_option(self) -> str:
        # Create concise display name
        model_info = f"{self.backbone}" if self.backbone and self.backbone != "unknown" else self.architecture

        # Add key training info
        if self.epochs is not None:
            training_info = f"ep{self.epochs}"
        else:
            # Extract step count from filename for concise display
            import re

            step_match = re.search(r"step_(\d+)", self.checkpoint_path.stem)
            training_info = f"step_{step_match.group(1)}" if step_match else self.checkpoint_path.stem[:20]

        return f"{model_info} · {training_info}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "config_path": str(self.config_path) if self.config_path else None,
            "display_name": self.display_name,
            "architecture": self.architecture,
            "backbone": self.backbone,
            "epochs": self.epochs,
            "exp_name": self.exp_name,
            "decoder": {
                "in_channels": self.decoder.in_channels,
                "inner_channels": self.decoder.inner_channels,
                "output_channels": self.decoder.output_channels,
            },
            "head": {
                "in_channels": self.head.in_channels,
            },
            "encoder_name": self.encoder_name,
            "schema_family_id": self.schema_family_id,
            "issues": self.issues,
        }
