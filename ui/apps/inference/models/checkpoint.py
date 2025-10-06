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
    validation_loss: float | None = None  # Renamed from validation_score
    created_timestamp: str | None = None
    recall: float | None = None
    hmean: float | None = None
    precision: float | None = None

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def short_label(self) -> str:
        return self.display_name or self.checkpoint_path.stem

    def to_display_option(self) -> str:
        # Parse experiment name for more descriptive components
        arch = self.architecture
        encoder = self.backbone or self.encoder_name or "unknown"
        decoder = "unknown"

        # Try to extract decoder from experiment name
        if self.exp_name:
            exp_parts = self.exp_name.split("-")
            # Look for decoder patterns in experiment name
            for part in exp_parts:
                if "decoder" in part:
                    # Shorten common decoder names for display
                    if "fpn" in part:
                        decoder = "fpn"
                    elif "pan" in part:
                        decoder = "pan"
                    else:
                        decoder = part.replace("_decoder", "")
                    break
                elif part in ["unet", "fpn", "pan", "dbnetpp"]:
                    decoder = part
                    break

        # Create model identifier: arch-encoder-decoder
        model_parts = [arch, encoder, decoder]
        model_info = "-".join(part for part in model_parts if part and part != "unknown")

        # Fallback to just encoder if parsing failed
        if not model_info or model_info == arch:
            model_info = encoder if encoder != "unknown" else arch

        # Add training info
        if self.epochs is not None:
            training_info = f"ep{self.epochs}"
        elif self.checkpoint_path.stem == "last":
            training_info = "last"
        else:
            # Extract step count from filename
            import re

            step_match = re.search(r"step[=_-](\d+)", self.checkpoint_path.stem)
            if step_match:
                step_count = int(step_match.group(1))
                training_info = f"step{step_count}"
            else:
                # Use checkpoint stem, truncated if too long
                training_info = self.checkpoint_path.stem
                if len(training_info) > 12:
                    training_info = training_info[:9] + "..."

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
            "validation_loss": self.validation_loss,
            "created_timestamp": self.created_timestamp,
            "recall": self.recall,
            "hmean": self.hmean,
        }
