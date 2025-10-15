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

        # Add training info with more detail
        training_parts = []
        if self.epochs is not None:
            training_parts.append(f"ep{self.epochs}")
        elif self.checkpoint_path.stem == "last":
            training_parts.append("last")
        else:
            # Extract step count from filename
            import re

            step_match = re.search(r"step[=_-](\d+)", self.checkpoint_path.stem)
            if step_match:
                step_count = int(step_match.group(1))
                training_parts.append(f"step{step_count}")
            else:
                # Use checkpoint stem, truncated if too long
                stem = self.checkpoint_path.stem
                if len(stem) > 12:
                    stem = stem[:9] + "..."
                training_parts.append(stem)

        # Add experiment name for uniqueness (truncated if too long)
        if self.exp_name and self.exp_name != model_info:
            exp_short = self.exp_name
            # Remove redundant parts that are already in model_info
            for part in [arch, encoder, decoder]:
                if part and part != "unknown":
                    exp_short = exp_short.replace(part, "").replace("--", "-").strip("-")
            if exp_short and len(exp_short) > 15:
                exp_short = exp_short[:12] + "..."
            if exp_short:
                training_parts.insert(0, exp_short)

        # Add validation score for distinction if available
        if self.validation_loss is not None:
            training_parts.append(f"loss{self.validation_loss:.3f}")
        elif self.hmean is not None:
            training_parts.append(f"hmean{self.hmean:.3f}")

        training_info = " · ".join(training_parts)
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


@dataclass(slots=True)
class CheckpointInfo:
    """Lightweight checkpoint information for fast catalog building."""

    checkpoint_path: Path
    config_path: Path | None = None
    display_name: str = ""
    exp_name: str | None = None
    epochs: int | None = None
    created_timestamp: str | None = None
    hmean: float | None = None

    @property
    def is_valid(self) -> bool:
        return True  # Basic info is always valid

    def short_label(self) -> str:
        return self.display_name or self.checkpoint_path.stem

    def to_display_option(self) -> str:
        """Create a descriptive display option for the checkpoint."""
        training_parts = []

        # Add epochs or step information
        if self.epochs is not None:
            training_parts.append(f"ep{self.epochs}")
        elif self.checkpoint_path.stem == "last":
            training_parts.append("last")
        else:
            # Extract step count from filename
            import re

            step_match = re.search(r"step[=_-](\d+)", self.checkpoint_path.stem)
            if step_match:
                step_count = int(step_match.group(1))
                training_parts.append(f"step{step_count}")
            else:
                # Use checkpoint stem, truncated if too long
                stem = self.checkpoint_path.stem
                if len(stem) > 20:
                    stem = stem[:17] + "..."
                training_parts.append(stem)

        # Add experiment name for context and uniqueness
        if self.exp_name:
            # Create a concise, meaningful display name from experiment name
            exp_display = self._create_concise_exp_name(self.exp_name)
            # Allow longer experiment names
            if len(exp_display) > 30:
                exp_display = exp_display[:27] + "..."
            training_parts.insert(0, exp_display)

        # Add hmean score if available
        if self.hmean is not None:
            training_parts.append(f"hmean{self.hmean:.3f}")

        training_info = " · ".join(training_parts)
        return training_info

    def _create_concise_exp_name(self, exp_name: str) -> str:
        """Create a concise, readable name from experiment name."""
        # Handle common experiment naming patterns
        # Experiment names may use underscores or hyphens
        parts = exp_name.replace("_", "-").split("-")

        # Look for architecture patterns
        architecture = None
        for part in parts:
            if part in ["dbnet", "dbnetpp", "craft", "pan", "psenet"]:
                architecture = part
                break

        # Look for encoder patterns
        encoder = None
        for part in parts:
            if "resnet" in part or "mobilenet" in part or "efficientnet" in part or "vgg" in part:
                encoder = part
                break

        # Look for key features - be more specific
        features = []
        if "polygons" in exp_name or "polygon" in exp_name:
            features.append("poly")
            # Check for polygon-related modifiers
            if "_add_" in exp_name and "polygons" in exp_name:
                features.append("add")
            elif "_no_" in exp_name and "polygons" in exp_name:
                features.append("no")

        # Build concise name
        name_parts = []
        if architecture:
            name_parts.append(architecture)
        if encoder:
            name_parts.append(encoder)
        if features:
            name_parts.extend(features)

        if name_parts:
            concise_name = "-".join(name_parts)
        else:
            # Fallback: take first meaningful parts
            meaningful_parts = [p for p in parts if len(p) > 2 and not p.isdigit()]
            concise_name = "-".join(meaningful_parts[:3])

        # Final fallback if still too long or empty
        if len(concise_name) > 25:
            concise_name = concise_name[:22] + "..."
        elif not concise_name:
            concise_name = exp_name[:22] + "..." if len(exp_name) > 25 else exp_name

        return concise_name

    def load_full_metadata(self, schema: Any = None) -> CheckpointMetadata:
        """Load the complete metadata for this checkpoint."""
        from ..services.checkpoint_catalog import CatalogOptions, _collect_metadata

        # Create options with default config filenames
        options = CatalogOptions(
            outputs_dir=self.checkpoint_path.parent.parent.parent,
            hydra_config_filenames=("config.yaml", "hparams.yaml", "train.yaml", "predict.yaml"),
        )
        metadata = _collect_metadata(self.checkpoint_path, options)
        if schema:
            metadata = schema.validate(metadata)
        return metadata
