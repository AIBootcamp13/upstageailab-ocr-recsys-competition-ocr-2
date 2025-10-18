"""Pydantic models for checkpoint metadata and catalog entries.

This module defines the data structures for the checkpoint catalog V2 system,
including the YAML metadata schema and catalog entry models.

All models use Pydantic V2 for validation and serialization.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TrainingInfo(BaseModel):
    """Training progress information.

    Attributes:
        epoch: Current training epoch (0-indexed)
        global_step: Global training step across all epochs
        training_phase: Training phase identifier
        max_epochs: Maximum epochs configured in trainer
    """

    epoch: int = Field(
        ...,
        ge=0,
        description="Current training epoch (0-indexed)",
    )

    global_step: int = Field(
        ...,
        ge=0,
        description="Global training step across all epochs",
    )

    training_phase: Literal["training", "validation", "finetuning"] = Field(
        default="training",
        description="Training phase identifier",
    )

    max_epochs: int | None = Field(
        default=None,
        ge=1,
        description="Maximum epochs configured in trainer",
    )


class EncoderInfo(BaseModel):
    """Encoder/backbone information.

    Attributes:
        model_name: Encoder model name (e.g., 'resnet50', 'mobilenetv3_large_100')
        pretrained: Whether encoder uses pretrained weights
        frozen: Whether encoder weights are frozen during training
    """

    model_name: str = Field(
        ...,
        description="Encoder model name (e.g., 'resnet50', 'mobilenetv3_large_100')",
    )

    pretrained: bool = Field(
        default=True,
        description="Whether encoder uses pretrained weights",
    )

    frozen: bool = Field(
        default=False,
        description="Whether encoder weights are frozen during training",
    )


class DecoderInfo(BaseModel):
    """Decoder configuration and signature.

    Attributes:
        name: Decoder type (e.g., 'pan_decoder', 'fpn_decoder', 'unet')
        in_channels: Input channel dimensions from encoder feature pyramid
        inner_channels: Internal feature channels (for FPN/PAN)
        output_channels: Output feature channels
        params: Additional decoder-specific parameters
    """

    name: str = Field(
        ...,
        description="Decoder type (e.g., 'pan_decoder', 'fpn_decoder', 'unet')",
    )

    in_channels: list[int] = Field(
        default_factory=list,
        description="Input channel dimensions from encoder feature pyramid",
    )

    inner_channels: int | None = Field(
        default=None,
        ge=1,
        description="Internal feature channels (for FPN/PAN)",
    )

    output_channels: int | None = Field(
        default=None,
        ge=1,
        description="Output feature channels",
    )

    params: dict[str, int | float | bool | str] = Field(
        default_factory=dict,
        description="Additional decoder-specific parameters",
    )


class HeadInfo(BaseModel):
    """Detection head configuration and signature.

    Attributes:
        name: Head type (e.g., 'db_head', 'craft_head')
        in_channels: Input channels from decoder
        params: Head-specific parameters (e.g., box_thresh, max_candidates)
    """

    name: str = Field(
        ...,
        description="Head type (e.g., 'db_head', 'craft_head')",
    )

    in_channels: int | None = Field(
        default=None,
        ge=1,
        description="Input channels from decoder",
    )

    params: dict[str, int | float | bool | str] = Field(
        default_factory=dict,
        description="Head-specific parameters (e.g., box_thresh, max_candidates)",
    )


class LossInfo(BaseModel):
    """Loss function configuration.

    Attributes:
        name: Loss function name (e.g., 'db_loss', 'craft_loss')
        params: Loss-specific parameters (e.g., alpha, beta, weights)
    """

    name: str = Field(
        ...,
        description="Loss function name (e.g., 'db_loss', 'craft_loss')",
    )

    params: dict[str, float] = Field(
        default_factory=dict,
        description="Loss-specific parameters (e.g., alpha, beta, weights)",
    )


class ModelInfo(BaseModel):
    """Model architecture and components.

    Attributes:
        architecture: Model architecture name
        encoder: Encoder/backbone configuration
        decoder: Decoder configuration and signature
        head: Detection head configuration and signature
        loss: Loss function configuration
    """

    architecture: str = Field(
        ...,
        description="Model architecture name (e.g., 'dbnet', 'craft', 'pan')",
    )

    encoder: EncoderInfo = Field(
        ...,
        description="Encoder/backbone configuration",
    )

    decoder: DecoderInfo = Field(
        ...,
        description="Decoder configuration and signature",
    )

    head: HeadInfo = Field(
        ...,
        description="Detection head configuration and signature",
    )

    loss: LossInfo = Field(
        ...,
        description="Loss function configuration",
    )


class MetricsInfo(BaseModel):
    """Validation metrics.

    Required fields: precision, recall, hmean (as per user requirements)

    Attributes:
        precision: Validation precision score (required)
        recall: Validation recall score (required)
        hmean: Validation harmonic mean F1 score (required)
        validation_loss: Validation loss value
        additional_metrics: Additional tracked metrics
    """

    precision: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Validation precision score (required)",
    )

    recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Validation recall score (required)",
    )

    hmean: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Validation harmonic mean (F1 score) (required)",
    )

    validation_loss: float | None = Field(
        default=None,
        ge=0.0,
        description="Validation loss value",
    )

    additional_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Additional tracked metrics (e.g., 'cleval/precision', 'val/iou')",
    )


class CheckpointingConfig(BaseModel):
    """ModelCheckpoint callback configuration.

    Attributes:
        monitor: Metric being monitored
        mode: Optimization mode for monitored metric
        save_top_k: Number of best checkpoints to save
        save_last: Whether to save 'last.ckpt' checkpoint
    """

    monitor: str = Field(
        ...,
        description="Metric being monitored (e.g., 'val/hmean', 'val/loss')",
    )

    mode: Literal["min", "max"] = Field(
        ...,
        description="Optimization mode for monitored metric",
    )

    save_top_k: int = Field(
        default=1,
        ge=-1,
        description="Number of best checkpoints to save (-1 = all)",
    )

    save_last: bool = Field(
        default=True,
        description="Whether to save 'last.ckpt' checkpoint",
    )


class CheckpointMetadataV1(BaseModel):
    """Checkpoint metadata schema V1.

    This is the primary data structure stored in `.metadata.yaml` files
    generated during training. Designed for fast loading without checkpoint access.

    File size: ~2-5 KB
    Load time: ~5-10ms (vs 2-5 seconds for checkpoint loading)

    Required metrics per user requirements:
        - precision, recall, hmean, epoch

    Attributes:
        schema_version: Metadata schema version for migration support
        checkpoint_path: Relative path to checkpoint file from outputs directory
        exp_name: Experiment name (directory name containing checkpoint)
        created_at: ISO 8601 timestamp when checkpoint was created
        training: Training progress information (includes epoch)
        model: Model architecture and component configuration
        metrics: Validation metrics (includes precision, recall, hmean)
        checkpointing: ModelCheckpoint callback configuration
        hydra_config_path: Optional relative path to Hydra config.yaml
        wandb_run_id: Optional Wandb run ID for online artifact retrieval
    """

    schema_version: Literal["1.0"] = Field(
        default="1.0",
        description="Metadata schema version for migration support",
    )

    checkpoint_path: str = Field(
        ...,
        description="Relative path to checkpoint file from outputs directory",
    )

    exp_name: str = Field(
        ...,
        description="Experiment name (directory name containing checkpoint)",
    )

    created_at: str = Field(
        ...,
        description="ISO 8601 timestamp when checkpoint was created",
    )

    training: TrainingInfo = Field(
        ...,
        description="Training progress information (includes epoch)",
    )

    model: ModelInfo = Field(
        ...,
        description="Model architecture and component configuration",
    )

    metrics: MetricsInfo = Field(
        ...,
        description="Validation metrics (includes precision, recall, hmean)",
    )

    checkpointing: CheckpointingConfig = Field(
        ...,
        description="ModelCheckpoint callback configuration",
    )

    hydra_config_path: str | None = Field(
        default=None,
        description="Relative path to Hydra config.yaml from outputs dir",
    )

    wandb_run_id: str | None = Field(
        default=None,
        description="Wandb run ID for online artifact retrieval",
    )

    @field_validator("created_at")
    @classmethod
    def validate_iso_timestamp(cls, v: str) -> str:
        """Ensure created_at is valid ISO 8601 timestamp."""
        try:
            datetime.fromisoformat(v)
        except ValueError as exc:
            msg = f"Invalid ISO 8601 timestamp: {v}"
            raise ValueError(msg) from exc
        return v

    model_config = {
        "validate_assignment": True,
        "str_strip_whitespace": True,
    }


class CheckpointCatalogEntry(BaseModel):
    """Lightweight catalog entry for UI dropdown display.

    This is the model returned by the catalog builder for fast UI rendering.
    Includes only essential display information without full metadata.

    Attributes:
        checkpoint_path: Absolute path to checkpoint file
        config_path: Resolved config file path for inference
        display_name: Human-readable display name for UI
        architecture: Model architecture
        backbone: Encoder model name
        exp_name: Experiment name
        epochs: Training epoch number
        created_timestamp: Creation timestamp (YYYYmmdd_HHMM format)
        hmean: Harmonic mean score (required per user)
        precision: Precision score (required per user)
        recall: Recall score (required per user)
        monitor: Monitored metric
        monitor_mode: Monitor optimization mode
        has_metadata: Whether checkpoint has .metadata.yaml file
    """

    checkpoint_path: Path = Field(
        ...,
        description="Absolute path to checkpoint file",
    )

    config_path: Path | None = Field(
        default=None,
        description="Resolved config file path for inference",
    )

    display_name: str = Field(
        ...,
        description="Human-readable display name for UI",
    )

    architecture: str = Field(
        ...,
        description="Model architecture (e.g., 'dbnet')",
    )

    backbone: str = Field(
        ...,
        description="Encoder model name (e.g., 'resnet50')",
    )

    exp_name: str = Field(
        ...,
        description="Experiment name",
    )

    epochs: int = Field(
        ...,
        ge=0,
        description="Training epoch number (required per user)",
    )

    created_timestamp: str = Field(
        ...,
        description="Creation timestamp (YYYYmmdd_HHMM format)",
    )

    hmean: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Harmonic mean score (required per user)",
    )

    precision: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Precision score (required per user)",
    )

    recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Recall score (required per user)",
    )

    monitor: str | None = Field(
        default=None,
        description="Monitored metric",
    )

    monitor_mode: Literal["min", "max"] | None = Field(
        default=None,
        description="Monitor optimization mode",
    )

    has_metadata: bool = Field(
        default=False,
        description="Whether checkpoint has .metadata.yaml file",
    )

    def to_display_option(self) -> str:
        """Generate formatted display string for UI dropdown.

        Format: "architecture · backbone [· exp_name] (epN • hmean X.XXX • timestamp)"

        Returns:
            Formatted display string
        """
        parts = [self.architecture, self.backbone]

        # Add experiment name if different from architecture
        if self.exp_name and self.exp_name not in parts:
            exp_short = self.exp_name[:20] + "..." if len(self.exp_name) > 23 else self.exp_name
            parts.append(exp_short)

        model_info = " · ".join(parts)

        # Training annotations
        annotations = [f"ep{self.epochs}"]

        if self.hmean is not None:
            annotations.append(f"hmean {self.hmean:.3f}")

        if self.precision is not None:
            annotations.append(f"prec {self.precision:.3f}")

        if self.recall is not None:
            annotations.append(f"rec {self.recall:.3f}")

        if self.created_timestamp:
            annotations.append(self.created_timestamp)

        return f"{model_info} ({' • '.join(annotations)})"

    model_config = {
        "arbitrary_types_allowed": True,  # For Path objects
    }


class CheckpointCatalog(BaseModel):
    """Complete checkpoint catalog with metadata.

    Attributes:
        entries: List of catalog entries
        total_count: Total number of checkpoints found
        metadata_available_count: Number of checkpoints with .metadata.yaml files
        catalog_build_time_seconds: Time taken to build catalog
        outputs_dir: Outputs directory scanned
    """

    entries: list[CheckpointCatalogEntry] = Field(
        default_factory=list,
        description="List of catalog entries",
    )

    total_count: int = Field(
        default=0,
        ge=0,
        description="Total number of checkpoints found",
    )

    metadata_available_count: int = Field(
        default=0,
        ge=0,
        description="Number of checkpoints with .metadata.yaml files",
    )

    catalog_build_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to build catalog",
    )

    outputs_dir: Path = Field(
        ...,
        description="Outputs directory scanned",
    )

    @property
    def metadata_coverage_percent(self) -> float:
        """Calculate percentage of checkpoints with metadata files.

        Returns:
            Percentage (0-100) of checkpoints with metadata
        """
        if self.total_count == 0:
            return 0.0
        return (self.metadata_available_count / self.total_count) * 100

    model_config = {
        "arbitrary_types_allowed": True,  # For Path objects
    }
