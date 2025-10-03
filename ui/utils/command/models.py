"""
Command Parameter Models

Data models for command parameters used in the OCR project.
"""

from dataclasses import dataclass


@dataclass
class CommandParams:
    """Base data model for command parameters."""

    exp_name: str | None = None
    checkpoint_path: str | None = None


@dataclass
class TrainCommandParams(CommandParams):
    """Parameters specific to training commands."""

    encoder: str | None = None
    decoder: str | None = None
    head: str | None = None
    loss: str | None = None
    optimizer: str | None = None
    learning_rate: float | None = 0.001
    weight_decay: float | None = 0.0
    batch_size: int | None = 16
    max_epochs: int | None = 10
    accumulate_grad_batches: int | None = 1
    gradient_clip_val: float | None = 0.0
    precision: str = "32"
    seed: int | None = 42
    wandb: bool = False
    resume: str | None = None


@dataclass
class TestCommandParams(CommandParams):
    """Parameters specific to testing commands."""

    # Currently uses minimal parameters, similar to predict
    pass


@dataclass
class PredictCommandParams(CommandParams):
    """Parameters specific to prediction commands."""

    minified_json: bool = False
