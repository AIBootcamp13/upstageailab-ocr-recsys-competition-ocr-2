"""Batch prediction request data models using Pydantic v2.

This module defines data contracts for batch OCR processing requests.
Follows Pydantic validation patterns from:
- docs/ai_handbook/07_planning/plans/pydantic-data-validation/SESSION_HANDOVER.md
- docs/ai_handbook/07_planning/plans/pydantic-data-validation/preprocessing-module-refactor-implementation-plan.md

Data Validation Standards:
- All input parameters are validated at runtime
- Path validation ensures directory exists and is accessible
- Output configuration validated for file system compatibility
- Hyperparameters validated for reasonable ranges
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BatchOutputConfig(BaseModel):
    """Configuration for batch prediction output files.

    Defines where and how batch prediction results should be saved.
    """

    model_config = ConfigDict(validate_assignment=True)

    output_dir: str = Field(
        default="submissions",
        description="Directory to save submission files (relative or absolute path)",
    )
    filename_prefix: str = Field(
        default="batch_prediction",
        description="Prefix for output files (timestamp will be appended)",
    )
    save_json: bool = Field(
        default=True,
        description="Whether to save results in JSON format",
    )
    save_csv: bool = Field(
        default=True,
        description="Whether to save results in CSV format",
    )
    include_confidence: bool = Field(
        default=False,
        description="Whether to include average confidence scores in output",
    )

    @field_validator("filename_prefix")
    @classmethod
    def _validate_filename_prefix(cls, value: str) -> str:
        """Validate filename prefix contains only safe characters."""
        if not value or not value.strip():
            raise ValueError("Filename prefix cannot be empty")

        # Check for filesystem-unsafe characters
        unsafe_chars = set('<>:"/\\|?*')
        if any(char in unsafe_chars for char in value):
            raise ValueError(f"Filename prefix contains unsafe characters. Avoid: {', '.join(unsafe_chars)}")

        return value.strip()

    @model_validator(mode="after")
    def _validate_output_formats(self) -> BatchOutputConfig:
        """Ensure at least one output format is enabled."""
        if not self.save_json and not self.save_csv:
            raise ValueError("At least one output format (JSON or CSV) must be enabled")
        return self


class BatchHyperparameters(BaseModel):
    """Hyperparameters for batch OCR inference.

    Validates inference parameters are within reasonable ranges.
    """

    model_config = ConfigDict(validate_assignment=True)

    binarization_thresh: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Threshold for binary segmentation",
    )
    box_thresh: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for bounding box detection",
    )
    max_candidates: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum number of text region candidates",
    )
    min_detection_size: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Minimum size of detected text regions",
    )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary format expected by inference engine."""
        return {
            "binarization_thresh": self.binarization_thresh,
            "box_thresh": self.box_thresh,
            "max_candidates": float(self.max_candidates),
            "min_detection_size": float(self.min_detection_size),
        }


class BatchPredictionRequest(BaseModel):
    """Data contract for batch OCR prediction requests.

    Validates all parameters required for batch processing including:
    - Input directory containing images
    - Model checkpoint path
    - Preprocessing configuration
    - Output configuration
    - Inference hyperparameters

    Example:
        >>> request = BatchPredictionRequest(
        ...     input_dir="/path/to/images",
        ...     model_path="/path/to/checkpoint.ckpt",
        ...     use_preprocessing=True,
        ...     output_config=BatchOutputConfig(),
        ...     hyperparameters=BatchHyperparameters(),
        ... )
    """

    model_config = ConfigDict(validate_assignment=True)

    input_dir: str = Field(
        ...,
        description="Directory containing images to process",
    )
    model_path: str = Field(
        ...,
        description="Path to model checkpoint file",
    )
    config_path: str | None = Field(
        None,
        description="Path to model config file",
    )
    use_preprocessing: bool = Field(
        default=False,
        description="Whether to apply document preprocessing",
    )
    output_config: BatchOutputConfig = Field(
        default_factory=BatchOutputConfig,
        description="Configuration for output files",
    )
    hyperparameters: BatchHyperparameters = Field(
        default_factory=BatchHyperparameters,
        description="Inference hyperparameters",
    )

    # Supported image extensions for batch processing
    supported_extensions: tuple[str, ...] = Field(
        default=(".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"),
        description="Supported image file extensions",
    )

    @field_validator("input_dir")
    @classmethod
    def _validate_input_dir(cls, value: str) -> str:
        """Validate input directory exists and is accessible."""
        if not value or not value.strip():
            raise ValueError("Input directory cannot be empty")

        path = Path(value)
        if not path.exists():
            raise ValueError(f"Input directory does not exist: {value}")

        if not path.is_dir():
            raise ValueError(f"Input path is not a directory: {value}")

        return str(path.resolve())

    @field_validator("model_path")
    @classmethod
    def _validate_model_path(cls, value: str) -> str:
        """Validate model checkpoint path exists."""
        if not value or not value.strip():
            raise ValueError("Model path cannot be empty")

        path = Path(value)
        if not path.exists():
            raise ValueError(f"Model checkpoint does not exist: {value}")

        if not path.is_file():
            raise ValueError(f"Model path is not a file: {value}")

        return str(path.resolve())

    def get_image_files(self) -> list[Path]:
        """Scan input directory and return list of valid image files.

        Returns:
            List of Path objects for valid image files, sorted by name.

        Raises:
            ValueError: If no valid images found in directory.
        """
        input_path = Path(self.input_dir)
        image_files: list[Path] = []

        for ext in self.supported_extensions:
            # Case-insensitive matching
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        # Remove duplicates and sort
        image_files = sorted(set(image_files), key=lambda p: p.name.lower())

        if not image_files:
            raise ValueError(
                f"No valid image files found in {self.input_dir}. Supported extensions: {', '.join(self.supported_extensions)}"
            )

        return image_files

    def get_output_path(self, suffix: str = ".json") -> Path:
        """Generate timestamped output file path.

        Args:
            suffix: File extension (e.g., '.json', '.csv')

        Returns:
            Path object for output file with timestamp.
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_config.filename_prefix}_{timestamp}{suffix}"

        output_dir = Path(self.output_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir / filename
