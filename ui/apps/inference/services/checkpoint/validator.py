"""Schema validation for checkpoint metadata.

This module provides validation for checkpoint metadata against the schema,
including business logic validation beyond Pydantic's structural validation.

Features:
    - Individual metadata validation with business rules
    - Batch validation for directories with progress tracking
    - Validation reporting with statistics and error details
    - Integration with metadata loader for file-based validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from .metadata_loader import load_metadata
from .types import CheckpointMetadataV1

LOGGER = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of metadata validation operation.

    Attributes:
        checkpoint_path: Path to checkpoint file
        is_valid: Whether validation passed
        metadata: Loaded and validated metadata (if successful)
        error: Error message (if validation failed)
        error_type: Type of error (e.g., 'missing', 'invalid', 'business_rule')
    """

    checkpoint_path: Path
    is_valid: bool
    metadata: CheckpointMetadataV1 | None = None
    error: str | None = None
    error_type: str | None = None


@dataclass
class ValidationReport:
    """Aggregated validation report for batch operations.

    Attributes:
        total: Total number of checkpoints checked
        valid: Number of valid metadata files
        invalid: Number of invalid metadata files
        missing: Number of missing metadata files
        results: Detailed results for each checkpoint
        errors_by_type: Count of errors grouped by type
    """

    total: int = 0
    valid: int = 0
    invalid: int = 0
    missing: int = 0
    results: list[ValidationResult] = field(default_factory=list)
    errors_by_type: dict[str, int] = field(default_factory=dict)

    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result to the report.

        Args:
            result: Validation result to add
        """
        self.results.append(result)
        self.total += 1

        if result.is_valid:
            self.valid += 1
        elif result.error_type == "missing":
            self.missing += 1
        else:
            self.invalid += 1
            error_type = result.error_type or "unknown"
            self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

    def success_rate(self) -> float:
        """Calculate validation success rate.

        Returns:
            Success rate as percentage (0-100)
        """
        if self.total == 0:
            return 0.0
        return (self.valid / self.total) * 100

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Multi-line summary string
        """
        lines = [
            "=" * 60,
            "Validation Report",
            "=" * 60,
            f"Total checkpoints: {self.total}",
            f"Valid:             {self.valid} ({self.success_rate():.1f}%)",
            f"Invalid:           {self.invalid}",
            f"Missing metadata:  {self.missing}",
            "=" * 60,
        ]

        if self.errors_by_type:
            lines.append("Errors by type:")
            for error_type, count in sorted(self.errors_by_type.items()):
                lines.append(f"  {error_type}: {count}")
            lines.append("=" * 60)

        return "\n".join(lines)


class MetadataValidator:
    """Validator for checkpoint metadata with schema compatibility checks."""

    def __init__(self, schema_version: str = "1.0"):
        """Initialize validator with target schema version.

        Args:
            schema_version: Target schema version to validate against
        """
        self.schema_version = schema_version

    def validate_metadata(
        self,
        metadata: CheckpointMetadataV1,
    ) -> CheckpointMetadataV1:
        """Validate metadata against schema and business rules.

        Args:
            metadata: Metadata to validate

        Returns:
            Validated metadata

        Raises:
            ValidationError: If metadata is invalid
        """
        # Pydantic already validates structure on model creation

        # Additional business logic validation
        if metadata.metrics.hmean is None:
            msg = "hmean metric is required for catalog entry (per user requirements)"
            raise ValueError(msg)

        if metadata.metrics.precision is None:
            LOGGER.warning(
                "precision metric missing for %s (recommended per user requirements)",
                metadata.checkpoint_path,
            )

        if metadata.metrics.recall is None:
            LOGGER.warning(
                "recall metric missing for %s (recommended per user requirements)",
                metadata.checkpoint_path,
            )

        if metadata.training.epoch < 0:
            msg = "Epoch cannot be negative"
            raise ValueError(msg)

        # Validate schema version compatibility
        if metadata.schema_version != self.schema_version:
            LOGGER.warning(
                "Schema version mismatch: expected %s, got %s for %s",
                self.schema_version,
                metadata.schema_version,
                metadata.checkpoint_path,
            )

        return metadata

    def validate_batch(
        self,
        metadata_list: list[CheckpointMetadataV1],
    ) -> list[CheckpointMetadataV1]:
        """Validate multiple metadata entries.

        Args:
            metadata_list: List of metadata to validate

        Returns:
            List of validated metadata

        Note:
            Individual validation errors are logged but don't stop batch processing
        """
        validated = []

        for metadata in metadata_list:
            try:
                validated.append(self.validate_metadata(metadata))
            except (ValidationError, ValueError) as exc:
                LOGGER.error(
                    "Validation failed for %s: %s",
                    metadata.checkpoint_path,
                    exc,
                )

        return validated

    def validate_checkpoint_file(self, checkpoint_path: Path) -> ValidationResult:
        """Validate metadata for a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            ValidationResult with validation status and details
        """
        # Check if metadata file exists
        metadata = load_metadata(checkpoint_path)

        if metadata is None:
            return ValidationResult(
                checkpoint_path=checkpoint_path,
                is_valid=False,
                error="Metadata file not found or failed to load",
                error_type="missing",
            )

        # Validate metadata
        try:
            validated_metadata = self.validate_metadata(metadata)
            return ValidationResult(
                checkpoint_path=checkpoint_path,
                is_valid=True,
                metadata=validated_metadata,
            )

        except ValidationError as exc:
            return ValidationResult(
                checkpoint_path=checkpoint_path,
                is_valid=False,
                error=str(exc),
                error_type="schema_validation",
            )

        except ValueError as exc:
            return ValidationResult(
                checkpoint_path=checkpoint_path,
                is_valid=False,
                error=str(exc),
                error_type="business_rule",
            )

    def validate_directory(
        self,
        directory: Path,
        recursive: bool = True,
        verbose: bool = False,
    ) -> ValidationReport:
        """Validate all checkpoints in a directory.

        Args:
            directory: Directory to search for checkpoints
            recursive: Recursively search subdirectories
            verbose: Log progress for each checkpoint

        Returns:
            ValidationReport with aggregated results
        """
        if not directory.exists():
            LOGGER.error("Directory does not exist: %s", directory)
            return ValidationReport()

        # Find all checkpoint files
        if recursive:
            checkpoint_paths = list(directory.glob("**/*.ckpt"))
        else:
            checkpoint_paths = list(directory.glob("*.ckpt"))

        if not checkpoint_paths:
            LOGGER.warning("No checkpoint files found in: %s", directory)
            return ValidationReport()

        LOGGER.info("Found %d checkpoint files in: %s", len(checkpoint_paths), directory)

        # Validate each checkpoint
        report = ValidationReport()

        for ckpt_path in checkpoint_paths:
            result = self.validate_checkpoint_file(ckpt_path)
            report.add_result(result)

            if verbose:
                if result.is_valid:
                    LOGGER.info("✓ Valid: %s", ckpt_path.name)
                else:
                    LOGGER.warning("✗ %s: %s - %s", result.error_type, ckpt_path.name, result.error)

        return report

    def validate_checkpoint_list(
        self,
        checkpoint_paths: list[Path],
        verbose: bool = False,
    ) -> ValidationReport:
        """Validate a list of checkpoint files.

        Args:
            checkpoint_paths: List of checkpoint file paths
            verbose: Log progress for each checkpoint

        Returns:
            ValidationReport with aggregated results
        """
        report = ValidationReport()

        for ckpt_path in checkpoint_paths:
            result = self.validate_checkpoint_file(ckpt_path)
            report.add_result(result)

            if verbose:
                if result.is_valid:
                    LOGGER.info("✓ Valid: %s", ckpt_path)
                else:
                    LOGGER.warning("✗ %s: %s - %s", result.error_type, ckpt_path, result.error)

        return report
