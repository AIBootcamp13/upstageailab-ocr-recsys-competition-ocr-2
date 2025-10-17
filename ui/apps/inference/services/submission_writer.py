"""Submission file writer service for batch predictions.

This module handles conversion of batch prediction results to competition-compliant
submission formats (JSON and CSV). Follows data contract patterns from:
- docs/ai_handbook/07_planning/plans/pydantic-data-validation/SESSION_HANDOVER.md
- docs/ai_handbook/07_planning/assessments/streamlit_batch_prediction_implementation_plan.md

Key Features:
- Converts InferenceResult objects to JSON format matching runners/predict.py output
- Generates CSV files compatible with competition submission format
- Validates output structure before writing
- Provides detailed logging for debugging
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..models.data_contracts import InferenceResult

LOGGER = logging.getLogger(__name__)


class SubmissionEntry(BaseModel):
    """Single entry in the submission file.

    Matches the format expected by the competition evaluation system.
    CSV format: filename,polygons[,confidence]
    """

    filename: str = Field(..., description="Name of the image file")
    polygons: str = Field(..., description="Pipe-separated polygon coordinates")
    confidence: float | None = Field(None, description="Optional average confidence score")

    def to_dict(self) -> dict[str, str | float]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "filename": self.filename,
            "polygons": self.polygons,
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result


class SubmissionWriter:
    """Service for writing batch prediction results to submission files.

    Handles conversion from InferenceResult objects to competition-compliant
    JSON and CSV formats.
    """

    @staticmethod
    def write_json(
        results: list[InferenceResult],
        output_path: Path,
        include_confidence: bool = False,
    ) -> None:
        """Write batch results to JSON file.

        Args:
            results: List of inference results to write
            output_path: Path where JSON file should be saved
            include_confidence: Whether to include average confidence scores

        Raises:
            ValueError: If results list is empty
            IOError: If file cannot be written
        """
        if not results:
            raise ValueError("Cannot write empty results to JSON")

        # Convert results to submission entries
        entries = []
        for result in results:
            confidence = None
            if include_confidence and result.success and result.predictions.confidences:
                # Calculate average confidence
                confidence = sum(result.predictions.confidences) / len(result.predictions.confidences)

            if result.success:
                entry = SubmissionEntry(
                    filename=result.filename,
                    polygons=result.predictions.polygons,
                    confidence=confidence,
                )
                entries.append(entry.to_dict())
            else:
                # Include failed results with empty polygons
                LOGGER.warning(f"Including failed result for {result.filename} with empty polygons")
                entry = SubmissionEntry(
                    filename=result.filename,
                    polygons="",
                    confidence=None,
                )
                entries.append(entry.to_dict())

        # Write to JSON file
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
            LOGGER.info(f"Successfully wrote {len(entries)} entries to {output_path}")
        except Exception as exc:
            LOGGER.exception(f"Failed to write JSON to {output_path}")
            raise OSError(f"Failed to write JSON file: {exc}") from exc

    @staticmethod
    def write_csv(
        results: list[InferenceResult],
        output_path: Path,
        include_confidence: bool = False,
    ) -> None:
        """Write batch results to CSV file.

        CSV format matches competition requirements:
        - Column 1: filename (not image_name!)
        - Column 2: polygons (pipe-separated coordinates)
        - Column 3 (optional): confidence (average confidence score)

        Args:
            results: List of inference results to write
            output_path: Path where CSV file should be saved
            include_confidence: Whether to include average confidence scores as third column

        Raises:
            ValueError: If results list is empty
            IOError: If file cannot be written
        """
        if not results:
            raise ValueError("Cannot write empty results to CSV")

        # Convert results to submission entries
        entries = []
        for result in results:
            confidence = None
            if include_confidence and result.success and result.predictions.confidences:
                # Calculate average confidence
                confidence = sum(result.predictions.confidences) / len(result.predictions.confidences)

            if result.success:
                entry = SubmissionEntry(
                    filename=result.filename,
                    polygons=result.predictions.polygons,
                    confidence=confidence,
                )
                entries.append(entry)
            else:
                # Include failed results with empty polygons
                LOGGER.warning(f"Including failed result for {result.filename} with empty polygons")
                entry = SubmissionEntry(
                    filename=result.filename,
                    polygons="",
                    confidence=None,
                )
                entries.append(entry)

        # Write to CSV file
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                # Write header - use 'filename' not 'image_name' per competition format
                if include_confidence:
                    writer.writerow(["filename", "polygons", "confidence"])
                else:
                    writer.writerow(["filename", "polygons"])
                # Write data rows
                for entry in entries:
                    if include_confidence:
                        writer.writerow([entry.filename, entry.polygons, entry.confidence if entry.confidence is not None else ""])
                    else:
                        writer.writerow([entry.filename, entry.polygons])
            LOGGER.info(f"Successfully wrote {len(entries)} entries to {output_path}")
        except Exception as exc:
            LOGGER.exception(f"Failed to write CSV to {output_path}")
            raise OSError(f"Failed to write CSV file: {exc}") from exc

    @staticmethod
    def write_batch_results(
        results: list[InferenceResult],
        json_path: Path | None = None,
        csv_path: Path | None = None,
        include_confidence: bool = False,
    ) -> dict[str, Path]:
        """Write batch results to both JSON and CSV formats.

        Args:
            results: List of inference results to write
            json_path: Optional path for JSON output (if None, JSON not written)
            csv_path: Optional path for CSV output (if None, CSV not written)
            include_confidence: Whether to include average confidence scores

        Returns:
            Dictionary mapping format names to output paths

        Raises:
            ValueError: If results list is empty or both paths are None
        """
        if not results:
            raise ValueError("Cannot write empty results")

        if json_path is None and csv_path is None:
            raise ValueError("At least one output path (JSON or CSV) must be provided")

        written_files: dict[str, Path] = {}

        # Write JSON if requested
        if json_path is not None:
            try:
                SubmissionWriter.write_json(results, json_path, include_confidence=include_confidence)
                written_files["json"] = json_path
            except Exception as exc:
                LOGGER.error(f"Failed to write JSON: {exc}")
                raise

        # Write CSV if requested
        if csv_path is not None:
            try:
                SubmissionWriter.write_csv(results, csv_path, include_confidence=include_confidence)
                written_files["csv"] = csv_path
            except Exception as exc:
                LOGGER.error(f"Failed to write CSV: {exc}")
                raise

        return written_files

    @staticmethod
    def generate_summary_stats(results: list[InferenceResult]) -> dict[str, Any]:
        """Generate summary statistics for batch results.

        Args:
            results: List of inference results

        Returns:
            Dictionary containing summary statistics
        """
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful

        total_polygons = 0
        total_texts = 0
        for result in results:
            if result.success:
                # Count polygons (pipe-separated)
                polygons_str = result.predictions.polygons.strip()
                if polygons_str:
                    polygon_count = len(polygons_str.split("|"))
                    total_polygons += polygon_count
                # Count texts
                total_texts += len(result.predictions.texts)

        return {
            "total_images": total,
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful / total * 100):.1f}%" if total > 0 else "0%",
            "total_polygons_detected": total_polygons,
            "total_texts_detected": total_texts,
            "avg_polygons_per_image": f"{(total_polygons / successful):.1f}" if successful > 0 else "0",
        }
