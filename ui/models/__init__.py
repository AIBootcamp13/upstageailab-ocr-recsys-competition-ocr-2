"""Data contracts and models for OCR evaluation data validation using Pydantic v2.

These models validate the data structures used throughout the UI to prevent
datatype mismatches and ensure consistency across the Streamlit applications.
"""

from .data_contracts import (
    DatasetStatistics,
    EvaluationMetrics,
    ModelComparisonResult,
    PredictionRow,
    RawPredictionRow,
)

__all__ = [
    "PredictionRow",
    "RawPredictionRow",
    "EvaluationMetrics",
    "DatasetStatistics",
    "ModelComparisonResult",
]
