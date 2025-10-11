"""Data contracts and models for OCR evaluation data validation using Pydantic v2.

These models validate the data structures used in the evaluation viewer to prevent
datatype mismatches and ensure consistency across the Streamlit UI.
"""

# Re-export models from ui.models for backward compatibility
from ui.models import (
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
