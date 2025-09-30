from __future__ import annotations

"""Streamlit visualization components for the OCR evaluation UI."""

from .advanced import display_advanced_analysis, export_results
from .comparison import (
    display_model_comparison_stats,
    display_model_differences,
    display_side_by_side_comparison,
    display_visual_comparison,
)
from .helpers import draw_predictions_on_image, parse_polygon_string, polygon_points
from .overview import (
    display_dataset_overview,
    display_prediction_analysis,
    display_statistical_analysis,
    display_statistical_summary,
    render_low_confidence_analysis,
)
from .viewer import display_image_grid, display_image_viewer, display_image_with_predictions

__all__ = [
    "display_advanced_analysis",
    "display_dataset_overview",
    "display_image_grid",
    "display_image_viewer",
    "display_image_with_predictions",
    "display_model_comparison_stats",
    "display_model_differences",
    "display_prediction_analysis",
    "display_side_by_side_comparison",
    "display_statistical_analysis",
    "display_statistical_summary",
    "display_visual_comparison",
    "draw_predictions_on_image",
    "export_results",
    "parse_polygon_string",
    "polygon_points",
    "render_low_confidence_analysis",
]
