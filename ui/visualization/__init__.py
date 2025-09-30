from __future__ import annotations

"""Streamlit visualization components for the OCR evaluation UI."""

from ui._visualization import (
    display_advanced_analysis,
    display_dataset_overview,
    display_image_grid,
    display_image_viewer,
    display_image_with_predictions,
    display_model_comparison_stats,
    display_model_differences,
    display_prediction_analysis,
    display_side_by_side_comparison,
    display_statistical_analysis,
    display_statistical_summary,
    display_visual_comparison,
    draw_predictions_on_image,
    export_results,
    parse_polygon_string,
    polygon_points,
    render_low_confidence_analysis,
)

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
