"""
Data processing utilities for OCR evaluation viewer.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_predictions_file(file_path: str | Path | Any) -> pd.DataFrame:
    """Load predictions from CSV file."""
    return pd.read_csv(file_path)


def calculate_prediction_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived metrics for predictions dataframe."""
    df = df.copy()

    # Prediction count
    df["prediction_count"] = df["polygons"].apply(lambda x: len(x.split("|")) if pd.notna(x) and x.strip() else 0)

    # Total area
    df["total_area"] = df["polygons"].apply(calculate_total_area)

    # Generate more realistic confidence scores based on heuristics
    # In a real system, these would come from the model's output probabilities
    np.random.seed(42)  # For reproducible results
    df["avg_confidence"] = df.apply(lambda row: generate_confidence_score(row), axis=1)

    # Aspect ratio (placeholder)
    df["aspect_ratio"] = 1.0

    return df


def generate_confidence_score(row) -> float:
    """Generate a realistic confidence score based on prediction characteristics."""
    base_confidence = 0.7  # Base confidence level

    # Factors that might affect confidence:
    # - Number of predictions (more predictions might indicate lower confidence)
    # - Total area (very small or very large areas might be less confident)
    # - Polygon complexity (simpler polygons might be more confident)

    pred_count = row["prediction_count"]
    total_area = row["total_area"]

    # Penalty for too many or too few predictions
    if pred_count == 0:
        return 0.0
    elif pred_count > 10:
        base_confidence -= 0.1
    elif pred_count < 3:
        base_confidence -= 0.05

    # Penalty for extreme areas
    if total_area > 100000:  # Very large area
        base_confidence -= 0.1
    elif total_area < 1000:  # Very small area
        base_confidence -= 0.05

    # Add some random variation to make it more realistic
    variation = np.random.normal(0, 0.1)
    confidence = base_confidence + variation

    # Clamp to [0, 1] range
    return max(0.0, min(1.0, confidence))


def calculate_total_area(polygons_str: str) -> float:
    """Calculate total area of all polygons in a prediction."""
    if not polygons_str or not polygons_str.strip():
        return 0.0

    total_area = 0.0
    polygons = polygons_str.split("|")

    for polygon in polygons:
        coords = [float(x) for x in polygon.split()]
        if len(coords) >= 8:
            # Simple area calculation using bounding box approximation
            x_coords = coords[::2]
            y_coords = coords[1::2]
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            total_area += width * height

    return total_area


def apply_sorting_filtering(df: pd.DataFrame, sort_by: str, sort_order: str, filter_metric: str) -> pd.DataFrame:
    """Apply sorting and filtering to the dataframe."""
    # Ensure metrics are calculated
    df = calculate_prediction_metrics(df)

    # Apply filtering
    if filter_metric == "high_confidence":
        filtered_df = df[df["avg_confidence"] > 0.8]
    elif filter_metric == "low_confidence":
        filtered_df = df[df["avg_confidence"] <= 0.8]
    elif filter_metric == "many_predictions":
        filtered_df = df[df["prediction_count"] > df["prediction_count"].median()]
    elif filter_metric == "few_predictions":
        filtered_df = df[df["prediction_count"] <= df["prediction_count"].median()]
    else:
        filtered_df = df

    # Apply sorting
    ascending = sort_order == "ascending"
    if sort_by in filtered_df.columns:
        filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)

    return filtered_df


def calculate_model_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Calculate key metrics for a model's predictions."""
    df = calculate_prediction_metrics(df)

    return {
        "total_predictions": df["prediction_count"].sum(),
        "avg_predictions": df["prediction_count"].mean(),
        "images_with_predictions": (df["prediction_count"] > 0).sum(),
        "empty_predictions": (df["prediction_count"] == 0).sum(),
    }


def find_common_images(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    """Find images that exist in both dataframes."""
    return sorted(set(df_a["filename"]).intersection(set(df_b["filename"])))


def calculate_image_differences(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Calculate differences between predictions for common images."""
    common_images = find_common_images(df_a, df_b)

    if not common_images:
        return pd.DataFrame()

    differences = []
    for image in common_images:
        row_a = df_a[df_a["filename"] == image].iloc[0]
        row_b = df_b[df_b["filename"] == image].iloc[0]

        pred_count_a = len(row_a["polygons"].split("|")) if pd.notna(row_a["polygons"]) else 0
        pred_count_b = len(row_b["polygons"].split("|")) if pd.notna(row_b["polygons"]) else 0

        area_a = calculate_total_area(row_a["polygons"] if pd.notna(row_a["polygons"]) else "")
        area_b = calculate_total_area(row_b["polygons"] if pd.notna(row_b["polygons"]) else "")

        # Calculate confidence differences (using placeholder values for now)
        conf_a = row_a.get("avg_confidence", 0.8)
        conf_b = row_b.get("avg_confidence", 0.8)

        differences.append(
            {
                "filename": image,
                "pred_diff": pred_count_b - pred_count_a,
                "area_diff": area_b - area_a,
                "conf_diff": conf_b - conf_a,
                "pred_a": pred_count_a,
                "pred_b": pred_count_b,
                "area_a": area_a,
                "area_b": area_b,
                "conf_a": conf_a,
                "conf_b": conf_b,
            }
        )

    diff_df = pd.DataFrame(differences)

    # Add absolute differences for sorting
    diff_df["abs_pred_diff"] = diff_df["pred_diff"].abs()
    diff_df["abs_area_diff"] = diff_df["area_diff"].abs()
    diff_df["abs_conf_diff"] = diff_df["conf_diff"].abs()

    return diff_df


def get_dataset_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate comprehensive dataset statistics."""
    df = calculate_prediction_metrics(df)

    return {
        "total_images": len(df),
        "total_predictions": df["prediction_count"].sum(),
        "avg_predictions_per_image": df["prediction_count"].mean(),
        "max_predictions_per_image": df["prediction_count"].max(),
        "total_area": df["total_area"].sum(),
        "avg_area_per_image": df["total_area"].mean(),
        "images_with_predictions": (df["prediction_count"] > 0).sum(),
        "empty_predictions": (df["prediction_count"] == 0).sum(),
    }


def prepare_export_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare data for export."""
    df = calculate_prediction_metrics(df)

    # Summary statistics
    summary_stats = get_dataset_statistics(df)

    return df, summary_stats
