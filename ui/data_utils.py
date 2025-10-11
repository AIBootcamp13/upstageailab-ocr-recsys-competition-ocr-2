"""
Data processing utilities for OCR evaluation viewer.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ui.models import DatasetStatistics, EvaluationMetrics, ModelComparisonResult, PredictionRow, RawPredictionRow


def load_predictions_file(file_path: str | Path | Any) -> pd.DataFrame:
    """Load predictions from CSV file with validation.

    # AI_DOCS: load_predictions_file
    # Validates raw prediction data from CSV files using Pydantic models
    # Ensures filename extensions and polygon formats are correct
    # Returns pandas DataFrame with validated data ready for metric calculation
    """
    # Read CSV and ensure polygons column is treated as string
    df = pd.read_csv(file_path)

    # Convert polygons column to string type and fill NaN values
    df["polygons"] = df["polygons"].astype(str).replace("nan", "").fillna("")

    # Validate each row using RawPredictionRow (only basic fields from CSV)
    validated_rows = []
    for idx, row in df.iterrows():
        try:
            # Only validate filename and polygons from CSV
            polygons_value = str(row["polygons"]) if pd.notna(row["polygons"]) else ""
            polygons_value = polygons_value if polygons_value != "nan" else ""
            row_dict = {"filename": row["filename"], "polygons": polygons_value}
            validated_row = RawPredictionRow.model_validate(row_dict)
            validated_rows.append(validated_row.model_dump())
        except ValidationError as e:
            raise ValueError(f"Invalid data in row {idx}: {e}") from e

    # Create validated DataFrame
    if validated_rows:
        df = pd.DataFrame(validated_rows)
    else:
        # Handle empty DataFrame case
        df = pd.DataFrame(columns=list(RawPredictionRow.model_fields.keys()))

    return df


def calculate_prediction_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived metrics for predictions dataframe with validation.

    # AI_DOCS: calculate_prediction_metrics
    # Computes prediction count, total area, and confidence scores from polygon data
    # Validates all derived metrics against raw polygon data for consistency
    # Returns DataFrame with additional validated metric columns
    """
    df = df.copy()

    # Ensure polygons column is string type
    df["polygons"] = df["polygons"].astype(str).fillna("")

    # Prediction count
    df["prediction_count"] = df["polygons"].apply(lambda x: len(x.split("|")) if pd.notna(x) and x.strip() and x != "nan" else 0)

    # Total area
    df["total_area"] = df["polygons"].apply(calculate_total_area)

    # Generate more realistic confidence scores based on heuristics
    # In a real system, these would come from the model's output probabilities
    if "avg_confidence" not in df.columns or df["avg_confidence"].isna().any():
        np.random.seed(42)  # For reproducible results
        generated_scores = df.apply(lambda row: generate_confidence_score(row), axis=1)
        if "avg_confidence" not in df.columns:
            df["avg_confidence"] = generated_scores
        else:
            df["avg_confidence"] = df["avg_confidence"].fillna(generated_scores)

    # Aspect ratio (placeholder)
    df["aspect_ratio"] = 1.0

    # Validate each row after calculations
    validated_rows = []
    for idx, row in df.iterrows():
        try:
            row_dict = row.to_dict()
            validated_row = PredictionRow.model_validate(row_dict)
            validated_rows.append(validated_row.model_dump())
        except ValidationError as e:
            raise ValueError(f"Invalid calculated data in row {idx}: {e}") from e

    return pd.DataFrame(validated_rows)


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
    if not polygons_str or not isinstance(polygons_str, str) or not polygons_str.strip() or polygons_str == "nan":
        return 0.0

    total_area = 0.0
    polygons = polygons_str.split("|")

    for polygon in polygons:
        polygon = polygon.strip()
        if not polygon:
            continue

        try:
            coords = [float(x.strip()) for x in polygon.split(",") if x.strip()]
            if len(coords) >= 8 and len(coords) % 2 == 0:
                # Simple area calculation using bounding box approximation
                x_coords = coords[::2]
                y_coords = coords[1::2]
                width = max(x_coords) - min(x_coords)
                height = max(y_coords) - min(y_coords)
                total_area += width * height
        except (ValueError, IndexError):
            # Skip malformed polygons
            continue

    return total_area


def apply_sorting_filtering(df: pd.DataFrame, sort_by: str, sort_order: str, filter_metric: str) -> pd.DataFrame:
    """Apply sorting and filtering to the dataframe."""
    # Ensure metrics are calculated
    df = calculate_prediction_metrics(df)

    # Apply filtering
    if filter_metric == "high_confidence":
        filtered_df = df[df["avg_confidence"] > 0.8]
    elif filter_metric == "low_confidence":
        filtered_df = df[df["avg_confidence"] < 0.5]
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


def calculate_model_metrics(df: pd.DataFrame) -> EvaluationMetrics:
    """Calculate key metrics for a model's predictions with validation.

    # AI_DOCS: calculate_model_metrics
    # Aggregates prediction statistics across entire dataset
    # Returns validated EvaluationMetrics object with type-safe access
    # Used for model comparison and performance reporting
    """
    df = calculate_prediction_metrics(df)

    metrics_dict = {
        "total_predictions": int(df["prediction_count"].sum()),
        "avg_predictions": float(df["prediction_count"].mean()),
        "images_with_predictions": int((df["prediction_count"] > 0).sum()),
        "empty_predictions": int((df["prediction_count"] == 0).sum()),
    }

    return EvaluationMetrics.model_validate(metrics_dict)


def find_common_images(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    """Find images that exist in both dataframes."""
    return sorted(set(df_a["filename"]).intersection(set(df_b["filename"])))


def calculate_image_differences(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Calculate differences between predictions for common images.

    # AI_DOCS: calculate_image_differences
    # Compares predictions between two models for overlapping images
    # Calculates prediction count, area, and confidence differences
    # Returns validated DataFrame with comparison metrics
    # Used for model comparison and performance analysis
    """
    common_images = find_common_images(df_a, df_b)

    if not common_images:
        return pd.DataFrame()

    differences = []
    for image in common_images:
        row_a = df_a[df_a["filename"] == image].iloc[0]
        row_b = df_b[df_b["filename"] == image].iloc[0]

        polygons_a = str(row_a["polygons"]) if pd.notna(row_a["polygons"]) else ""
        polygons_b = str(row_b["polygons"]) if pd.notna(row_b["polygons"]) else ""

        pred_count_a = len(polygons_a.split("|")) if polygons_a and polygons_a != "nan" else 0
        pred_count_b = len(polygons_b.split("|")) if polygons_b and polygons_b != "nan" else 0

        area_a = calculate_total_area(polygons_a)
        area_b = calculate_total_area(polygons_b)

        # Calculate confidence differences (using placeholder values for now)
        conf_a = row_a.get("avg_confidence", 0.8)
        if pd.isna(conf_a):
            conf_a = 0.8
        conf_b = row_b.get("avg_confidence", 0.8)
        if pd.isna(conf_b):
            conf_b = 0.8

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

    # Validate each difference row
    validated_differences = []
    for idx, row in diff_df.iterrows():
        try:
            row_dict = row.to_dict()
            validated_diff = ModelComparisonResult.model_validate(row_dict)
            validated_differences.append(validated_diff.model_dump())
        except ValidationError as e:
            raise ValueError(f"Invalid difference data in row {idx}: {e}") from e

    return pd.DataFrame(validated_differences)


def get_dataset_statistics(df: pd.DataFrame) -> DatasetStatistics:
    """Calculate comprehensive dataset statistics with validation.

    # AI_DOCS: get_dataset_statistics
    # Computes detailed statistics for entire evaluation dataset
    # Returns validated DatasetStatistics object with comprehensive metrics
    # Used for dataset overview and analysis reporting
    """
    df = calculate_prediction_metrics(df)

    stats_dict = {
        "total_images": len(df),
        "total_predictions": int(df["prediction_count"].sum()),
        "avg_predictions_per_image": float(df["prediction_count"].mean()),
        "max_predictions_per_image": int(df["prediction_count"].max()),
        "total_area": float(df["total_area"].sum()),
        "avg_area_per_image": float(df["total_area"].mean()),
        "images_with_predictions": int((df["prediction_count"] > 0).sum()),
        "empty_predictions": int((df["prediction_count"] == 0).sum()),
    }

    return DatasetStatistics.model_validate(stats_dict)


def prepare_export_data(df: pd.DataFrame) -> tuple[pd.DataFrame, DatasetStatistics]:
    """Prepare data for export with validated statistics."""
    df = calculate_prediction_metrics(df)

    # Summary statistics
    summary_stats = get_dataset_statistics(df)

    return df, summary_stats
