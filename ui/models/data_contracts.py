"""Data contracts for evaluation results and predictions using Pydantic v2.

These models validate the data structures used in the evaluation viewer to prevent
datatype mismatches and ensure consistency across the Streamlit UI.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _EvaluationBase(BaseModel):
    """Common configuration for evaluation data models."""

    model_config = ConfigDict(validate_assignment=True)


class RawPredictionRow(_EvaluationBase):
    """Data contract for raw prediction data from CSV files.

    # AI_DOCS: RawPredictionRow
    # Validates basic fields from uploaded prediction CSV files
    # Ensures filename has valid image extension and polygons are properly formatted
    # Used as first validation step before calculating derived metrics
    """

    filename: str = Field(..., description="Name of the image file")
    polygons: str = Field(default="", description="Pipe-separated polygon coordinates as strings")

    @field_validator("filename")
    @classmethod
    def _validate_filename(cls, value: str) -> str:
        """Validate filename is not empty and has valid extension."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Filename cannot be empty")
        value = value.strip()
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        if not any(value.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Filename must have valid image extension: {valid_extensions}")
        return value

    @field_validator("polygons")
    @classmethod
    def _validate_polygons(cls, value: str) -> str:
        """Validate that polygons string is properly formatted."""
        if not isinstance(value, str):
            raise TypeError("Polygons must be a string")
        if value.strip().lower() == "nan":
            return ""  # Treat "nan" as empty
        if not value.strip():
            return ""  # Allow empty polygons
        # Basic validation that it's pipe-separated coordinate strings
        for polygon_str in value.split("|"):
            if not polygon_str.strip():
                continue
            coords = polygon_str.split()
            if len(coords) < 8 or len(coords) % 2 != 0:
                raise ValueError(f"Invalid polygon format: {polygon_str}. Must have even number of coordinates >= 8")
            try:
                [float(coord) for coord in coords if coord.strip()]
            except ValueError as exc:
                raise ValueError(f"Polygon coordinates must be numeric: {polygon_str}") from exc
        return value


class PredictionRow(_EvaluationBase):
    """Data contract for a single prediction row from CSV.

    # AI_DOCS: PredictionRow
    # Validates complete prediction data including derived metrics
    # Ensures consistency between raw polygons and calculated fields
    # Used throughout evaluation pipeline for type-safe data access
    """

    filename: str = Field(..., description="Name of the image file")
    polygons: str = Field(default="", description="Pipe-separated polygon coordinates as strings")
    prediction_count: int = Field(default=0, description="Number of predictions in this row")
    total_area: float = Field(default=0.0, description="Total area of all polygons")
    avg_confidence: float = Field(default=0.0, description="Average confidence score")
    aspect_ratio: float = Field(default=1.0, description="Aspect ratio placeholder")

    @field_validator("filename")
    @classmethod
    def _validate_filename(cls, value: str) -> str:
        """Validate filename is not empty and has valid extension."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Filename cannot be empty")
        value = value.strip()
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        if not any(value.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Filename must have valid image extension: {valid_extensions}")
        return value

    @field_validator("polygons")
    @classmethod
    def _validate_polygons(cls, value: str) -> str:
        """Validate that polygons string is properly formatted."""
        if not isinstance(value, str):
            raise TypeError("Polygons must be a string")
        if value.strip().lower() == "nan":
            return ""  # Treat "nan" as empty
        if not value.strip():
            return ""  # Allow empty polygons
        # Basic validation that it's pipe-separated coordinate strings
        for polygon_str in value.split("|"):
            if not polygon_str.strip():
                continue
            coords = polygon_str.split()
            if len(coords) < 8 or len(coords) % 2 != 0:
                raise ValueError(f"Invalid polygon format: {polygon_str}. Must have even number of coordinates >= 8")
            try:
                [float(coord) for coord in coords if coord.strip()]
            except ValueError as exc:
                raise ValueError(f"Polygon coordinates must be numeric: {polygon_str}") from exc
        return value

    @field_validator("prediction_count")
    @classmethod
    def _validate_prediction_count(cls, value: int) -> int:
        """Validate prediction count is non-negative."""
        if not isinstance(value, int) or value < 0:
            raise ValueError("Prediction count must be a non-negative integer")
        return value

    @field_validator("total_area", "avg_confidence", "aspect_ratio")
    @classmethod
    def _validate_numeric_fields(cls, value: float) -> float:
        """Validate numeric fields are finite."""
        if not isinstance(value, int | float) or not (value >= 0):
            raise ValueError("Numeric fields must be non-negative numbers")
        return float(value)

    @model_validator(mode="after")
    def _validate_consistency(self) -> PredictionRow:
        """Validate that derived fields are consistent with polygons."""
        if self.polygons and self.polygons.strip():
            actual_count = len([p for p in self.polygons.split("|") if p.strip()])
            if self.prediction_count != actual_count:
                raise ValueError(f"prediction_count ({self.prediction_count}) doesn't match polygon count ({actual_count})")
        elif self.prediction_count != 0:
            raise ValueError("prediction_count must be 0 when polygons is empty")
        return self


class EvaluationMetrics(_EvaluationBase):
    """Data contract for evaluation metrics.

    # AI_DOCS: EvaluationMetrics
    # Validates aggregated metrics for model performance evaluation
    # Provides type-safe access to prediction statistics
    # Used for model comparison and performance reporting
    """

    total_predictions: int = Field(default=0, description="Total number of predictions across all images")
    avg_predictions: float = Field(default=0.0, description="Average predictions per image")
    images_with_predictions: int = Field(default=0, description="Number of images with at least one prediction")
    empty_predictions: int = Field(default=0, description="Number of images with no predictions")

    @field_validator("total_predictions", "images_with_predictions", "empty_predictions")
    @classmethod
    def _validate_counts(cls, value: int) -> int:
        """Validate count fields are non-negative integers."""
        if not isinstance(value, int) or value < 0:
            raise ValueError("Count fields must be non-negative integers")
        return value

    @field_validator("avg_predictions")
    @classmethod
    def _validate_avg_predictions(cls, value: float) -> float:
        """Validate average predictions is non-negative."""
        if not isinstance(value, int | float) or value < 0:
            raise ValueError("Average predictions must be non-negative")
        return float(value)


class DatasetStatistics(_EvaluationBase):
    """Data contract for comprehensive dataset statistics.

    # AI_DOCS: DatasetStatistics
    # Validates detailed statistics for entire evaluation datasets
    # Provides comprehensive metrics for dataset analysis
    # Used for dataset overview and quality assessment
    """

    total_images: int = Field(default=0, description="Total number of images in dataset")
    total_predictions: int = Field(default=0, description="Total number of predictions across all images")
    avg_predictions_per_image: float = Field(default=0.0, description="Average predictions per image")
    max_predictions_per_image: int = Field(default=0, description="Maximum predictions in a single image")
    total_area: float = Field(default=0.0, description="Total area of all predictions")
    avg_area_per_image: float = Field(default=0.0, description="Average area per image")
    images_with_predictions: int = Field(default=0, description="Number of images with predictions")
    empty_predictions: int = Field(default=0, description="Number of images with no predictions")

    @field_validator("total_images", "total_predictions", "max_predictions_per_image", "images_with_predictions", "empty_predictions")
    @classmethod
    def _validate_counts(cls, value: int) -> int:
        """Validate count fields are non-negative integers."""
        if not isinstance(value, int) or value < 0:
            raise ValueError("Count fields must be non-negative integers")
        return value

    @field_validator("avg_predictions_per_image", "total_area", "avg_area_per_image")
    @classmethod
    def _validate_numeric_fields(cls, value: float) -> float:
        """Validate numeric fields are non-negative."""
        if not isinstance(value, int | float) or value < 0:
            raise ValueError("Numeric fields must be non-negative")
        return float(value)

    @model_validator(mode="after")
    def _validate_consistency(self) -> DatasetStatistics:
        """Validate statistical consistency."""
        if self.total_images > 0:
            if self.images_with_predictions + self.empty_predictions != self.total_images:
                raise ValueError("Images with predictions + empty predictions must equal total images")
        return self


class ModelComparisonResult(_EvaluationBase):
    """Data contract for model comparison results.

    # AI_DOCS: ModelComparisonResult
    # Validates comparison metrics between two models
    # Ensures difference calculations are mathematically correct
    # Used for side-by-side model performance analysis
    """

    filename: str = Field(..., description="Image filename")
    pred_diff: int = Field(default=0, description="Difference in prediction count (B - A)")
    area_diff: float = Field(default=0.0, description="Difference in total area (B - A)")
    conf_diff: float = Field(default=0.0, description="Difference in confidence (B - A)")
    pred_a: int = Field(default=0, description="Prediction count for model A")
    pred_b: int = Field(default=0, description="Prediction count for model B")
    area_a: float = Field(default=0.0, description="Total area for model A")
    area_b: float = Field(default=0.0, description="Total area for model B")
    conf_a: float = Field(default=0.0, description="Confidence for model A")
    conf_b: float = Field(default=0.0, description="Confidence for model B")
    abs_pred_diff: int = Field(default=0, description="Absolute prediction difference")
    abs_area_diff: float = Field(default=0.0, description="Absolute area difference")
    abs_conf_diff: float = Field(default=0.0, description="Absolute confidence difference")

    @field_validator("filename")
    @classmethod
    def _validate_filename(cls, value: str) -> str:
        """Validate filename is not empty."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Filename cannot be empty")
        return value.strip()

    @field_validator("pred_a", "pred_b", "abs_pred_diff")
    @classmethod
    def _validate_prediction_counts(cls, value: int) -> int:
        """Validate prediction counts are non-negative."""
        if not isinstance(value, int) or value < 0:
            raise ValueError("Prediction counts must be non-negative integers")
        return value

    @field_validator("area_a", "area_b", "area_diff", "abs_area_diff", "conf_a", "conf_b", "conf_diff", "abs_conf_diff")
    @classmethod
    def _validate_numeric_fields(cls, value: float) -> float:
        """Validate numeric fields are finite."""
        if not isinstance(value, int | float):
            raise ValueError("Numeric fields must be numbers")
        return float(value)

    @model_validator(mode="after")
    def _validate_differences(self) -> ModelComparisonResult:
        """Validate that differences are calculated correctly."""
        if self.pred_diff != self.pred_b - self.pred_a:
            raise ValueError("pred_diff must equal pred_b - pred_a")
        if abs(self.pred_diff) != self.abs_pred_diff:
            raise ValueError("abs_pred_diff must equal abs(pred_diff)")
        if abs(self.area_diff) != self.abs_area_diff:
            raise ValueError("abs_area_diff must equal abs(area_diff)")
        if abs(self.conf_diff) != self.abs_conf_diff:
            raise ValueError("abs_conf_diff must equal abs(conf_diff)")
        return self
