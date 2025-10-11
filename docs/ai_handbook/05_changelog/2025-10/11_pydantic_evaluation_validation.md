# 2025-10-11: Pydantic Data Validation for Evaluation Viewer

## Summary
Implemented comprehensive Pydantic v2 data validation for the OCR Evaluation Viewer Streamlit application to prevent type-checking errors and ensure data integrity throughout the evaluation pipeline. This addresses frequent runtime type errors by introducing robust validation at all data processing stages.

## Data Contracts

### New Pydantic Models Introduced
- **`RawPredictionRow`**: Validates basic CSV data (filename, polygons) from uploaded files
- **`PredictionRow`**: Validates complete prediction data with derived metrics
- **`EvaluationMetrics`**: Validates model performance metrics (total_predictions, avg_predictions, etc.)
- **`DatasetStatistics`**: Validates comprehensive dataset statistics
- **`ModelComparisonResult`**: Validates model comparison results with difference calculations

### Validation Rules Implemented
- **Filename validation**: Ensures valid image extensions (.jpg, .jpeg, .png, .bmp, .tiff, .tif)
- **Polygon validation**: Checks coordinate format, numeric values, and proper structure
- **Consistency validation**: Ensures derived fields (prediction_count, total_area) match raw polygon data
- **Type safety**: Prevents runtime type errors with strict field validation

### Integration Points
- `load_predictions_file()`: Validates raw CSV data on upload
- `calculate_prediction_metrics()`: Validates derived metrics after calculation
- `calculate_model_metrics()`: Returns validated EvaluationMetrics object
- `get_dataset_statistics()`: Returns validated DatasetStatistics object
- `calculate_image_differences()`: Validates comparison results

## Implementation Details

### Architecture Decisions
- **Moved models to `ui/models/`**: Centralized data contracts to avoid circular imports
- **Two-stage validation**: Raw data validation (RawPredictionRow) followed by complete validation (PredictionRow)
- **Backward compatibility**: Existing UI components continue receiving pandas DataFrames with validated data
- **Error handling**: Clear validation error messages with row-specific context

### Key Components Added/Modified
- `ui/models/data_contracts.py`: New Pydantic v2 data models
- `ui/models/__init__.py`: Model exports and package structure
- `ui/data_utils.py`: Updated functions to use Pydantic validation
- `ui/visualization/comparison.py`: Updated to handle EvaluationMetrics objects
- `ui/__init__.py`: Fixed visualization module aliasing

### Dependencies Introduced
- Enhanced Pydantic v2 usage (already present in project)
- No new external dependencies required

## Usage Examples

### Loading Validated Predictions
```python
from ui.data_utils import load_predictions_file, calculate_prediction_metrics

# Load and validate CSV data
df = load_predictions_file("predictions.csv")  # Validates filename/polygons

# Calculate and validate derived metrics
df_with_metrics = calculate_prediction_metrics(df)  # Validates all fields
```

### Using Validated Metrics
```python
from ui.data_utils import calculate_model_metrics

metrics = calculate_model_metrics(df)
print(f"Total predictions: {metrics.total_predictions}")
print(f"Average per image: {metrics.avg_predictions:.1f}")
```

### Model Comparison with Validation
```python
from ui.data_utils import calculate_image_differences

comparison_df = calculate_image_differences(df_a, df_b)  # Validates all comparison results
```

## Testing

### Test Coverage Achieved
- **Unit tests**: Individual model validation
- **Integration tests**: Complete data processing pipeline
- **Edge case testing**: Empty data, malformed polygons, NaN values
- **Error handling**: Validation error propagation and user feedback

### Key Test Scenarios
- Valid CSV loading with proper polygon data
- Invalid filename extensions rejected
- Malformed polygon coordinates caught
- Empty predictions handled correctly
- Derived metrics consistency validated
- Model comparison calculations verified

### Validation of Data Contracts
- All models pass Pydantic validation with test data
- Error messages provide clear feedback for invalid data
- Type safety maintained throughout processing pipeline

## Related Changes

### Files Modified
- `ui/models/data_contracts.py` (new): Pydantic data models
- `ui/models/__init__.py` (new): Model exports
- `ui/data_utils.py`: Updated validation logic
- `ui/visualization/comparison.py`: Updated to use validated objects
- `ui/__init__.py`: Fixed module aliasing
- `ui/evaluation/models/` (modified): Re-export models for compatibility

### Documentation Updated
- `docs/ai_handbook/05_changelog/2025-10/11_pydantic_evaluation_validation.md` (this file)
- `docs/CHANGELOG.md`: Added feature entry

### Breaking Changes
- None: All existing APIs maintain the same interface
- Internal functions now return validated Pydantic objects instead of dicts
- UI components continue receiving pandas DataFrames as before
