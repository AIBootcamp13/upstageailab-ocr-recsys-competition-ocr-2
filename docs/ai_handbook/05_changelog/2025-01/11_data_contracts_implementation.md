# 2025-01-11: Data Contracts Implementation for Inference Pipeline

## Summary
Implemented comprehensive data validation using Pydantic v2 models throughout the Streamlit inference pipeline to prevent datatype mismatches and ensure data integrity. This addresses recurring bugs from inconsistent data structures and reduces unnecessary AI compute usage from failed inferences.

## Data Contracts
- **Predictions**: Validates OCR prediction data with polygon coordinates, detected texts, and confidence scores
- **PreprocessingInfo**: Validates preprocessing metadata and results from docTR operations
- **InferenceResult**: Validates complete inference results including success status, images, predictions, and preprocessing info
- **InferenceRequest**: Converted existing dataclass to Pydantic model for request validation

## Implementation Details
- Created `ui/apps/inference/models/data_contracts.py` with Pydantic v2 models
- Converted existing dataclasses in `config.py` and `ui_events.py` to Pydantic BaseModel classes
- Updated `inference_runner.py` to validate requests and results at API boundaries
- Modified UI components (`results.py`) to work with typed model attributes instead of dict access
- Updated state management to use `list[InferenceResult]` instead of `list[dict[str, Any]]`

## Validation Rules
- **Predictions**: Polygon strings must be properly formatted, confidence scores between 0.0-1.0, texts/confidences arrays must have matching lengths
- **Images**: Must be numpy arrays with shape (H, W, 3) and uint8/float32 dtypes
- **InferenceResult**: Success/error fields must be consistent (cannot have error when success=True)
- **PreprocessingInfo**: Mode must be "docTR:on" or "docTR:off"

## Usage Examples
```python
# Creating validated predictions
predictions = Predictions(
    polygons="0,0,10,0,10,10,0,10|20,20,30,20,30,30,20,30",
    texts=["Text 1", "Text 2"],
    confidences=[0.95, 0.87]
)

# Inference results with validation
result = InferenceResult(
    filename="document.jpg",
    success=True,
    image=image_array,
    predictions=predictions,
    preprocessing=PreprocessingInfo(enabled=True, doctr_available=True)
)
```

## Testing
- All models compile and import successfully
- Validation correctly rejects invalid data (mismatched array lengths, invalid confidence scores, malformed polygons)
- Existing UI functionality preserved with improved type safety
- Runtime validation catches data contract violations before they cause downstream errors

## Related Changes
- `ui/apps/inference/models/data_contracts.py` (new)
- `ui/apps/inference/models/config.py` (converted to Pydantic)
- `ui/apps/inference/models/ui_events.py` (converted to Pydantic)
- `ui/apps/inference/services/inference_runner.py` (added validation)
- `ui/apps/inference/state.py` (updated types)
- `ui/apps/inference/components/results.py` (updated to use model attributes)

## Breaking Changes
- Inference results are now strongly typed `InferenceResult` objects instead of loose dictionaries
- UI components must access result attributes directly instead of using `.get()` dict access
- State persistence may need migration for existing saved results (handled gracefully with fallbacks)
