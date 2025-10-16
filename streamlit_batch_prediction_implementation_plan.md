# Streamlit Batch Prediction Implementation Plan

## Overview
This document outlines the implementation plan for adding batch prediction and submission file output capabilities to the Streamlit 'Real-time OCR Inference' app. The goal is to enable users to process multiple images and generate competition-compliant submission files directly from the Streamlit interface.

## Current Architecture Analysis

### Normal Workflow (runners/predict.py)
- **Framework**: PyTorch Lightning with batch processing
- **Data Processing**: DataLoader with image preprocessing and augmentation
- **Postprocessing**: `DBPostProcessor.represent()` with inverse transformation matrices
- **Output**: JSON via `SubmissionWriter` → CSV conversion
- **Coordinate Handling**: `remap_polygons_to_original()` in SubmissionWriter

### Streamlit App Workflow (ui/utils/inference/engine.py)
- **Framework**: Direct model loading with single-image processing
- **Data Processing**: PIL/cv2 preprocessing with EXIF handling
- **Postprocessing**: `decode_polygons_with_head()` → `get_polygons_from_maps()`
- **Output**: Dictionary with polygons as comma-separated strings
- **Coordinate Handling**: `remap_polygons()` with orientation correction

## Key Differences & Challenges

### 1. Processing Scale
- **Challenge**: Single image vs batch processing
- **Solution**: Extend `InferenceService` to handle multiple images with progress tracking

### 2. Coordinate System Consistency
- **Challenge**: Different remapping approaches (`remap_polygons` vs `remap_polygons_to_original`)
- **Solution**: Verify and align coordinate transformations

### 3. Output Format Compatibility
- **Challenge**: Dictionary format vs JSON structure
- **Solution**: Create submission writer integration for Streamlit app

### 4. UI Responsiveness
- **Challenge**: Long-running batch operations blocking UI
- **Solution**: Implement progress bars and background processing

## Implementation Plan

### Phase 1: Core Infrastructure (2-3 days)

#### 1.1 Extend InferenceService for Batch Processing
**File**: `ui/apps/inference/services/inference_runner.py`

**Changes**:
- Add `run_batch_prediction()` method
- Implement directory scanning for image files
- Add progress tracking with `st.progress()`
- Accumulate results across multiple images
- Handle errors gracefully for individual images

**Code Structure**:
```python
def run_batch_prediction(self, state: InferenceState, request: BatchPredictionRequest, hyperparams: dict[str, float]) -> None:
    # Scan directory for images
    # Process each image with existing single-image logic
    # Accumulate results
    # Generate submission output
```

#### 1.2 Create Batch Prediction Data Models
**File**: `ui/apps/inference/models/`
- Add `BatchPredictionRequest` model
- Define batch processing parameters
- Include output format specifications

#### 1.3 Implement Submission Output Generation
**File**: `ui/apps/inference/services/submission_writer.py` (new)

**Functionality**:
- Convert accumulated predictions to JSON format
- Save to timestamped file in submissions directory
- Optionally convert to CSV format
- Match normal workflow output structure

### Phase 2: UI Integration (1-2 days)

#### 2.1 Add Batch Mode to Sidebar
**File**: `ui/apps/inference/components/sidebar.py`

**Changes**:
- Add toggle between "Single Image" and "Batch Prediction" modes
- Directory selection component for batch input
- Output filename configuration
- Batch processing parameters (thresholds, etc.)

#### 2.2 Create Batch Results Display
**File**: `ui/apps/inference/components/results.py`

**Changes**:
- Add batch results visualization
- Show processing statistics (images processed, total polygons)
- Display output file location
- Add download links for generated files

#### 2.3 Progress Indicators
**File**: `ui/apps/inference/components/progress.py` (new)

**Functionality**:
- Real-time progress bars for batch processing
- Individual image status updates
- Error reporting for failed images
- Estimated time remaining

### Phase 3: Coordinate System Alignment (1 day)

#### 3.1 Coordinate Transformation Verification
**File**: `ui/utils/inference/engine.py`

**Changes**:
- Compare `remap_polygons()` vs `remap_polygons_to_original()` outputs
- Create test cases with known transformations
- Ensure identical results between pipelines

#### 3.2 EXIF Orientation Handling
**Verification**:
- Test with images of different EXIF orientations
- Verify coordinate remapping accuracy
- Compare with normal workflow results

### Phase 4: Testing & Validation (1-2 days)

#### 4.1 Unit Tests
**Files**: `tests/unit/test_batch_prediction.py` (new)

**Test Cases**:
- Single image processing consistency
- Batch processing with multiple images
- Error handling for corrupted images
- Coordinate transformation accuracy
- Output format validation

#### 4.2 Integration Tests
**Files**: `tests/integration/test_streamlit_batch.py` (new)

**Test Cases**:
- End-to-end batch processing
- UI component interactions
- File I/O operations
- Memory usage with large batches

#### 4.3 Coordinate Accuracy Validation
**Script**: `scripts/validate_coordinate_consistency.py` (new)

**Functionality**:
- Compare Streamlit batch output vs normal workflow output
- Generate validation reports
- Identify and fix discrepancies

### Phase 5: Documentation & Deployment (0.5 days)

#### 5.1 Update Documentation
**Files**:
- `docs/streamlit_batch_prediction.md`
- Update `ui_meta/` configuration files
- Add usage examples

#### 5.2 Configuration Updates
**Files**:
- `configs/ui/inference.yaml`
- Add batch prediction parameters
- Update UI metadata schemas

## Implementation Details

### File Structure Changes
```
ui/apps/inference/
├── components/
│   ├── progress.py          # NEW: Progress indicators
│   └── ...
├── models/
│   ├── batch_request.py     # NEW: Batch prediction models
│   └── ...
├── services/
│   ├── inference_runner.py  # MODIFIED: Add batch processing
│   ├── submission_writer.py # NEW: Output generation
│   └── ...
└── ...

tests/
├── unit/
│   └── test_batch_prediction.py  # NEW: Unit tests
└── integration/
    └── test_streamlit_batch.py   # NEW: Integration tests

scripts/
└── validate_coordinate_consistency.py  # NEW: Validation script
```

### Key Technical Decisions

#### 1. Batch Size & Memory Management
- Process images individually to avoid memory issues
- Use streaming approach for large directories
- Implement configurable batch sizes for future optimization

#### 2. Error Handling Strategy
- Continue processing on individual image failures
- Log errors without stopping batch operation
- Provide detailed error reports in UI

#### 3. Output Format Compatibility
- Generate JSON in same format as normal workflow
- Ensure CSV conversion compatibility
- Maintain backward compatibility with existing single-image mode

#### 4. UI State Management
- Extend `InferenceState` for batch operations
- Persist batch results across UI interactions
- Implement result caching for performance

### Risk Mitigation

#### 1. Coordinate Inconsistencies
- **Risk**: Different remapping logic produces different results
- **Mitigation**: Comprehensive testing with validation scripts
- **Fallback**: Allow users to choose processing method

#### 2. Performance Issues
- **Risk**: Large batches cause UI freezing
- **Mitigation**: Implement proper progress tracking and background processing
- **Fallback**: Process in smaller chunks with user confirmation

#### 3. Memory Usage
- **Risk**: Loading many images simultaneously
- **Mitigation**: Process images one-by-one with cleanup
- **Fallback**: Add memory monitoring and automatic batch size reduction

## Success Criteria

### Functional Requirements
- [ ] Process directories containing 100+ images
- [ ] Generate competition-compliant submission files
- [ ] Maintain coordinate accuracy matching normal workflow
- [ ] Provide real-time progress feedback
- [ ] Handle various image formats and orientations

### Non-Functional Requirements
- [ ] UI remains responsive during processing
- [ ] Memory usage scales with input size
- [ ] Error handling doesn't crash the application
- [ ] Output format matches competition specifications

### Quality Assurance
- [ ] Unit test coverage > 80% for new components
- [ ] Integration tests pass for all scenarios
- [ ] Coordinate validation shows < 1px average difference
- [ ] Performance benchmarks meet requirements

## Timeline & Milestones

### Week 1: Core Infrastructure
- Day 1-2: Extend InferenceService for batch processing
- Day 3: Implement submission output generation
- Day 4-5: Create data models and basic UI components

### Week 2: UI Integration & Testing
- Day 1-2: Complete UI components and progress indicators
- Day 3: Coordinate system alignment and verification
- Day 4-5: Comprehensive testing and validation

### Week 3: Polish & Documentation
- Day 1: Performance optimization and edge case handling
- Day 2: Documentation and configuration updates
- Day 3: Final testing and deployment preparation

## Dependencies & Prerequisites

### Code Dependencies
- Existing Streamlit app infrastructure
- OCR model and postprocessing pipeline
- File I/O and image processing libraries

### Testing Dependencies
- Sample image datasets with known ground truth
- Coordinate validation test cases
- Performance benchmarking tools

### Documentation Dependencies
- UI configuration schemas
- API documentation updates
- User guide updates

## Future Enhancements

### Phase 2 Features (Post-Implementation)
- Parallel processing for multiple images
- GPU memory optimization
- Advanced preprocessing options
- Result comparison tools
- Export to different formats (COCO, Pascal VOC, etc.)

### Integration Opportunities
- Connect with evaluation pipeline
- Add model comparison capabilities
- Implement result caching and history
- Support for custom preprocessing pipelines

---

## Quick Start Implementation Guide

For rapid prototyping, focus on these minimal changes:

1. **Extend InferenceService** with basic directory iteration
2. **Add progress bar** to existing UI
3. **Create simple JSON output** using existing SubmissionWriter patterns
4. **Test coordinate consistency** before full UI integration

This approach allows validating the core functionality before investing in full UI polish.
