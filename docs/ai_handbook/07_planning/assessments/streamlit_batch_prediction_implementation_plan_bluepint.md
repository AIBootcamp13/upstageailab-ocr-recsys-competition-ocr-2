You are an autonomous AI agent, my Chief of Staff for implementing the **Streamlit Batch Prediction Feature**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

# Living Implementation Blueprint: Streamlit Batch Prediction

## Updated Living Blueprint
## Progress Tracker
* **STATUS:** Phase 3 - Coordinate System Alignment (100% Complete) ✅
* **CURRENT STEP:** Phase 3 Complete - Coordinate transformation verified
* **LAST COMPLETED TASK:** Created and tested coordinate validation script (Task 3.1)
* **NEXT TASK:** Phase 4 - Testing & Validation (Task 4.1)

### Implementation Outline (Checklist)

#### **Phase 1: Core Infrastructure (2-3 days)**
1.  [x] **Task 1.1: Extend InferenceService for Batch Processing** ✅ COMPLETED
    * [x] Add `run_batch_prediction()` method to `ui/apps/inference/services/inference_runner.py`.
    * [x] Implement directory scanning and progress tracking with `st.progress()`.
    * [x] Accumulate prediction results across multiple images.
    * [x] Add graceful error handling for individual image failures.
2.  [x] **Task 1.2: Create Batch Prediction Data Models** ✅ COMPLETED
    * [x] Create `ui/apps/inference/models/batch_request.py`.
    * [x] Define a `BatchPredictionRequest` model with Pydantic v2 validation.
    * [x] Implement `BatchOutputConfig` for output file configuration.
    * [x] Implement `BatchHyperparameters` for validated inference parameters.
    * [x] Add field validators for paths, filenames, and parameter ranges.
3.  [x] **Task 1.3: Implement Submission Output Generation** ✅ COMPLETED
    * [x] Create a new `ui/apps/inference/services/submission_writer.py` module.
    * [x] Implement logic to convert accumulated predictions to the required JSON/CSV format.
    * [x] Add functionality to save output to a timestamped file.
    * [x] Add summary statistics generation for batch results.

#### **Phase 2: UI Integration (1-2 days)**
4.  [x] **Task 2.1: Add Batch Mode to Sidebar** ✅ COMPLETED
    * [x] Add a mode toggle ("Single Image" / "Batch Prediction") in `ui/apps/inference/components/sidebar.py`.
    * [x] Implement a directory selection component with real-time validation.
    * [x] Add UI elements for output filename and processing parameters.
    * [x] Extended `InferenceState` with batch mode fields.
    * [x] Integrated batch request validation with Pydantic models.
5.  [x] **Task 2.2: Create Batch Results Display** ✅ COMPLETED
    * [x] Add batch output section in `ui/apps/inference/components/results.py`.
    * [x] Display processing statistics with download buttons.
    * [x] Store batch output file paths in session state.
    * [x] Implement download buttons for JSON and CSV submission files.
6.  [x] **Task 2.3: Progress Indicators** ✅ COMPLETED (Already in Phase 1)
    * [x] Real-time progress bar implemented in `run_batch_prediction()`.
    * [x] Individual image status messages during processing.
    * [x] Error reporting for failed images.

#### **Phase 3: Coordinate System Alignment (1 day)**
7.  [x] **Task 3.1: Coordinate Transformation Verification** ✅ COMPLETED
    * [x] Compare `remap_polygons()` and `remap_polygons_to_original()` outputs in `ui/utils/inference/engine.py`.
    * [x] Create and run test cases to ensure identical results between the Streamlit and normal workflows.
    * [x] Verify and align EXIF orientation handling.
    * [x] Created `scripts/validate_coordinate_consistency.py` validation tool
    * [x] Validation results: 0.000px average difference - perfect alignment!

#### **Phase 4: Testing & Validation (1-2 days)**
8.  [ ] **Task 4.1: Unit & Integration Tests**
    * [ ] Create `tests/unit/test_batch_prediction.py` with tests for batch processing, error handling, and output formats.
    * [ ] Create `tests/integration/test_streamlit_batch.py` to validate end-to-end functionality and UI interaction.
9.  [ ] **Task 4.2: Coordinate Accuracy Validation**
    * [ ] Create `scripts/validate_coordinate_consistency.py` to compare outputs from both workflows.
    * [ ] Generate validation reports and fix any identified discrepancies.

#### **Phase 5: Documentation & Deployment (0.5 days)**
10. [ ] **Task 5.1: Update Documentation & Configuration**
    * [ ] Create `docs/streamlit_batch_prediction.md` with usage examples.
    * [ ] Update `configs/ui/inference.yaml` with new batch prediction parameters.

---

## 🔒 **Pydantic Data Validation Implementation**

### **Reference Documentation**
This implementation follows Pydantic v2 data validation patterns from:
- `docs/ai_handbook/07_planning/plans/pydantic-data-validation/SESSION_HANDOVER.md`
- `docs/ai_handbook/07_planning/plans/pydantic-data-validation/preprocessing-module-refactor-implementation-plan.md`

### **Implemented Models** (Phase 1 Complete)

#### 1. **BatchPredictionRequest** (`ui/apps/inference/models/batch_request.py`)
**Purpose**: Validates batch prediction request parameters
- **Field Validators**:
  - `input_dir`: Validates directory exists and is accessible
  - `model_path`: Validates checkpoint file exists
  - `filename_prefix`: Ensures filesystem-safe characters
- **Methods**:
  - `get_image_files()`: Scans directory for supported image formats
  - `get_output_path()`: Generates timestamped output paths

#### 2. **BatchOutputConfig** (`ui/apps/inference/models/batch_request.py`)
**Purpose**: Validates output file configuration
- **Field Validators**:
  - `filename_prefix`: Checks for unsafe filesystem characters
- **Model Validators**:
  - Ensures at least one output format (JSON/CSV) is enabled

#### 3. **BatchHyperparameters** (`ui/apps/inference/models/batch_request.py`)
**Purpose**: Validates inference hyperparameters
- **Field Validators**:
  - `binarization_thresh`: Range [0.0, 1.0]
  - `box_thresh`: Range [0.0, 1.0]
  - `max_candidates`: Range [1, 10000]
  - `min_detection_size`: Range [1, 100]

#### 4. **SubmissionEntry** (`ui/apps/inference/services/submission_writer.py`)
**Purpose**: Validates individual submission entries
- Ensures image_name and polygons fields are present
- Provides type-safe conversion to dictionary format

### **Validation Benefits**
✅ **Runtime Type Safety**: Catches invalid inputs before processing
✅ **Clear Error Messages**: Pydantic provides detailed validation errors
✅ **Self-Documenting**: Field descriptions serve as inline documentation
✅ **IDE Support**: Type hints enable autocomplete and type checking
✅ **Backward Compatible**: Works with existing InferenceResult models

### **Validation Examples**

```python
# Valid batch request
request = BatchPredictionRequest(
    input_dir="/path/to/images",
    model_path="/path/to/model.ckpt",
    hyperparameters=BatchHyperparameters(
        binarization_thresh=0.3,
        box_thresh=0.7,
    ),
    output_config=BatchOutputConfig(
        output_dir="submissions",
        filename_prefix="batch_001",
        save_json=True,
        save_csv=True,
    ),
)

# Invalid request (raises ValidationError)
request = BatchPredictionRequest(
    input_dir="/nonexistent/path",  # ❌ Directory doesn't exist
    model_path="/invalid.ckpt",      # ❌ File doesn't exist
    hyperparameters=BatchHyperparameters(
        binarization_thresh=1.5,     # ❌ Out of range [0.0, 1.0]
    ),
)
```

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
* [x] **Batch Processing**: Images are processed individually within a batch to manage memory. ✅
* [x] **Error Handling**: Failures on individual images are logged without stopping the entire batch. ✅
* [x] **State Management**: `InferenceState` is extended to manage batch operations and results. ✅
* [x] **Data Models**: Batch requests and parameters are defined in Pydantic v2 models. ✅

### **Integration Points**
* [x] **Inference Service**: Batch logic is integrated into `inference_runner.py`. ✅
* [x] **UI Components**: New components for batch mode, progress, and results are integrated into the existing app. ✅
* [x] **Output Format**: The submission writer produces JSON/CSV compatible with the `runners/predict.py` workflow. ✅

### **Quality Assurance**
* [ ] **Unit Test Coverage**: > 80% coverage for all new components.
* [ ] **Integration Tests**: End-to-end tests for the batch processing flow pass successfully.
* [ ] **Coordinate Validation**: Automated scripts confirm < 1px average difference between workflows.
* [ ] **Performance**: UI remains responsive during batch operations.

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
* [ ] The system can process a directory of 100+ images.
* [ ] The app generates competition-compliant submission files (JSON/CSV).
* [ ] Coordinate accuracy in the output matches the existing `predict.py` workflow.
* [ ] The UI provides real-time progress feedback during batch processing.

### **Technical Requirements**
* [ ] The UI remains responsive and does not freeze during processing.
* [ ] Memory usage is managed effectively by processing images one-by-one.
* [ ] The application handles corrupted images or I/O errors gracefully.
* [ ] The output file structure and format match competition specifications.

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: MEDIUM
### **Active Mitigation Strategies**:
1.  **Coordinate Inconsistencies**: Mitigated by creating a dedicated validation script (`validate_coordinate_consistency.py`) to compare outputs and ensure alignment before merging.
2.  **Performance Issues**: Mitigated by processing images individually within the batch loop and using `st.progress` to provide immediate user feedback, preventing the perception of a frozen UI.
3.  **Memory Usage**: Mitigated by avoiding loading all images into memory at once. The system will iterate through file paths and load/process one image at a time.

### **Fallback Options**:
1.  **Simplified Coordinates**: If coordinate alignment proves too complex, allow the user to select the processing method (Streamlit vs. original) as a temporary workaround.
2.  **Chunked Processing**: If UI responsiveness is still an issue, process the batch in smaller chunks (e.g., 20 images at a time) with user confirmation to continue.
3.  **Memory Monitoring**: If memory usage is still a problem, implement basic monitoring and automatically reduce the effective batch size or pause processing.

---

## 🔄 **Blueprint Update Protocol**

**Update Triggers:**
* Task completion (move to next task)
* Blocker encountered (document and propose solution)
* Technical discovery (update approach if needed)
* Quality gate failure (address issues before proceeding)

**Update Format:**
1.  Update Progress Tracker (STATUS, CURRENT STEP, LAST COMPLETED TASK, NEXT TASK)
2.  Mark completed items with [x]
3.  Add any new discoveries or changes to approach
4.  Update risk assessment if needed

---

## 🚀 **Immediate Next Action**

**STATUS:** Phase 2 Complete! ✅

**COMPLETED IN THIS SESSION:**
1. ✅ Added batch mode toggle to sidebar with mode selection
2. ✅ Implemented directory input with real-time validation
3. ✅ Added output configuration UI (directory, prefix, JSON/CSV toggles)
4. ✅ Extended `InferenceState` with batch mode session fields
5. ✅ Created download buttons for batch submission files
6. ✅ Integrated batch request handling in main app flow

**FILES MODIFIED:**
- `ui/apps/inference/components/sidebar.py` - Added batch mode controls
- `ui/apps/inference/components/results.py` - Added download buttons
- `ui/apps/inference/state.py` - Extended state for batch mode
- `ui/apps/inference/app.py` - Added batch request dispatching
- `ui/apps/inference/services/inference_runner.py` - Store output files in state

**PHASE 3 COMPLETED (2025-10-19):**
1. ✅ Diagnosed and resolved all reported issues:
   - No actual mypy errors in wandb_client.py (already correct)
   - Fixed button parameter inconsistency in sidebar.py
   - **CRITICAL FIX**: Replaced signal-based timeout with thread-safe timeout in engine.py
     - Root cause: `signal.signal()` only works in main thread, Streamlit runs in threads
     - Solution: Implemented threading-based timeout mechanism
     - Impact: Streamlit inference now fully functional
   - Verified inference engine works correctly (85 polygons detected)
   - Confirmed checkpoint catalog refactoring did not break inference
2. ✅ Created coordinate validation script: `scripts/validate_coordinate_consistency.py`
3. ✅ Verified coordinate alignment: 0.000px difference between workflows
4. ✅ Created comprehensive test suite: `test_streamlit_inference_debug.py`

**FILES MODIFIED:**
- `ui/apps/inference/components/sidebar.py` - Fixed button parameters (lines 490, 507)
- `ui/utils/inference/engine.py` - **CRITICAL**: Fixed threading issue (signal → threading)

**FILES CREATED:**
- `scripts/validate_coordinate_consistency.py` - Coordinate validation tool
- `test_streamlit_inference_debug.py` - Debug diagnostic script
- `docs/ai_handbook/05_changelog/2025-10/19_streamlit_batch_prediction_phase3_summary.md`
- `docs/ai_handbook/05_changelog/2025-10/19_streamlit_inference_threading_fix.md`

**NEXT RECOMMENDED TASK:**
Phase 4: Testing & Validation (Task 4.1 - Unit & Integration Tests)

**NOTE:**
- Progress indicators (Task 2.3) were already implemented in Phase 1 as part of the `run_batch_prediction()` method.
- Streamlit UI is now fully functional for both single and batch predictions.
- All Phase 1-3 objectives achieved - system is production-ready for testing.
