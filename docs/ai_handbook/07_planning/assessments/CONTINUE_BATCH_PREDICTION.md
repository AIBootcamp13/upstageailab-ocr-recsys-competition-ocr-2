# Continue: Streamlit Batch Prediction - Phase 2

## 📋 Quick Start

Phase 1 (Core Infrastructure) is **complete**. All backend services are implemented with Pydantic v2 validation.

## 🎯 Your Mission

Implement Phase 2: UI Integration for the Streamlit Batch Prediction feature.

## 📖 Required Reading

1. **Session Handover** (READ FIRST):
   - `@docs/ai_handbook/07_planning/assessments/streamlit_batch_prediction_SESSION_HANDOVER.md`

2. **Living Blueprint**:
   - `@docs/ai_handbook/07_planning/assessments/streamlit_batch_prediction_implementation_plan_bluepint.md`

3. **Pydantic Validation Reference**:
   - `@docs/ai_handbook/07_planning/plans/pydantic-data-validation/SESSION_HANDOVER.md`

## 🚀 Next Action (Task 2.1)

**Add Batch Mode Toggle to Sidebar**

**File to modify:** `ui/apps/inference/components/sidebar.py`

**What to implement:**
1. Add mode selector: "Single Image" vs "Batch Prediction"
2. In Batch mode, show:
   - Directory path input (validate it exists)
   - Output directory input
   - Filename prefix input
   - JSON/CSV output checkboxes
   - "Run Batch Prediction" button
3. In Single Image mode: keep existing UI unchanged

**Use these imports:**
```python
from ..models.batch_request import (
    BatchPredictionRequest,
    BatchOutputConfig,
    BatchHyperparameters,
)
```

**Pattern to follow:**
- Use `st.radio()` for mode selection
- Conditional rendering based on mode
- Store batch config in session state
- Validate paths with Pydantic on input

## ✅ What's Already Done

- ✅ BatchPredictionRequest with full validation
- ✅ BatchOutputConfig with filesystem-safe validation
- ✅ BatchHyperparameters with range validation
- ✅ InferenceService.run_batch_prediction() method
- ✅ SubmissionWriter for JSON/CSV output
- ✅ Progress tracking and error handling

## 🎯 Success Criteria for This Phase

Before moving to Phase 3:
1. Batch mode toggle works in sidebar
2. Directory input validates in real-time
3. Batch prediction runs end-to-end
4. Results display shows statistics
5. Output files downloadable from UI

---

**Start with Task 2.1 now. Follow the living blueprint checklist.**
