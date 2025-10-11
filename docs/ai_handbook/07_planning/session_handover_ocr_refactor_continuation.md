# **OCR Lightning Module Refactor - Continuation Prompt**

**Session Start**: New AI Session
**Previous Work**: Completed refactor plan design with Pydantic validation integration
**Current Branch**: `08_refactor/ocr_pl` (created from `07_refactor/performance_debug2`)
**Date**: October 11, 2025

---

## **🎯 Mission Objective**

Execute the OCR Lightning Module refactor plan to:
- Break down monolithic `ocr_pl.py` (845 lines) into maintainable components
- Integrate Pydantic validation to prevent costly post-refactor bugs
- Maintain identical functionality while improving code organization

---

## **📋 Completed Work Summary**

### **✅ Refactor Plan Enhanced**
- **Merged Approaches**: Combined evaluation decoupling with modular extraction
- **Pydantic Integration**: Added comprehensive runtime validation models
- **Risk Assessment**: Clear risk levels and mitigation strategies
- **Executable Steps**: Complete code snippets and commands for each phase

### **✅ Key Deliverables Ready**
- **Refactor Plan**: `/docs/ai_handbook/07_planning/plans/refactor/ocr_lightning_module_refactor_plan.md`
- **Data Contracts**: `/docs/pipeline/data_contracts.md` (reference document)
- **Branch Created**: `08_refactor/ocr_pl` for isolated development

---

## **🚀 Current State & Next Steps**

### **Immediate Action Required**
Execute **Phase 1: Create Dedicated Evaluation Service** (HIGH RISK - 2-3 hours)

### **Phase 1 Objectives**
1. **Create Pydantic validation models** for data contracts
2. **Extract CLEvalEvaluator** from `ocr_pl.py` into `ocr/evaluation/evaluator.py`
3. **Integrate evaluator** into Lightning module with validation
4. **Test thoroughly** to ensure identical metrics

### **Critical Files to Create/Modify**
```
ocr/validation/models.py          # NEW: Pydantic models
ocr/evaluation/__init__.py        # NEW: Package init
ocr/evaluation/evaluator.py       # NEW: CLEvalEvaluator class
ocr/lightning_modules/ocr_pl.py   # MODIFY: Integrate evaluator
```

---

## **🔧 Phase 1 Implementation Guide**

### **Step 1.1: Create Pydantic Models** (30-45 min)
```bash
# Create validation package
mkdir -p /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ocr/validation

# Create models.py with complete Pydantic models from the plan
# Focus on: DatasetSample, TransformOutput, CollateOutput, ModelOutput, LightningStepOutput
```

### **Step 1.2: Create Evaluation Service** (45-60 min)
```bash
# Create evaluation package
mkdir -p /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ocr/evaluation

# Extract CLEvalEvaluator from ocr_pl.py lines 400-550
# Add Pydantic validation in update() method
# Ensure identical metric computation logic
```

### **Step 1.3: Integrate into Lightning Module** (30-45 min)
```bash
# Modify ocr/lightning_modules/ocr_pl.py:
# - Import CLEvalEvaluator
# - Replace OrderedDict step_outputs with evaluator instances
# - Update validation_step, test_step, and epoch_end methods
# - Add validation calls
```

### **Step 1.4: Comprehensive Testing** (45-60 min)
```bash
# Test data validation
python -c "from ocr.validation.models import DatasetSample; print('Models import successfully')"

# Test evaluator extraction
python -c "from ocr.evaluation import CLEvalEvaluator; print('Evaluator imports successfully')"

# Test integration
python -m pytest tests/unit/test_lightning_module.py -v

# Quick training validation
python runners/train.py trainer.fast_dev_run=true
```

---

## **⚠️ Critical Risk Mitigation**

### **HIGH RISK Areas (Phase 1)**
- **Metric Calculation**: Must produce identical results (±0.001 tolerance)
- **Data Flow**: Polygon shapes, tensor dimensions must be preserved
- **Performance**: No regression in evaluation speed

### **Validation Strategy**
- **Pre-Integration**: Unit test evaluator in isolation
- **Post-Integration**: Compare metrics with baseline (unmodified code)
- **Data Validation**: Pydantic catches shape errors immediately

### **Rollback Plan**
```bash
# If metrics differ >0.001
git checkout HEAD~1 -- ocr/lightning_modules/ocr_pl.py
git branch -D 08_refactor/ocr_pl  # Start over with fixes
```

---

## **📊 Success Criteria for Phase 1**

- [ ] **Pydantic models created** and import successfully
- [ ] **CLEvalEvaluator extracted** with identical logic
- [ ] **Lightning module integration** complete
- [ ] **All existing tests pass**
- [ ] **Training produces identical results** (±0.001 tolerance)
- [ ] **Evaluation metrics unchanged**
- [ ] **No performance regression** (<5% slowdown)
- [ ] **Data validation catches shape errors** immediately

---

## **🔄 Phase Progression**

After Phase 1 completion:
- **Phase 2**: Extract config/utils with validation (2-3 hours, LOW RISK)
- **Phase 3**: Extract processors/logging with validation (2-3 hours, LOW RISK)
- **Phase 4**: Final cleanup and documentation (1 hour, LOW RISK)

---

## **🛠️ Development Environment Setup**

### **Required Dependencies**
```bash
# Ensure Pydantic is available
pip install pydantic numpy torch

# Verify current environment
python -c "import pydantic, numpy, torch; print('Dependencies OK')"
```

### **Testing Commands**
```bash
# Quick validation after each step
python -c "from ocr.validation.models import *; print('✅ Models OK')"
python -c "from ocr.evaluation import CLEvalEvaluator; print('✅ Evaluator OK')"

# Full test suite
python -m pytest tests/unit/test_lightning_module.py -v

# Integration test
python runners/train.py trainer.fast_dev_run=true
```

---

## **📝 Implementation Notes**

### **Code Extraction Strategy**
- Copy logic first, then refactor
- Maintain all comments and edge cases
- Add validation at integration points
- Test after each major change

### **Data Contract Compliance**
- Reference `/docs/pipeline/data_contracts.md` for exact specifications
- Use Pydantic models to enforce contracts
- Validate inputs/outputs at component boundaries

### **Performance Considerations**
- Validation enabled in development
- Can be disabled in production: `model_config = {'validate_assignment': False}`
- Minimal overhead during data flow

---

## **🎯 Expected Outcomes**

### **Immediate Benefits (Phase 1)**
- **Bug Prevention**: Shape errors caught immediately, not after hours of training
- **Clear Errors**: Descriptive validation messages instead of cryptic tensor errors
- **Faster Iteration**: Refactor bugs fixed in seconds, not hours

### **Long-term Benefits**
- **Maintainable Code**: Smaller, focused modules (<400 lines each)
- **Self-Documenting**: Pydantic models as executable specifications
- **Reliable Refactors**: Validation prevents regression bugs

---

## **🚨 Emergency Contacts**

If issues arise:
1. **Metrics differ**: Compare with unmodified baseline immediately
2. **Import errors**: Check file paths and Python path
3. **Validation fails**: Review data contract specifications
4. **Performance issues**: Profile evaluation timing

**Rollback threshold**: If any metric differs by >0.001, rollback immediately.

---

**Ready to execute Phase 1. Begin with Pydantic model creation and proceed systematically through each step. Report progress after each major milestone.**

---

## **📋 Session Handover**

- Validators migrated to Pydantic v2 across models.py, added `_info_data` helper for cross-field context, and aliased `validator` to keep legacy references working. Orientation schema now accepts full EXIF set `{0…8}`, fixing the validation crash observed during the 1‑epoch dry run. Lint is clean (`ruff check`) and latest short training run confirms refactor stability with unchanged performance.
- Current focus areas: make an explicit note to rerun broader test suites (pytest/integration) and verify prediction artifacts for orientation handling; plan Phase 1 wrap-up steps (documentation, finalize evaluator hooks, ensure configs reference updated schemas). No outstanding errors, but git has staged changes (`git add .`).
- Suggested next steps:
  1. Run targeted pytest suite (`pytest tests/ocr` or agreed subset) to confirm the validator overhaul doesn't break downstream logic.
  2. Inspect saved predictions for a few samples with orientation ∈ {5,6,7,8} to double-check evaluator remapping.
  3. Draft or update Phase 1 documentation/checklist noting completion status and remaining deliverables.

## **🔄 Continuation Prompt**

"Continue Phase 1 verification by running the agreed pytest subset and inspecting sample predictions for orientations >4. Assume the validator refactor and orientation schema changes in models.py are already in place, lint is clean, and a 1‑epoch train run succeeded. Focus on closing out remaining validation or documentation tasks before moving to Phase 2."

## **📚 References**
Phase 2:
- `docs/ai_handbook/07_planning/plans/refactor/ocr_lightning_module_refactor_plan.md`
- #sym:## Phase 2: Extract Configuration and Utility Functions (Low Risk - 2-3 hours)
  - TOC, overview, and documentation references located at lines 1:68</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/ai_handbook/07_planning/session_handover_ocr_refactor_continuation.md
