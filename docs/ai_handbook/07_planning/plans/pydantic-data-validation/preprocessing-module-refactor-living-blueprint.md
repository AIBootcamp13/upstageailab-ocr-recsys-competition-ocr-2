# Preprocessing Module Refactor: Living Implementation Blueprint

## Overview

You are an autonomous AI software engineer executing a systematic refactor of the preprocessing module to address data type uncertainties, improve type safety, and reduce development friction. Your primary responsibility is to execute the "Living Refactor Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will run the `[COMMAND]` provided to work towards that goal.
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

## 1. Current State (Based on Assessment)

- **Project:** Preprocessing module refactor on branch `09_refactor/ocr_base`.
- **Blueprint:** "Preprocessing Module Pydantic Validation Refactor".
- **Current Position:** Phase 1 implementation in progress - ImageShape model successfully created.
- **Risk Classification:**
  - **High Risk**: `metadata.py`, `config.py`, `pipeline.py` - Core data structures with loose typing
  - **Medium Risk**: `detector.py`, `advanced_detector.py`, `advanced_preprocessor.py` - Complex logic with unvalidated inputs
  - **Low Risk**: `enhancement.py`, `resize.py`, `padding.py` - Simple utilities with clear contracts
- **Key Issues Identified:**
  - Loose typing with `Any` types in core data structures
  - Minimal input validation for numpy arrays
  - Complex initialization patterns with multiple parameter sources
  - Inconsistent error handling and return types
  - Lack of clear data contracts between components
- **Dependencies Status:**
  - `pydantic>=2.0` - Available
  - `numpy` - Available
  - Test suite - Green for existing functionality

---

## 2. The Plan (The Living Blueprint)

## Progress Tracker
- **STATUS:** Phase 1 Complete - Ready for Phase 2
- **CURRENT PHASE:** Phase 1 - Core Data Structures
- **LAST COMPLETED TASK:** Create shared validation utilities
- **NEXT TASK:** Phase 1 Testing & Validation

### Implementation Phases (Checklist)

#### Phase 1: Core Data Structures (Weeks 1-2)
**Goal**: Establish validated data models for all core structures

1. [x] **Create ImageShape Pydantic model**
   - Add dimension validation (height, width, channels)
   - Implement custom validators for numpy arrays
   - Add to metadata.py

2. [x] **Refactor DocumentMetadata with Pydantic**
   - Replace loose `Any` types with strict typing
   - Maintain backward compatibility with `to_dict()` method
   - Add comprehensive field validation

3. [x] **Convert DocumentPreprocessorConfig to Pydantic**
   - Add comprehensive field validators
   - Implement cross-field validation for interdependent settings
   - Generate configuration schema

4. [x] **Create shared validation utilities**
   - `ImageValidator` class for numpy array validation
   - `ContractValidator` for data contract enforcement
   - Custom Pydantic types for common patterns

5. [ ] **Phase 1 Testing & Validation**
   - Type checking passes with strict mypy settings
   - Existing tests continue to pass
   - Clear error messages for validation failures

#### Phase 2: Input/Output Contracts (Week 3)
**Goal**: Define and implement data contracts for all component interfaces

1. [ ] **Define contract interfaces**
   - `ImageInput` contract with shape, dtype, and channel validation
   - `PreprocessingResult` contract with guaranteed fields
   - `DetectionResult` contract with confidence and metadata
   - `ErrorResponse` contract with standardized error codes

2. [ ] **Implement contract validation**
   - Add `@validate_call` decorators to public methods
   - Create contract enforcement utilities
   - Implement graceful degradation for invalid inputs

3. [ ] **Update pipeline interfaces**
   - Refactor `DocumentPreprocessor.__call__()` with contracts
   - Standardize return types across all components
   - Add input sanitization and validation

4. [ ] **Phase 2 Testing & Validation**
   - All public methods have validated contracts
   - Clear error messages for contract violations
   - Contract tests added to test suite

#### Phase 3: Component Refactoring (Week 4)
**Goal**: Refactor individual components to use validated interfaces

1. [ ] **Refactor detector components**
   - Update `DocumentDetector` with input validation
   - Implement `AdvancedDocumentDetector` contracts
   - Standardize detection result formats

2. [ ] **Update processing pipeline**
   - Simplify initialization patterns
   - Remove legacy compatibility layers where safe
   - Implement proper error propagation

3. [ ] **Enhance advanced preprocessor**
   - Simplify configuration mapping
   - Implement proper validation
   - Remove TODO items

4. [ ] **Phase 3 Testing & Validation**
   - All components use validated interfaces
   - Error handling is consistent across components
   - Performance impact < 5%

#### Phase 4: Testing and Documentation (Week 5)
**Goal**: Ensure quality and maintainability of refactored code

1. [ ] **Implement comprehensive testing**
   - Add property-based tests for validation
   - Create contract compliance tests
   - Add edge case testing for validation

2. [ ] **Update documentation**
   - Generate API documentation from Pydantic models
   - Create data contract documentation
   - Add migration guide for breaking changes

3. [ ] **Code quality improvements**
   - Add type stubs for external dependencies
   - Implement proper error codes and messages
   - Create validation utilities library

4. [ ] **Final Validation**
   - Test coverage > 90% for new validation code
   - All documentation updated
   - No performance regressions

---

## 3. 🎯 Goal & Contingencies

**Goal:** Complete Phase 1 testing and validation including type checking with strict mypy settings, ensuring existing tests continue to pass, and validating clear error messages for validation failures.

* **Success Condition:** If Phase 1 testing and validation passes completely:
  1. Update the `Progress Tracker` to mark task 1.5 as complete and set STATUS to "Phase 1 Complete".
  2. Set the `NEXT TASK` to "Define contract interfaces" (Phase 2).

* **Failure Condition:** If Phase 1 testing and validation fails:
  1. In your report, analyze the specific failures and identify root causes.
  2. Update the `Progress Tracker`'s `LAST COMPLETED TASK` to note the testing issues.
  3. Set the `NEXT TASK` to "Diagnose and fix Phase 1 testing issues".

---

## 4. Command
```bash
# Complete Phase 1 testing and validation
# This ensures all refactored components work correctly together
cd /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ocr/datasets/preprocessing

# Run comprehensive validation tests
python -c "
# Test all refactored components together
from .metadata import DocumentMetadata, ImageShape
from .config import DocumentPreprocessorConfig, EnhancementMethod
from .validators import ImageValidator, ContractValidator
from .pipeline import DocumentPreprocessor
import numpy as np

print('Running comprehensive Phase 1 validation...')

# Test 1: ImageShape and DocumentMetadata integration
shape = ImageShape(height=100, width=200, channels=3)
metadata = DocumentMetadata(original_shape=shape)
print('✓ ImageShape-DocumentMetadata integration works')

# Test 2: Config validation
config = DocumentPreprocessorConfig(
    target_size=(256, 256),
    enhancement_method='conservative',
    document_detection_min_area_ratio=0.2
)
print('✓ Config validation works')

# Test 3: Pipeline integration with new config
preprocessor = DocumentPreprocessor(config)
test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
result = preprocessor(test_image)
print('✓ Pipeline integration works')

# Test 4: Validator utilities
validated = ImageValidator.validate_image_array(test_image)
contract_validated = ContractValidator.validate_image_input_contract(test_image)
print('✓ Validator utilities work')

print('Phase 1 comprehensive validation PASSED!')
"
```

---

## Technical Implementation Details

### Pydantic Integration Strategy
```python
# Example: Enhanced metadata model structure
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import numpy as np

class ImageShape(BaseModel):
    """Validated image shape specification."""
    height: int = Field(gt=0, le=10000)
    width: int = Field(gt=0, le=10000)
    channels: int = Field(ge=1, le=4)

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> 'ImageShape':
        h, w = array.shape[:2]
        c = array.shape[2] if len(array.shape) > 2 else 1
        return cls(height=h, width=w, channels=c)

class DocumentMetadata(BaseModel):
    original_shape: ImageShape  # Strict typing replaces Any
    processing_steps: List[str] = Field(default_factory=list)
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True  # For numpy arrays
```

### Success Metrics
- **Type Safety**: 100% of public APIs with proper type hints
- **Test Coverage**: >90% coverage for validation logic
- **Performance**: <5% overhead from validation
- **Error Reduction**: 80% reduction in type-related runtime errors

### Risk Mitigation
- **Backward Compatibility**: Maintain existing APIs with deprecation warnings
- **Gradual Rollout**: Feature flags for validation components
- **Testing**: Comprehensive test coverage before each phase completion
- **Rollback**: Clear rollback procedures for each phase
