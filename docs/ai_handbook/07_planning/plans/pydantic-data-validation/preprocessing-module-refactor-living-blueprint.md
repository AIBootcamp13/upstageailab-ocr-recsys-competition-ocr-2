# Preprocessing Module Refactor: Living Implementation Blueprint

## Overview

You are an autonomous AI software engineer executing a systematic refactor of the preprocessing module to address data type uncertainties, improve type safety, and reduce development friction. Your primary responsibility is to execute the "Living Refactor Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will run the `[COMMAND]` provided to work towards that goal.
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Display the results and your analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Updated Living Blueprint:** Provide the COMPLETE, UPDATED content of the "Living Refactor Blueprint", updating the `Progress Tracker` with the new status and the correct `NEXT TASK` based on the outcome.

---

## 1. Current State (Based on Assessment)

- **Project:** Preprocessing module refactor on branch `09_refactor/ocr_base`.
- **Blueprint:** "Preprocessing Module Pydantic Validation Refactor".
- **Current Position:** Pre-implementation phase - assessment complete, ready to begin Phase 1.
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
- **STATUS:** Ready to Start
- **CURRENT PHASE:** Pre-implementation
- **LAST COMPLETED TASK:** Comprehensive assessment and risk classification completed
- **NEXT TASK:** Begin Phase 1 - Create ImageShape Pydantic model in metadata.py

### Implementation Phases (Checklist)

#### Phase 1: Core Data Structures (Weeks 1-2)
**Goal**: Establish validated data models for all core structures

1. [ ] **Create ImageShape Pydantic model**
   - Add dimension validation (height, width, channels)
   - Implement custom validators for numpy arrays
   - Add to metadata.py

2. [ ] **Refactor DocumentMetadata with Pydantic**
   - Replace loose `Any` types with strict typing
   - Maintain backward compatibility with `to_dict()` method
   - Add comprehensive field validation

3. [ ] **Convert DocumentPreprocessorConfig to Pydantic**
   - Add comprehensive field validators
   - Implement cross-field validation for interdependent settings
   - Generate configuration schema

4. [ ] **Create shared validation utilities**
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

**Goal:** Create the ImageShape Pydantic model as the foundation for the refactor.

* **Success Condition:** If the ImageShape model is successfully created and integrated:
  1. Update the `Progress Tracker` to mark task 1.1 as complete.
  2. Set the `NEXT TASK` to "Refactor DocumentMetadata with Pydantic".

* **Failure Condition:** If the ImageShape model creation fails:
  1. In your report, analyze the error and identify the root cause.
  2. Update the `Progress Tracker`'s `LAST COMPLETED TASK` to note the failure.
  3. Set the `NEXT TASK` to "Diagnose and fix ImageShape model implementation issues".

---

## 4. Command
```bash
# Create ImageShape Pydantic model in metadata.py
# This establishes the foundation for validated data structures
cd /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ocr/datasets/preprocessing

# First, let's examine the current metadata.py structure
head -50 metadata.py
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
