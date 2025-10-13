# **filename: docs/ai_handbook/02_protocols/development/04_utility_adoption.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=utility_adoption,code_reuse,refactoring -->

# **Protocol: Utility Adoption**

## **Overview**
This protocol governs the use and contribution to the project's shared utility modules. The primary goal is to adhere to the Don't Repeat Yourself (DRY) principle, ensuring that common logic is centralized, maintainable, and consistently applied across the codebase.

## **Prerequisites**
- Access to project repository and utility modules
- Familiarity with Python import system and module structure
- Understanding of the project's utility reference documentation
- Knowledge of basic refactoring principles

## **Procedure**

### **Step 1: Check the Toolbox First**
Before writing any new helper function, you **must** first check if a suitable utility already exists:

- Re-implementing existing logic leads to code duplication, bugs, and maintenance overhead
- Always prioritize reuse of existing utilities over creating new implementations
- Consult the utility reference document as the primary discovery resource

### **Step 2: Discover Available Utilities**
Use the reference documentation to find appropriate tools for common tasks:

**Consult the Reference Document:**
```markdown
docs/ai_handbook/03_references/03_utility_functions.md
```

This document serves as the canonical catalog of all approved, reusable utility functions covering:
- Path manipulation and file system operations
- Logging and visualization helpers
- OCR-specific utilities and formatting functions

### **Step 3: Adopt Existing Utilities**
When you identify a suitable utility function, integrate it properly:

**Import Correctly:**
```python
from ocr.utils.path_utils import get_project_root
from ocr.utils.wandb_utils import log_experiment
from ocr.utils.ocr_utils import visualize_predictions
```

**Replace Local Logic:**
- Remove any duplicated or similar logic from your current script
- Replace with calls to the shared utility function
- Ensure consistent API usage across the codebase

**Validate Integration:**
- Confirm code functions as expected after replacement
- Run relevant tests to prevent regressions
- Verify no breaking changes in dependent functionality

### **Step 4: Contribute New Utilities**
If you develop generic logic useful across the codebase, contribute it back:

**Identify Candidates:**
- Pure functions (stateless, same output for same input)
- Generic operations: formatting, calculation, I/O operations
- Logic that appears in multiple modules

**Add to Appropriate Module:**
- `ocr/utils/path_utils.py` - path and file system operations
- `ocr/utils/wandb_utils.py` - Weights & Biases integration
- `ocr/utils/ocr_utils.py` - visualization and OCR-specific helpers

**Document Properly:**
- Add clear docstring with parameters and return values
- Include type hints for better IDE support
- Follow existing documentation patterns

**Update Reference (CRITICAL):**
- Update `docs/ai_handbook/03_references/03_utility_functions.md`
- Include function signature, description, and usage examples
- This step makes the utility officially "discoverable" for other developers

## **Validation**
- All new code uses existing utilities where appropriate
- No duplicate implementations of common functionality
- Utility reference documentation is current and accurate
- Tests pass after utility adoption or contribution
- Code follows established import patterns and conventions

## **Troubleshooting**
- If utility function doesn't meet needs, consider extending rather than replacing
- When reference documentation is outdated, update it immediately
- For complex utilities, ensure proper error handling and edge cases
- If tests fail after adoption, verify parameter compatibility

## **Related Documents**
- [Utility Functions Reference](../../references/architecture/03_utility_functions.md) - Catalog of available utilities
- [Coding Standards](01_coding_standards.md) - Development best practices
- [Modular Refactor](05_modular_refactor.md) - Code organization principles
- [Refactoring Guide](10_refactoring_guide.md) - Advanced refactoring techniques
