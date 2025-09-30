# **filename: docs/ai_handbook/02_protocols/04_utility_adoption.md**

# **Protocol: Utility Adoption**

This protocol governs the use and contribution to the project's shared utility modules. The primary goal is to adhere to the Don't Repeat Yourself (DRY) principle, ensuring that common logic is centralized, maintainable, and consistently applied.

## **1. The Principle: Check the Toolbox First**

Before writing any new helper function, you **must** first check if a suitable utility already exists in the shared "toolbox." Re-implementing logic that already exists leads to code duplication, bugs, and maintenance overhead.

## **2. The Discovery Process**

Your primary resource for discovering available tools is the reference document.

* **Consult the Reference:** Before implementing any common task (e.g., path manipulation, logging, visualization), you must review the **docs/ai_handbook/03_references/03_utility_functions.md** document.

This document serves as the canonical catalog of all approved, reusable utility functions.

## **3. The Adoption Workflow**

When you identify a utility function that fits your needs, follow these steps to integrate it:

1. **Import Correctly:** Import the function from its canonical path as specified in the reference document (e.g., from ocr.utils.path_utils import get_project_root).
2. **Replace Local Logic:** Remove any duplicated or similar logic from your current script and replace it with a call to the shared utility function.
3. **Validate:** Ensure that your code still functions as expected after the replacement. If the utility function is being used in a script covered by tests, run the relevant tests to confirm there are no regressions.

## **4. The Contribution Workflow**

If you develop a new piece of logic that is generic and could be useful in other parts of the codebase, you should contribute it back to the shared utilities.

1. **Identify a Candidate:** The logic should be a "pure" function where possible—meaning it is stateless and produces the same output for the same input. Good candidates are functions for formatting, calculation, or I/O operations.
2. **Add to the Correct Module:** Place your new function in the most appropriate utility module:
   * `ocr/utils/path_utils.py` for path and file system operations.
   * `ocr/utils/wandb_utils.py` for Weights & Biases integration.
   * `ocr/utils/ocr_utils.py` for general visualization or OCR-specific helpers.
3. **Document the Function:** Add a clear docstring and type hints to your new function, explaining what it does, its parameters, and what it returns.
4. **Update the Reference (CRITICAL):** This is the most important step. You **must** update the `docs/ai_handbook/03_references/03_utility_functions.md` document to include your new function. If the documentation is not updated, the utility does not officially "exist" for other agents or developers.
