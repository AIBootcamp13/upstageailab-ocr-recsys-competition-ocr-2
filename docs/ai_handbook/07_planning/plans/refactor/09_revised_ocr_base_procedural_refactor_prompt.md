You are an autonomous AI software engineer, my Chief of Staff for a complex code refactor. Your primary responsibility is to execute the "Living Refactor Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

**Your Core Workflow is a Goal-Execute-Update Loop:**
1.  **Goal:** I will provide a clear `🎯 Goal` for you to achieve.
2.  **Execute:** You will run the `[COMMAND]` provided to work towards that goal.
3.  **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
    * **Part 1: Execution Report:** Display the results and your analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
    * **Part 2: Updated Living Blueprint:** Provide the COMPLETE, UPDATED content of the "Living Refactor Blueprint", updating the `Progress Tracker` with the new status and the correct `NEXT TASK` based on the outcome.

---


### ## 1. Current State (Based on Last Session)
- **Project:** `ocr_pl` dataset refactor on branch `08_refactor/ocr_pl`.
- **Blueprint:** "Procedural Refactor Blueprint: OCR Dataset Base".
- **Current Position:** Execution is in **Step 4** of the "Migration Outline".
- **Completed Migrations:**
    - `scripts/analysis_validation/profile_data_loading.py`
    - `scripts/analysis_validation/validate_pipeline_contracts.py`
    - `ocr/datasets/db_collate_fn.py`
- **Completed Tests:**
    - `tests/test_validated_ocr_dataset.py` (11 passed).
    - Contract validation script is green.
- **Pending Tests:**
    - `test_ocr_lightning_predict_integration.py` (re-run required).
- **Pending Tasks:**
    - Refactor unit/integration tests.
- **Known Issues:**
    - Training run fails with ValueError: Batch validation failed: 1 validation error for CollateOutput
    - Degenerate polygons logged in `logs/convex_hull_failures.jsonl`.

---

### ## 2. The Plan (The Living Blueprint)

## Progress Tracker
- **STATUS:** In Progress
- **CURRENT STEP:** 4. Refactor unit/integration tests.
- **LAST COMPLETED TASK:** Lightning callbacks now source metadata from `ValidatedOCRDataset` artifacts.
- **NEXT TASK:** Expand regression coverage for the validated dataset + Lightning bridge.

### Migration Outline (Checklist)
1. [x] Introduce compatibility accessors.
2. [x] Update Hydra schemas.
3. [x] Migrate runtime scripts, callbacks, etc.
    - [x] profile_data_loading.py
    - [x] validate_pipeline_contracts.py
    - [x] db_collate_fn.py
    - [x] Preprocessing CLI (preprocess_data.py)
    - [x] Benchmarking/ablation scripts
    - [x] Hydra runtime configs
    - [x] Lightning callbacks
4. [ ] Refactor unit/integration tests.
5. [ ] Remove dead code paths and run final tests.

Next up, move into Step 4 by tightening unit/integration coverage around the validated-dataset ↔ Lightning handoff (e.g., end-to-end callback smoke tests and cache-path regressions).

---

### ## 3. 🎯 Goal & Contingencies

**Goal:** Verify that all integration tests pass after the recent refactoring.

* **Success Condition:** If all tests pass, your task is to:
    1.  Update the `Progress Tracker` to mark the verification step as complete.
    2.  Set the `NEXT TASK` to "Migrate the preprocessing CLI script" as per the Migration Outline.

* **Failure Condition:** If any test fails, your task is to:
    1.  In your report, analyze the traceback and diagnose the root cause of the failure.
    2.  Update the `Progress Tracker`'s `LAST COMPLETED TASK` to note the test failure.
    3.  Set the `NEXT TASK` to "Diagnose and fix the failing integration tests."

---

### ## 4. Command
```bash
pytest tests/integration/test_ocr_lightning_predict_integration.py
````

---
