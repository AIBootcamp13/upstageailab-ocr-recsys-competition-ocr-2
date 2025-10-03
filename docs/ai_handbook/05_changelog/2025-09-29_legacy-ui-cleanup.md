# 2025-09-29 Legacy UI Cleanup

## Summary
Removed the deprecated Streamlit viewer wrappers and aligned the AI handbook with the modular UI workflow. All Streamlit launch commands are now consolidated behind `run_ui.py`.

## Changes Made

### **Codebase**
- Deleted the legacy `ui/test_viewer.py` entrypoint in favour of the modular evaluation dashboard.
- Removed the `ui/visualization/` compatibility package and the redundant `ui/utils/inference_engine.py` shim.
- Updated `ui/apps/inference/services/inference_runner.py` to consume the modular inference toolkit directly.
- Pruned the `serve-test-viewer` target from the `Makefile`.

### **Documentation**
- Bumped the handbook to version 1.1 with an updated quick link bundle for Streamlit operations.
- Added Streamlit launcher commands to the Command Registry and clarified the modular refactor protocol expectations.
- Recorded this cleanup in the changelog for traceability.

## Impact
- Streamlit launch instructions now map 1:1 with the code that ships in the repo.
- Agents no longer rely on stale compatibility wrappers; the modular UI stack is the single source of truth.
- Future refactors have clearer guidance on when it is safe to delete transitional shims.

## Next Steps
- Audit remaining docs or scripts for references to removed modules and update if encountered.
- Monitor Streamlit usage to confirm no workflows depended on the deleted test viewer.
