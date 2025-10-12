# 2025-10-12: Wandb Run Name Generation Logic Bug

## Summary
Fixed a bug in Wandb run name generation where component token extraction incorrectly prioritized `component_overrides` over direct component configurations, causing run names to display outdated model names instead of the actual models being used.

## Root Cause Analysis
The `_extract_component_token` function in `ocr/utils/wandb_utils.py` was checking `component_overrides` before the direct component configuration. This caused issues when users specified model parameters directly (e.g., `model.encoder.model_name=resnet50`) but the system used values from `component_overrides` that were set during config loading.

In the reported case:
- User command: `model.encoder.model_name=resnet50`
- Config loaded: `component_overrides.encoder.model_name` from dbnet config (resnet18)
- Run name showed: "resnet18" (incorrect)
- Model actually used: "resnet50" (correct)

## Changes Made
- **File**: `ocr/utils/wandb_utils.py`
  - Modified `_extract_component_token` function to prioritize direct component configuration over `component_overrides`
  - Swapped the order: check `component_cfg` first, then `overrides`, then architecture defaults
  - This ensures user-specified overrides take precedence over preset component overrides

## Impact
- **User-facing**: Wandb run names now accurately reflect the actual model configurations being executed
- **Functionality**: No breaking changes - existing behavior preserved for cases without direct overrides
- **Compatibility**: Maintains backward compatibility while fixing the precedence logic

## Testing
- Verified the fix with a test case showing correct prioritization
- Confirmed that direct component overrides now take precedence
- Validated that existing functionality remains intact

## Related Issues
- This fix addresses the mismatch between Wandb UI run names and executed commands
- Related to ongoing refactoring efforts to modularize `wandb_utils.py` with Pydantic validation
