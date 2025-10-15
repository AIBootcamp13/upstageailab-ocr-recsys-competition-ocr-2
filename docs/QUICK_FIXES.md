# Quick Fixes Log

This log tracks quick fixes, patches, and hotfixes applied to the codebase.
Format follows the [Quick Fixes Protocol](docs/ai_handbook/02_protocols/development/11_quick_fixes_protocol.md).

---

## 2025-10-15 14:30 BUG - Pydantic replace() compatibility

**Issue**: TypeError: replace() should be called on dataclass instances in Streamlit UI preprocessing config
**Fix**: Replace dataclass replace() with Pydantic model_copy(update=) for PreprocessingConfig
**Files**: ui/apps/inference/state.py
**Impact**: minimal
**Test**: ui
