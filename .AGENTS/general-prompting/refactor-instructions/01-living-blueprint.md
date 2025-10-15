# Living Refactor Blueprint: OCR Dataset Base

## Progress Tracker
- **STATUS:** In Progress
- **CURRENT STEP:** 3. Migrate runtime scripts...
- **LAST COMPLETED TASK:** Re-ran pytest coverage for CacheManager and integration tests.
- **NEXT TASK:** Migrate the preprocessing CLI script.

### Migration Outline (Checklist)
1. [x] Introduce compatibility accessors.
2. [x] Update Hydra schemas.
3. [ ] Migrate runtime scripts, callbacks, etc.
    - [x] `scripts/analysis_validation/profile_data_loading.py`
    - [x] `scripts/analysis_validation/validate_pipeline_contracts.py`
    - [x] `ocr/datasets/db_collate_fn.py`
    - [ ] **Pending:** Preprocessing CLI
    - [ ] **Pending:** Benchmarking/ablation scripts
    - [ ] **Pending:** Hydra runtime configs
    - [ ] **Pending:** Lightning callbacks
4. [ ] Refactor unit/integration tests.
5. [ ] Remove dead code paths and run final tests.

---
# [ The rest of your existing Procedural Blueprint follows... ]
