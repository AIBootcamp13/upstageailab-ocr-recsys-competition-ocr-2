Here is the latest version of our "Living Refactor Blueprint". Please continue with `NEXT` into Step 5 by removing dead code paths and running final tests to complete the refactor.

---


## Updated Living Blueprint

## Progress Tracker
- **STATUS:** In Progress
- **CURRENT STEP:** 4. Refactor unit/integration tests.
- **LAST COMPLETED TASK:** Integration tests now pass after fixing mock batch to include all CollateOutput fields.
- **NEXT TASK:** Remove dead code paths and run final tests.

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
4. [x] Refactor unit/integration tests.
5. [ ] Remove dead code paths and run final tests.


---
# [ The rest of your existing Procedural Blueprint follows... ]
