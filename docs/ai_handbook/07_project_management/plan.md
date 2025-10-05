
---

# **Enhanced Actionable Refactor Plan: Consolidate Redundant Configurations**

**Objective:** Refactor the configs directory to eliminate duplication, improve modularity, and align with best practices from the [lightning-hydra-template](docs/external/lightning-hydra-template). The goal is to make the configuration more intuitive and maintainable.

### **Prerequisites for Qwen Coder**
- **Context Files to Attach:**
  - `#file:repomix-output.xml` (compressed codebase overview; search this file for class definitions, imports, and existing code patterns).
  - Current `configs/base.yaml`, `configs/data/default.yaml`, `configs/preset/datasets/db.yaml`, `configs/dataloaders/default.yaml`, `configs/train.yaml`, `configs/test.yaml`, `configs/predict.yaml`.
  - Reference: Local `docs/external/lightning-hydra-template/configs/` for best practices (adapt `conf/` to `configs/` as needed).
- **Context Window Management:** Work in phases. After each phase, summarize changes and reset context if Qwen's responses show signs of confusion (e.g., repeating old code). Limit each prompt to 1-2 phases. Always attach `#file:repomix-output.xml` for codebase context.
- **Validation Environment:** Ensure Qwen can run `uv run python runners/train.py --cfg job` in a terminal to test configurations.

### **Phase 1: Create Base Config Groups**

**Objective:** Establish modular config groups by splitting data, transforms, and dataloaders.

**Action 1: Create `configs/data/base.yaml`.**
- This defines datasets without transforms (transforms will reference separately).
- **Exact Content to Create:**
  ```yaml
  # @package _global_

  dataset_base_path: "${hydra:runtime.cwd}/data/datasets/"

  datasets:
    train_dataset:
      _target_: ${dataset_path}.OCRDataset
      image_path: ${dataset_base_path}images/train
      annotation_path: ${dataset_base_path}jsons/train.json
      transform: ${transforms.train_transform}
    val_dataset:
      _target_: ${dataset_path}.OCRDataset
      image_path: ${dataset_base_path}images/val
      annotation_path: ${dataset_base_path}jsons/val.json
      transform: ${transforms.val_transform}
    test_dataset:
      _target_: ${dataset_path}.OCRDataset
      image_path: ${dataset_base_path}images/val
      annotation_path: ${dataset_base_path}jsons/val.json
      transform: ${transforms.test_transform}
    predict_dataset:
      _target_: ${dataset_path}.OCRDataset
      image_path: ${dataset_base_path}images/test
      annotation_path: null
      transform: ${transforms.test_transform}

  collate_fn:
    _target_: ${dataset_path}.DBCollateFN
    shrink_ratio: 0.4
    thresh_min: 0.3
    thresh_max: 0.7
  ```
- **Why:** Separates data sources from transforms for reusability.

**Action 2: Create `configs/transforms/base.yaml`.**
- Move transforms here as a reusable group.
- **Exact Content to Create:** Copy the `transforms` section from the current default.yaml (lines ~10-80).
- **Expected Structure:**
  ```yaml
  # @package _global_

  transforms:
    train_transform:
      _target_: ${dataset_path}.DBTransforms
      # ... (paste exact transforms from default.yaml)
    val_transform:
      # ... (paste exact)
    test_transform:
      # ... (paste exact)
    predict_transform:
      # ... (paste exact)
  ```

**Action 3: Verify `configs/dataloaders/default.yaml`.**
- It should already be clean (check for any hardcoded values; ensure it uses `${data.batch_size}`).

**Validation for Phase 1:**
- Run: `uv run python -c "import hydra; from omegaconf import OmegaConf; cfg = hydra.compose(config_name='data/base'); print(OmegaConf.to_yaml(cfg))"`
- Expected: No errors; datasets and transforms sections should resolve.

### **Phase 2: Refactor Existing Configs to Use Defaults**

**Objective:** Replace duplicated content with inheritance.

**Action 1: Modify `configs/preset/datasets/db.yaml`.**
- **Replace Entire Content With:**
  ```yaml
  # @package _global_

  defaults:
    - /data/base
    - /transforms/base
    - /dataloaders/default
  ```
- **Why:** Inherits from the new base groups instead of duplicating.

**Action 2: Modify `configs/data/default.yaml`.**
- **Replace Entire Content With:**
  ```yaml
  # @package _global_

  defaults:
    - /data/base
    - /transforms/base
    - /dataloaders/default
  ```
- **Why:** Turns it into a preset that composes the bases.

**Validation for Phase 2:**
- Run: `uv run python -c "import hydra; cfg = hydra.compose(config_name='preset/datasets/db'); print('Success' if 'datasets' in cfg else 'Failed')"`
- Expected: "Success"; no YAML errors.

### **Phase 3: Update Main Entry-Point Configs**

**Objective:** Explicitly compose groups in entry points for clarity.

**Action 1: Modify `configs/train.yaml`.**
- **Update Defaults List To:**
  ```yaml
  defaults:
    - _self_
    - base                  # Keep for other settings
    - data: base            # Compose data group
    - transforms: base      # Compose transforms group
    - dataloaders: default  # Compose dataloaders group
    - /preset/models/model_example
    - /preset/lightning_modules/base
    - override hydra/hydra_logging: disabled
    - override hydra/job_logging: disabled
  ```
- **Note:** If conflicts arise (e.g., `base` already includes `data: default`), remove `data: default` from `configs/base.yaml` temporarily.

**Action 2: Apply Same Changes to test.yaml and predict.yaml.**
- Use the same defaults list as above.

**Validation for Phase 3:**
- Run: `uv run python train.py --cfg job | head -50`
- Expected: No errors; check that `datasets`, `transforms`, and `dataloaders` sections are populated correctly.

### **Phase 4: Final Validation**

**Action 1: Full Configuration Print.**
- Run: `uv run python train.py --cfg job`
- Inspect output: Ensure `datasets.train_dataset`, `transforms.train_transform`, and `dataloaders.train_dataloader` are correctly composed.
- Expected Metrics: No missing keys; transforms reference properly.

**Action 2: Smoke Test.**
- Run: `uv run python train.py trainer.fast_dev_run=true`
- Expected: Runs without config errors.

**Rollback Instructions:** If issues occur, `git checkout HEAD -- configs/` to revert all changes.

---

### Detailed Prompts for Qwen Coder
Use these prompts sequentially, attaching the required files each time. Start a new session for each phase to manage context.

**Prompt for Phase 1:**
```
Objective: Execute Phase 1 of the config consolidation from the enhanced plan. Create `configs/data/base.yaml` and `configs/transforms/base.yaml` with the exact content provided. Verify `configs/dataloaders/default.yaml` is clean. Run the validation command and share the output. If errors, explain and suggest fixes.

Attachments: #file:repomix-output.xml, current configs files, lightning-hydra-template reference.
```

**Prompt for Phase 2:**
```
Objective: Execute Phase 2. Replace the content of `configs/preset/datasets/db.yaml` and `configs/data/default.yaml` with the exact defaults lists provided. Run the validation and share results.

Attachments: Updated files from Phase 1, repomix.
```

**Prompt for Phase 3:**
```
Objective: Execute Phase 3. Update the defaults in `configs/train.yaml`, `configs/test.yaml`, and `configs/predict.yaml` as specified. Handle any conflicts with `base.yaml`. Run validation and share output.

Attachments: Files from Phase 2.
```

**Prompt for Phase 4:**
```
Objective: Execute Phase 4. Run the full --cfg job command and smoke test. Confirm the configuration is correct. If issues, provide exact error messages for debugging.

Attachments: All updated configs.
```

### Additional Guidance for You
- **Context Overload Prevention:** Monitor Qwen's responses for repetition or hallucinations (e.g., generating old YAML). If seen, restart with a fresh prompt and summary of prior changes.
- **Hydra Template Reference:** Point Qwen to [this config structure](https://github.com/Lightning-AI/lightning-hydra-template/tree/main/conf) for inspiration on modularity.
- **Testing:** After Qwen completes, run a full training epoch to ensure no regressions.
- **Next Steps:** Once configs are done, use similar structured prompts for the Lightning module refactor.

This should give Qwen a clear, executable path while keeping things efficient! Let me know if you need the Lightning module prompts too.
