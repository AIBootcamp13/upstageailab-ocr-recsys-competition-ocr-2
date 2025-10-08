# **Actionable Refactor Plan: Consolidate Redundant Configurations**

**Objective:** Refactor the configs/ directory to eliminate duplication, improve modularity, and align with best practices from templates like lightning-hydra-template. The goal is to make the configuration more intuitive and maintainable.

### **In-Depth Analysis of Current State**

1. **Duplicated Data Definitions:**
   * configs/data/default.yaml defines a complete dataset and transform pipeline.
   * configs/preset/datasets/db.yaml defines an almost identical dataset and transform pipeline.
   * This means a change to a transform, like adding a new augmentation, must be manually duplicated in both files, which is error-prone.
2. **Tightly Coupled Transforms:**
   * The transforms definitions are nested *inside* the dataset configuration files.
   * This makes it difficult to reuse a set of transforms with a different dataset or to experiment with transforms independently of the data source.
3. **Inconsistent Hierarchy:**
   * Some core settings are in configs/base.yaml.
   * Others are in configs/data/default.yaml.
   * And still more are in configs/preset/datasets/db.yaml.
   * This scattered approach makes it hard to understand the default configuration at a glance.

**The Target State:** A structure where data, transforms, and dataloaders are independent, composable config groups.

### **Phase 1: Create Base Config Groups**

This phase establishes the new, modular structure.

Action 1: Create configs/data/base.yaml.
This file will define the data sources, not the transforms.
```yaml
# @package _global_

dataset_base_path: "${hydra:runtime.cwd}/data/datasets/"

datasets:
  train_dataset:
    _target_: ocr.datasets.OCRDataset
    image_path: ${dataset_base_path}images/train
    annotation_path: ${dataset_base_path}jsons/train.json
    transform: ${transforms.train_transform} # Note: References the transforms group
  val_dataset:
    _target_: ocr.datasets.OCRDataset
    image_path: ${dataset_base_path}images/val
    annotation_path: ${dataset_base_path}jsons/val.json
    transform: ${transforms.val_transform}
  # ... and so on for test/predict
```

Action 2: Create configs/transforms/base.yaml.
Move the transform definitions here, making them a reusable component.
```yaml
# @package _global_

transforms:
  train_transform:
    _target_: ocr.datasets.DBTransforms
    # ... paste the transforms list from the old file here
  val_transform:
    _target_: ocr.datasets.DBTransforms
    # ... paste the transforms list here
  # ... and so on for test/predict
```

Action 3: Update configs/dataloaders/default.yaml.
Ensure this file is clean and ready to be used as a default group. It already looks good.

### **Phase 2: Refactor Existing Configs to Use Defaults**

Update the old files to inherit from the new base configurations.

Action 1: Modify configs/preset/datasets/db.yaml.
Replace its entire content with a defaults list.
```yaml
# @package _global_

defaults:
  - /data/base
  - /transforms/base
  - /dataloaders/default
```

Action 2: Modify configs/data/default.yaml.
Do the same for this file. It now becomes a simple preset.
```yaml
# @package _global_

defaults:
  - /data/base
  - /transforms/base
  - /dataloaders/default
```

### **Phase 3: Update Main Entry-Point Configs**

Modify train.yaml, test.yaml, and predict.yaml to use the new config groups.

Action 1: Modify configs/train.yaml.
Update its defaults list to compose the new groups.
```yaml
# In configs/train.yaml
defaults:
  - _self_
  - base                  # Keep this for other base settings
  - data: base            # Compose the data group
  - transforms: base      # Compose the transforms group
  - dataloaders: default  # Compose the dataloaders group
  - /preset/models/model_example
  - /preset/lightning_modules/base
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled
```

*Note: You may need to adjust the base default or other items depending on the exact hierarchy.*

### **Phase 4: Validation**

Use Hydra's built-in tools to verify that the final composed configuration is correct.

Action 1: Print the resolved configuration.
Run your training script with the --cfg job flag. This will print the final configuration that Hydra has composed, without starting the training.
```bash
uv run python runners/train.py --cfg job
```

Inspect the output to ensure that the datasets, transforms, and dataloaders sections are correctly populated.

### **Prompt for Agentic AI (Next Session)**

```text
Objective: Execute the configuration consolidation plan. Follow the four phases outlined in `refactor_plan_consolidate_configs.md`. Create the new base config groups, refactor the old files to use them, update the main `train.yaml` entry point, and use Hydra's `--cfg job` command to validate the final structure.
```
