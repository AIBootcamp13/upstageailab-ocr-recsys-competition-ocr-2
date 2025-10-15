# # **filename: docs/ai_handbook/03_references/architecture/07_checkpoint_naming_scheme.md**

<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=when_working_on_training_checkpoints_or_experiment_management -->

## **Overview**

This reference document describes the enhanced checkpoint naming convention for better organization, clarity, and management of model checkpoints in the OCR training system.

## **Key Concepts**

### **Purpose**

The checkpoint naming scheme provides:
- **Clarity**: Each component provides actionable information
- **Consistency**: Standardized separators and formatting
- **Searchability**: Easy filtering and finding of checkpoints
- **Automation-Friendly**: Scripts can easily parse and manage checkpoints
- **Version Control**: Track experiment iterations over time

### **Directory Structure**

```
outputs/
└── <experiment_tag>-<model>_<phase>_<timestamp>/
    ├── checkpoints/
    │   ├── epoch-03_step-000103.ckpt
    │   ├── last.ckpt
    │   └── best-hmean-0.8920.ckpt
    ├── logs/
    └── submissions/
```

## **Detailed Information**

### **Experiment Directory Format**

Format: `<experiment_tag>-<model>_<phase>_<timestamp>`

**Components:**

1. **experiment_tag**: Unique identifier for the experiment
   - Default: Uses `exp_name` from config
   - Override: Set via environment variable `EXPERIMENT_TAG`
   - Examples: `ocr_pl_refactor_phase1`, `baseline_v2`, `ablation_test3`

2. **model**: Model architecture and encoder
   - Format: `<architecture>-<encoder>`
   - Automatically extracted from model configuration
   - Examples: `dbnet-resnet18`, `craft-mobilenetv3`, `unknown-unknown`

3. **phase**: Training phase/stage
   - Default: `training`
   - Configurable in checkpoint callback config
   - Examples: `training`, `validation`, `finetuning`, `preprocessing`

4. **timestamp**: Creation timestamp (YYYYMMDD_HHMMSS)
   - Automatically generated on initialization
   - Example: `20251015_120000`

**Example Directory Names:**
```
ocr_pl_refactor_phase1-dbnet-resnet18_training_20251015_120000
baseline_v2-craft-mobilenetv3_validation_20251015_143022
ablation_test3-unknown-unknown_training_20251015_160512
```

### **Checkpoint Filename Patterns**

Three checkpoint types with distinct naming patterns:

#### **Epoch Checkpoints**

Saved at the end of each epoch.

Format: `epoch-<epoch>_step-<step>[_<metric>-<value>].ckpt`

Components:
- `epoch`: Zero-padded epoch number (2 digits)
- `step`: Zero-padded global step number (6 digits)
- `metric`: Optional metric name and value (if `auto_insert_metric_name: true`)

Examples:
```
epoch-03_step-000103.ckpt
epoch-05_step-000205_hmean-0.8756.ckpt
epoch-10_step-000512_hmean-0.9123.ckpt
```

#### **Last Checkpoint**

The most recent checkpoint, updated continuously.

Format: `last.ckpt`

This is a simple, constant filename that always represents the latest state.

#### **Best Checkpoints**

Saved when a monitored metric improves.

Format: `best-<metric_name>-<value>.ckpt`

Components:
- `metric_name`: Name of monitored metric (without prefix)
- `value`: Metric value (4 decimal places)

Examples:
```
best-hmean-0.8920.ckpt
best-hmean-0.9145.ckpt
best-loss-0.0234.ckpt
```

## **Examples**

### **Example 1: Training a New Model**

```bash
export EXPERIMENT_TAG="ocr_baseline_v1"
python runners/train.py preset=example
```

**Result:**
```
outputs/ocr_baseline_v1-dbnet-resnet18_training_20251015_120000/
├── checkpoints/
│   ├── epoch-00_step-000000.ckpt
│   ├── epoch-01_step-000051.ckpt
│   ├── epoch-02_step-000102.ckpt
│   ├── best-hmean-0.8567.ckpt
│   ├── best-hmean-0.8923.ckpt
│   └── last.ckpt
```

### **Example 2: Fine-tuning**

```bash
export EXPERIMENT_TAG="ocr_finetuned_receipts"
python runners/train.py preset=finetune \
  callbacks.model_checkpoint.training_phase="finetuning"
```

**Result:**
```
outputs/ocr_finetuned_receipts-dbnet-resnet50_finetuning_20251015_143500/
├── checkpoints/
│   ├── epoch-00_step-000000.ckpt
│   ├── best-hmean-0.9201.ckpt
│   └── last.ckpt
```

### **Example 3: Searching for Checkpoints**

```bash
# Find all phase1 experiments
ls outputs/ocr_pl_refactor_phase1-*

# Find all ResNet18 checkpoints
ls outputs/*-resnet18_*/checkpoints/

# Find all best checkpoints
find outputs -name "best-*.ckpt"

# Find specific training phase
ls outputs/*_validation_*/checkpoints/
```

### **Example 4: Parsing Checkpoint Names**

```python
import re
from pathlib import Path

# Parse checkpoint directory
pattern = r"(.+)-(.+)_(.+)_(\d{8}_\d{6})"
for checkpoint_dir in Path("outputs").iterdir():
    if match := re.match(pattern, checkpoint_dir.name):
        exp_tag, model, phase, timestamp = match.groups()
        print(f"Found: {exp_tag} using {model} in {phase} phase")

# Find latest best checkpoint
checkpoints = sorted(Path("outputs").glob("*/checkpoints/best-*.ckpt"))
latest_best = checkpoints[-1] if checkpoints else None
```

## **Configuration Options**

### **Basic Configuration**

Update `configs/callbacks/model_checkpoint.yaml`:

```yaml
model_checkpoint:
  _target_: ocr.lightning_modules.callbacks.unique_checkpoint.UniqueModelCheckpoint
  dirpath: ${paths.checkpoint_dir}

  # Experiment identification
  experiment_tag: ${oc.env:EXPERIMENT_TAG,${exp_name}}
  training_phase: "training"
  add_timestamp: true

  # Checkpoint settings
  filename: "best"
  monitor: "val/hmean"
  save_last: true
  save_top_k: 3
  mode: "max"
  auto_insert_metric_name: true
  every_n_epochs: 1
```

### **Configuration Parameters Reference**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_tag` | str | `${exp_name}` | Unique experiment identifier |
| `training_phase` | str | `"training"` | Training phase/stage |
| `add_timestamp` | bool | `true` | Add timestamp to directory |
| `filename` | str | `"best"` | Template for checkpoint type |
| `monitor` | str | `"val/hmean"` | Metric to monitor |
| `save_last` | bool | `true` | Save last checkpoint |
| `save_top_k` | int | `3` | Number of best checkpoints to keep |
| `mode` | str | `"max"` | Optimization mode ("max" or "min") |
| `auto_insert_metric_name` | bool | `true` | Include metric in filename |
| `every_n_epochs` | int | `1` | Checkpoint frequency |

### **Setting Experiment Tag**

#### Option 1: Environment Variable (Recommended)
```bash
export EXPERIMENT_TAG="ocr_pl_refactor_phase1"
python runners/train.py
```

#### Option 2: Command Line Override
```bash
python runners/train.py callbacks.model_checkpoint.experiment_tag="ocr_pl_refactor_phase1"
```

#### Option 3: Config File
```yaml
# In your preset config
defaults:
  - /callbacks/model_checkpoint

callbacks:
  model_checkpoint:
    experiment_tag: "ocr_pl_refactor_phase1"
```

### **Setting Training Phase**

```yaml
callbacks:
  model_checkpoint:
    training_phase: "validation"  # or "finetuning", "preprocessing", etc.
```

## **Best Practices**

### **1. Use Descriptive Experiment Tags**
❌ Bad: `test1`, `exp2`, `final`
✅ Good: `ocr_baseline_v1`, `ablation_fpn_decoder`, `finetuned_receipts_aug`

### **2. Use Semantic Training Phases**
- `training`: Initial training from scratch or pretrained weights
- `finetuning`: Fine-tuning on specific dataset
- `validation`: Validation runs
- `preprocessing`: Experiments with preprocessing
- `ablation`: Ablation studies

### **3. Set Experiment Tags via Environment Variables**
This allows easy scripting without modifying configs:

```bash
#!/bin/bash
experiments=("baseline_v1" "with_augmentation" "larger_model")

for exp in "${experiments[@]}"; do
    export EXPERIMENT_TAG="ocr_${exp}"
    python runners/train.py preset=example
done
```

### **4. Maintain an Experiment Log**
Keep a log of experiments with their tags in `docs/ai_handbook/04_experiments/`:

```markdown
# Experiment Log

## 2025-10-15
- `ocr_baseline_v1`: Initial baseline with ResNet18
- `ocr_baseline_v2`: Same as v1 but with data augmentation
- `ablation_decoder_fpn`: Testing FPN decoder architecture
```

### **5. Regular Cleanup**
Run cleanup scripts weekly to manage disk space. See [Checkpoint Migration Protocol](../../02_protocols/components/18_checkpoint_migration_protocol.md) for cleanup strategies.

## **Troubleshooting**

### **Issue: Directory name doesn't include model info**

**Cause**: Model information extraction failed or model doesn't have the expected attributes.

**Solution**: The callback will use "unknown" as fallback. To fix:
1. Ensure your model has `architecture_name` attribute
2. Ensure your encoder has `model_name` attribute
3. Check logs for extraction errors

### **Issue: Checkpoint filenames don't include metrics**

**Cause**: `auto_insert_metric_name` is set to `False` or monitored metric is not in the metrics dict.

**Solution**:
1. Set `auto_insert_metric_name: true` in config
2. Verify the `monitor` value matches your logged metric name
3. Check that metrics are logged correctly in your Lightning module

### **Issue: Timestamps are the same for multiple runs**

**Cause**: The callback is being reused across multiple training runs in the same process.

**Solution**: Each callback instance generates its timestamp on initialization. For multiple runs, create a new callback instance or restart the process.

## **Related References**

- [Checkpoint Migration Protocol](../../02_protocols/components/18_checkpoint_migration_protocol.md) - How to migrate existing checkpoints
- [Training Protocol](../../02_protocols/components/13_training_protocol.md) - Complete training workflow
- [PyTorch Lightning Checkpoint Callbacks](../guides/pytorch_lightning_checkpoint_callbacks.md) - General checkpoint callback reference

---

**Last Updated**: 2025-10-15
**Owner**: ml-platform
**Related Files**:
- `ocr/lightning_modules/callbacks/unique_checkpoint.py`
- `configs/callbacks/model_checkpoint.yaml`
