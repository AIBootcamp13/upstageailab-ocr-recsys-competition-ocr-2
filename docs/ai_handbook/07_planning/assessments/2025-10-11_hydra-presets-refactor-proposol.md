# Refactor Notes

## Configuration Management Refactor Proposal

### Current Structure Issues
- Numerous small YAML files scattered across `configs/preset/` subdirectories
- Deep nesting (e.g., `preset/models/encoder/`, `decoder/`, `head/`, `loss/`)
- Each model configuration composes from 5-7 separate component files
- Dataset configurations are minimal and just reference other groups
- Hard to overview the full configuration for a model or dataset
- Maintenance burden with many small files

### Proposed Refactored Structure

#### 1. Flatten Model Configurations
**Before:**
```
configs/preset/models/
├── dbnetpp.yaml (composes from 6+ files)
├── encoder/
│   ├── craft_vgg.yaml
│   └── timm_backbone.yaml
├── decoder/
│   ├── craft_decoder.yaml
│   ├── dbpp_decoder.yaml
│   └── unet.yaml
├── head/
│   ├── craft_head.yaml
│   ├── db_head.yaml
│   └── dbpp_head.yaml
└── loss/
    ├── craft_loss.yaml
    └── db_loss.yaml
```

**After:**
```
configs/preset/models/
├── base.yaml          # Common model settings
├── dbnet.yaml         # Complete DBNet config with inline components
├── dbnetpp.yaml       # Complete DBNet++ config with inline components
└── craft.yaml         # Complete CRAFT config with inline components
```

#### 2. Inline Component Configurations
Instead of separate files for encoder/decoder/head/loss, include them as nested keys in the model config:

```yaml
# configs/preset/models/dbnetpp.yaml
# @package model

defaults:
  - /model/architectures: dbnetpp
  - /model/optimizers: adamw
  - _self_

# Inline component configurations
encoder:
  _target_: ${encoder_path}.TimmBackbone
  model_name: "resnet50"
  select_features: [1, 2, 3, 4]
  pretrained: true

decoder:
  _target_: ${decoder_path}.DBPPDecoder
  inner_channels: 256
  out_channels: 128

head:
  _target_: ${head_path}.DBHead
  upscale: 4

loss:
  _target_: ${loss_path}.DBLoss
  l1_scale: 10
  bce_scale: 1

optimizer:
  lr: 0.0003
  weight_decay: 0.0001
```

#### 3. Environment-Based Dataset Configurations
**Before:**
```
configs/preset/datasets/
├── db.yaml (just compositions)
├── preprocessing.yaml
├── preprocessing_camscanner.yaml
└── preprocessing_docTR_demo.yaml
```

**After:**
```
configs/preset/datasets/
├── base.yaml          # Common dataset settings
├── development.yaml   # Local development paths
├── production.yaml    # Production/cloud paths
└── synthetic.yaml     # Synthetic data settings
```

Example development.yaml:
```yaml
# @package _global_

defaults:
  - base
  - /transforms/base
  - /dataloaders/default

dataset_base_path: "${hydra:runtime.cwd}/data/datasets/"
batch_size: 4  # Smaller for development
```

#### 4. Shared Components (Optional)
For truly shared configurations, create a components directory:
```
configs/preset/components/
├── encoders.yaml      # Common encoder presets
├── decoders.yaml      # Common decoder presets
├── heads.yaml         # Common head presets
└── losses.yaml        # Common loss presets
```

#### Benefits
- **Reduced file count**: From ~20 model-related files to ~5
- **Easier navigation**: All model settings in one file
- **Better maintainability**: Changes localized to model files
- **Clearer dependencies**: No deep composition chains
- **Environment flexibility**: Dataset configs per environment
- **Preserved modularity**: Components can still be overridden

#### Migration Strategy
1. Create new structure alongside existing
2. Update main config files (train.yaml, test.yaml) to use new presets
3. Test configurations thoroughly
4. Remove old preset files once validated
5. Update documentation

#### Potential Drawbacks
- Larger individual files (but more cohesive)
- Less granular reuse (mitigated by base configs and overrides)
- Initial migration effort

This refactor prioritizes simplicity and maintainability over maximal component reuse.
