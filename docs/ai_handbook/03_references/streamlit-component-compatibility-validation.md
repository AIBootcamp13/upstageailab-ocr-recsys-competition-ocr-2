# Component Compatibility Validation System

## Overview

The Streamlit Command Builder now includes comprehensive validation to ensure compatible combinations of model components (encoder, decoder, head, loss) with the selected architecture. This prevents runtime errors like the `state_dict` mismatch you encountered.

## How It Works

### 1. Architecture Metadata System

Each architecture defines its compatible components in YAML files located at:
```
configs/ui_meta/architectures/
├── dbnet.yaml
├── craft.yaml
└── dbnetpp.yaml
```

**Example from `dbnet.yaml`:**
```yaml
ui_metadata:
  compatible_decoders:
    - unet
    - fpn_decoder
    - pan_decoder
  compatible_backbones:
    - resnet18
    - resnet34
    - resnet50
    - mobilenetv3_small_050
    - efficientnet_b0
  compatible_heads:
    - db_head
    - dbpp_head
  compatible_losses:
    - db_loss
```

### 2. Schema-Level Validation

The UI schemas (`ui/apps/command_builder/schemas/`) now include:

- **Required validation**: All components must be selected
- **Help text warnings**: Remind users to match checkpoint configuration
- **Filtered options**: Only compatible components shown in dropdowns

**Key schema features:**
```yaml
- key: decoder
  type: selectbox
  options_source: "models.decoders"
  filter_by_architecture_key: "compatible_decoders"  # Filters options
  metadata_default_key: "default_decoder"             # Sets smart default
  help: "Must match checkpoint's training configuration."
  validation:
    required: true
```

### 3. Runtime Validation

The `ui/utils/ui_validator.py` performs cross-field validation:

```python
# Validates encoder compatibility
compatible_backbones = ui_meta.get("compatible_backbones") or []
if selected_backbone not in compatible_backbones:
    errors.append(f"Encoder '{selected_backbone}' is not compatible...")

# Validates decoder compatibility
compatible_decoders = ui_meta.get("compatible_decoders") or []
if selected_decoder not in compatible_decoders:
    errors.append(f"Decoder '{selected_decoder}' is not compatible...")

# Similar checks for head and loss
```

## Component Compatibility Matrix

### DBNet Architecture
| Component | Compatible Options |
|-----------|-------------------|
| **Encoders** | resnet18, resnet34, resnet50, mobilenetv3_small_050, efficientnet_b0 |
| **Decoders** | unet, fpn_decoder, pan_decoder |
| **Heads** | db_head, dbpp_head |
| **Losses** | db_loss |

### CRAFT Architecture
| Component | Compatible Options |
|-----------|-------------------|
| **Encoders** | vgg16_bn, vgg19_bn |
| **Decoders** | craft_decoder |
| **Heads** | craft_head |
| **Losses** | craft_loss |

### DBNet++ Architecture
| Component | Compatible Options |
|-----------|-------------------|
| **Encoders** | resnet18, resnet50, mobilenetv3_large_100 |
| **Decoders** | dbnetpp_decoder |
| **Heads** | dbnetpp_head |
| **Losses** | dbnetpp_loss |

## Using the Validation System

### For Test/Predict Pages

1. **Select Checkpoint**: Choose your trained model checkpoint
2. **Match Architecture**: Select the same architecture used during training
3. **Match Components**: The UI will:
   - Filter options to show only compatible components
   - Set smart defaults based on architecture metadata
   - Validate your selections before generating commands

**Example workflow:**
```
Checkpoint: ocr_training-dbnet-pan_decoder-resnet34/epoch=28.ckpt
↓
Architecture: dbnet
↓
Encoder: resnet34     ← Must match training
Decoder: pan_decoder  ← Must match training
Head: db_head         ← Must match training
Loss: db_loss         ← Must match training
```

### Error Messages

The system provides clear, actionable error messages:

```
❌ Decoder 'unet' is not compatible with 'dbnet' architecture when checkpoint was trained with 'pan_decoder'.
   Compatible decoders: unet, fpn_decoder, pan_decoder

✓ Select decoder: pan_decoder
```

## Checkpoint Naming Convention

To make component identification easier, use this naming pattern for checkpoints:

```
{exp_name}-{architecture}-{decoder}-{encoder}/checkpoints/{details}.ckpt
```

**Examples:**
- `ocr_training-dbnet-pan_decoder-resnet34/checkpoints/epoch=28.ckpt`
- `receipts_balanced-dbnet-fpn_decoder-resnet34/checkpoints/best.ckpt`
- `docs_high_res-craft-craft_decoder-vgg19_bn/checkpoints/last.ckpt`

This makes it easy to identify which components to select when loading the checkpoint.

## Adding New Components

### 1. Register the Component

Add to appropriate registry in `ocr/models/`:
```python
# ocr/models/decoders/__init__.py
DECODER_REGISTRY = {
    "unet": UNetDecoder,
    "my_new_decoder": MyNewDecoder,  # Add here
}
```

### 2. Update Architecture Metadata

Edit `configs/ui_meta/architectures/{architecture}.yaml`:
```yaml
ui_metadata:
  compatible_decoders:
    - unet
    - fpn_decoder
    - pan_decoder
    - my_new_decoder  # Add here
```

### 3. Test Validation

The UI will automatically:
- Show the new component in filtered dropdowns
- Validate compatibility with selected architecture
- Generate correct Hydra overrides

## Troubleshooting

### State Dict Mismatch Error

**Problem:**
```
RuntimeError: Error(s) in loading state_dict for OCRPLModule:
    Missing key(s): "model.decoder.inners.0.weight"
    Unexpected key(s): "model.decoder.reduce_convs.0.weight"
```

**Solution:**
1. Check your checkpoint path for clues (e.g., `pan_decoder` in the path)
2. Ensure all components match the training configuration
3. Use the validation system to verify compatibility

### Component Not Showing in Dropdown

**Check:**
1. Component is registered in `ocr/models/{component_type}/__init__.py`
2. Component is listed in architecture metadata's `compatible_{type}` list
3. Correct architecture is selected

### Custom Validation Rules

You can add custom validation logic in `ui/utils/ui_validator.py`:

```python
# Example: Prevent specific encoder+decoder combinations
if selected_encoder == "mobilenetv3" and selected_decoder == "pan_decoder":
    errors.append("MobileNetV3 + PAN decoder may have performance issues...")
```

## Benefits

✅ **Prevents Runtime Errors**: Catches incompatible combinations before execution
✅ **Guided Selection**: Filters options to show only valid choices
✅ **Clear Feedback**: Actionable error messages guide users to correct config
✅ **Metadata-Driven**: Easy to maintain and extend
✅ **Checkpoint Safety**: Ensures test/predict configs match training configs

## Future Enhancements

Potential improvements to consider:

1. **Checkpoint Introspection**: Parse checkpoint metadata to auto-detect components
2. **Configuration Presets**: Save and load component combinations
3. **Visual Compatibility Matrix**: Show all valid combinations in a table
4. **Smart Suggestions**: Recommend components based on checkpoint path pattern matching
5. **Version Compatibility**: Track breaking changes in component APIs
