# Quick Fix: Checkpoint State Dict Mismatch

## Your Error

```
RuntimeError: Error(s) in loading state_dict for OCRPLModule:
    Missing key(s) in state_dict: "model.decoder.inners.0.weight", ...
    Unexpected key(s) in state_dict: "model.decoder.reduce_convs.0.0.weight", ...
```

## The Problem

Your checkpoint was trained with **different model components** than what you're trying to load it with.

Based on your checkpoint path: `ocr_training-dbnet-pan_decoder-resnet34/checkpoints/resnet34_bs4_epoch_epoch=28_step_step=23722.ckpt`

The checkpoint contains:
- ✓ Architecture: **dbnet**
- ✓ Decoder: **pan_decoder** (has `reduce_convs`, `top_down_smooth`, `bottom_up`)
- ✓ Encoder: **resnet34**

But you tried to load it with:
- ✓ Architecture: dbnet
- ✗ Decoder: **unet** (has `inners`, `outers` - different structure!)
- ✗ Encoder: **resnet18** (different from resnet34!)

## The Solution

### Using Streamlit UI (Recommended)

1. Open the Streamlit Command Builder Test/Predict page
2. Select your checkpoint: `ocr_training-dbnet-pan_decoder-resnet34/.../epoch=28.ckpt`
3. Configure components to **match the checkpoint**:
   - Architecture: **dbnet**
   - Encoder: **resnet34** ← Match checkpoint!
   - Decoder: **pan_decoder** ← Match checkpoint!
   - Head: **db_head**
   - Loss: **db_loss**

The UI will now:
- ✓ Filter options to show only compatible components
- ✓ Validate your selection before generating the command
- ✓ Prevent state_dict mismatch errors

### Using Terminal (Manual)

```bash
uv run python runners/test.py \
  exp_name=pan_decoder_test \
  'checkpoint_path="outputs/ocr_training-dbnet-pan_decoder-resnet34/checkpoints/resnet34_bs4_epoch_epoch=28_step_step=23722.ckpt"' \
  model.architecture_name=dbnet \
  model/architectures=dbnet \
  model.encoder.model_name=resnet34 \
  model.component_overrides.decoder.name=pan_decoder \
  model.component_overrides.head.name=db_head \
  model.component_overrides.loss.name=db_loss
```

Note the key changes:
- ✓ `checkpoint_path` properly quoted (our earlier fix!)
- ✓ `encoder.model_name=resnet34` (not resnet18)
- ✓ `decoder.name=pan_decoder` (not unet)

## How to Identify Checkpoint Components

### From Checkpoint Path (Best Practice)

Use this naming pattern for checkpoints:
```
{exp_name}-{architecture}-{decoder}-{encoder}/checkpoints/{details}.ckpt
```

Examples:
- `ocr_training-dbnet-pan_decoder-resnet34/` → dbnet, pan_decoder, resnet34
- `model-dbnet-unet-resnet18/` → dbnet, unet, resnet18
- `craft_training-craft-craft_decoder-vgg16_bn/` → craft, craft_decoder, vgg16_bn

### From Training Logs

Check your training logs for lines like:
```
Architecture: dbnet
Encoder: resnet34
Decoder: pan_decoder
Head: db_head
Loss: db_loss
```

### From WandB/MLflow

Check your experiment tracking dashboard for the hyperparameters used.

## Component Compatibility Reference

### DBNet Architecture

| Component | Options |
|-----------|---------|
| **Encoders** | resnet18, resnet34, resnet50, mobilenetv3_small_050, efficientnet_b0 |
| **Decoders** | unet, fpn_decoder, pan_decoder |
| **Heads** | db_head, dbpp_head |
| **Losses** | db_loss |

**Common Combinations:**
- Balanced: resnet34 + fpn_decoder + db_head
- High Recall: resnet34 + pan_decoder + db_head
- Fast: mobilenetv3_small_050 + unet + db_head

### CRAFT Architecture

| Component | Options |
|-----------|---------|
| **Encoders** | vgg16_bn, vgg19_bn |
| **Decoders** | craft_decoder |
| **Heads** | craft_head |
| **Losses** | craft_loss |

**Only Combination:** vgg16_bn/vgg19_bn + craft_decoder + craft_head + craft_loss

### DBNet++ Architecture

| Component | Options |
|-----------|---------|
| **Encoders** | resnet18, resnet50, mobilenetv3_large_100 |
| **Decoders** | dbnetpp_decoder |
| **Heads** | dbnetpp_head |
| **Losses** | dbnetpp_loss |

**Only Combination:** resnet18/50/mobilenet + dbnetpp_decoder + dbnetpp_head + dbnetpp_loss

## Prevention Tips

### 1. Name Checkpoints Clearly

Instead of:
```
outputs/training_run_1/checkpoints/epoch=28.ckpt
```

Use:
```
outputs/dbnet-pan_decoder-resnet34/checkpoints/epoch=28.ckpt
```

### 2. Use Streamlit UI

The UI now prevents these errors by:
- Filtering incompatible options
- Validating before command generation
- Showing helpful error messages

### 3. Document Training Configs

Save a README in your checkpoint directory:
```
outputs/my_model/
├── checkpoints/
│   └── best.ckpt
└── README.md  ← Document components here
```

Example README.md:
```markdown
# Model: my_model

## Components
- Architecture: dbnet
- Encoder: resnet34
- Decoder: pan_decoder
- Head: db_head
- Loss: db_loss

## Training Config
- Batch size: 4
- Learning rate: 0.001
- Optimizer: adam
```

## Still Having Issues?

If you see state_dict errors after matching all components:

1. **Check architecture version**: Was the checkpoint trained with an older code version?
2. **Verify component registrations**: Ensure all components are properly registered
3. **Try strict=False**: As a last resort, you can load with `strict=False` (but this may cause issues)

For help, share:
- Full checkpoint path
- Complete error message
- Components you're trying to use
