# # **filename: docs/ai_handbook/03_references/guides/ui_inference_compatibility_schema.md**

<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=when_working_on_ui_inference_or_checkpoint_loading -->

## **Overview**

The UI inference compatibility schema (`configs/schemas/ui_inference_compat.yaml`) defines which model configurations are compatible for inference in the Streamlit UI. This ensures that checkpoints can be properly loaded and used for inference with correct architecture matching.

## **Key Concepts**

### **Purpose**

When loading a checkpoint in the UI, the system needs to:
1. Extract encoder, decoder, and head information from the checkpoint
2. Match it against known compatible model families
3. Validate that the checkpoint configuration is supported
4. Provide clear error messages if incompatible

### **Model Family Definition**

Each model family in the schema defines:
- **ID**: Unique identifier for the family
- **Description**: Human-readable description
- **Encoder**: Compatible encoder model names
- **Decoder**: Expected decoder class and channels
- **Head**: Expected head class and input channels

## **Detailed Information**

### **Schema Structure**

```yaml
model_families:
  - id: dbnetpp_resnet18
    description: "DBNet++ architecture trained with ResNet18 backbone and DBPP decoder."
    encoder:
      model_names: ["resnet18"]
    decoder:
      class: ocr.models.decoder.dbpp_decoder.DBPPDecoder
      inner_channels: 256
      output_channels: 128
      in_channels: [64, 128, 256, 512]
    head:
      class: ocr.models.head.db_head.DBHead
      in_channels: 128
```

### **Supported Model Families**

The schema currently defines 9 model families:

| Family ID | Encoder | Decoder | Use Case |
|-----------|---------|---------|----------|
| `dbnet_resnet18_unet64` | ResNet18 | UNet (64ch) | Standard DBNet |
| `dbnet_resnet18_unet256` | ResNet18 | UNet (256ch) | Wider decoder |
| `dbnet_resnet18_pan` | ResNet18 | PAN | Better feature fusion |
| `dbnet_resnet34_pan` | ResNet34 | PAN | Larger backbone |
| `dbnet_mobilenetv3_small_unet256` | MobileNetV3 | UNet (256ch) | Lightweight |
| `dbnetpp_resnet18` | ResNet18 | DBPP (128ch) | DBNet++ improved |
| `dbnetpp_resnet50` | ResNet50/101 | DBPP (128ch) | Larger DBPP |
| `craft_resnet50` | ResNet50 | UNet | CRAFT detection |
| `craft_mobilenetv3_large` | MobileNetV3 | UNet | CRAFT mobile |

### **Encoder Channel Configurations**

The `in_channels` list depends on the encoder architecture:

| Encoder | in_channels |
|---------|-------------|
| resnet18, resnet34 | [64, 128, 256, 512] |
| resnet50, resnet101, resnet152 | [256, 512, 1024, 2048] |
| mobilenetv3_small_050, mobilenetv3 | [8, 16, 24, 288] |
| mobilenetv3_large | [16, 24, 40, 112, 960] |
| efficientnet_b0 | [16, 24, 40, 112, 320] |

## **Examples**

### **Example 1: Adding a New Model Family**

When training with a new encoder-decoder combination:

```yaml
- id: dbnet_efficientnet_b0_unet256
  description: "DBNet with EfficientNet-B0 backbone and UNet decoder"
  encoder:
    model_names: ["efficientnet_b0", "tf_efficientnet_b0"]
  decoder:
    class: ocr.models.decoder.UNet
    inner_channels: 256
    output_channels: 256
    in_channels: [16, 24, 40, 112, 320]
  head:
    class: ocr.models.head.DBHead
    in_channels: 256
```

### **Example 2: Finding Encoder Configuration**

Check your training config to determine the correct channels:

```bash
# View encoder configuration from a checkpoint
grep -A 10 "encoder:" outputs/<exp_name>/.hydra/config.yaml

# Or from the config directly
cat configs/model/encoder/resnet18.yaml
```

### **Example 3: Validating Schema**

Use the validation script to check schema correctness:

```bash
# Run validation
python scripts/validate_ui_schema.py

# Expected output
Found 9 model families
✓ dbnet_resnet18_unet64: 1 encoders
✓ dbnet_resnet18_unet256: 1 encoders
✓ dbnetpp_resnet18: 1 encoders
...
✅ Schema validation passed! All 9 families are valid.
```

### **Example 4: Troubleshooting Checkpoint Loading**

When encountering compatibility errors:

```python
# Check checkpoint configuration
import torch
ckpt = torch.load("path/to/checkpoint.ckpt")
hparams = ckpt["hyper_parameters"]

# Print relevant configuration
print(f"Encoder: {hparams['encoder']}")
print(f"Decoder: {hparams['decoder']}")
print(f"Head: {hparams['head']}")
```

## **Configuration Options**

### **Adding a New Family - Step by Step**

1. **Identify the Configuration**
   - Check encoder model name
   - Check decoder class and channels
   - Check head class and input channels

2. **Create the Family Entry**
   - Add to `configs/schemas/ui_inference_compat.yaml`
   - Use descriptive ID
   - Document the use case

3. **Validate the Schema**
   - Run `python scripts/validate_ui_schema.py`
   - Fix any validation errors

4. **Test in UI**
   - Start inference UI: `python run_ui.py --app inference`
   - Load a checkpoint from the new family
   - Verify it loads without errors

### **Multiple Encoders in One Family**

Some families support multiple encoder variants:

```yaml
- id: dbnetpp_resnet50
  description: "DBNet++ with larger ResNet backbones"
  encoder:
    model_names: ["resnet50", "resnet101"]  # Multiple encoders
  decoder:
    class: ocr.models.decoder.dbpp_decoder.DBPPDecoder
    inner_channels: 256
    output_channels: 128
    in_channels: [256, 512, 1024, 2048]  # Channels for ResNet50/101
```

## **Best Practices**

### **1. Use Descriptive Family IDs**

✅ Good: `dbnetpp_resnet18_pan256`
❌ Bad: `model_v2`

The ID should clearly indicate:
- Architecture (dbnet, dbnetpp, craft)
- Encoder (resnet18, mobilenet)
- Decoder variant if non-standard (pan, unet256)

### **2. Document Non-Standard Configurations**

Add comments for custom channel configurations:

```yaml
- id: dbnet_mobile_optimized
  description: "Custom channel configuration for mobile deployment"
  # Using reduced channels (64) for mobile efficiency
  decoder:
    output_channels: 64
```

### **3. Group Related Families**

Organize families by architecture in the YAML file:
```yaml
# DBNet families
- id: dbnet_resnet18_unet64
- id: dbnet_resnet18_pan

# DBNet++ families
- id: dbnetpp_resnet18
- id: dbnetpp_resnet50

# CRAFT families
- id: craft_resnet50
```

### **4. Keep Sync with Code**

When adding new decoder/head classes to the codebase:
1. Add them to the schema immediately
2. Test with a checkpoint
3. Update this documentation

### **5. Test Compatibility First**

Always test new families with actual checkpoints before committing:

```bash
# Train a checkpoint
export EXPERIMENT_TAG="test_new_family"
python runners/train.py preset=new_config

# Test loading in UI
python run_ui.py --app inference
# Select the new checkpoint and verify loading
```

## **Troubleshooting**

### **Error: "No compatibility schema found for encoder 'X'"**

**Cause**: The encoder name from your checkpoint isn't in any model family.

**Solution**:
1. Check checkpoint encoder name: `grep -A 5 "encoder:" outputs/<exp_name>/.hydra/config.yaml`
2. Add family entry with that encoder name
3. Restart the UI

### **Error: "Decoder output channels mismatch"**

**Cause**: The checkpoint's decoder outputs don't match the schema.

**Solution**:
1. Check actual decoder configuration in checkpoint
2. Update schema's `output_channels` to match
3. Or create a new family variant

### **Error: "Head input channels mismatch"**

**Cause**: The checkpoint's head expects different inputs than specified.

**Solution**:
1. Verify head configuration in checkpoint
2. Update schema `head.in_channels` to match
3. Or create a new family variant

### **Schema Validation Fails**

**Cause**: YAML syntax error or missing required fields.

**Solution**:
1. Check YAML syntax with: `python -c "import yaml; yaml.safe_load(open('configs/schemas/ui_inference_compat.yaml'))"`
2. Ensure all required fields present: `id`, `encoder`, `decoder`, `head`
3. Run validation script for detailed error messages

## **Related References**

- [Checkpoint Naming Scheme](../architecture/07_checkpoint_naming_scheme.md) - Checkpoint directory and file naming
- [Streamlit Component Compatibility Validation](./streamlit-component-compatibility-validation.md) - UI component validation
- [UI Architecture](../architecture/05_ui_architecture.md) - Overall UI architecture

---

**Last Updated**: 2025-10-15
**Owner**: frontend
**Related Files**:
- `configs/schemas/ui_inference_compat.yaml` - The schema file
- `ui/apps/inference/services/schema_validator.py` - Validation logic
- `ui/apps/inference/services/checkpoint_catalog.py` - Checkpoint discovery
- `scripts/validate_ui_schema.py` - Schema validation script
