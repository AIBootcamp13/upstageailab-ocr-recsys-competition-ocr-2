# Checkpoint Loading Protocol

<!-- ai_cue:priority=critical -->
<!-- ai_cue:use_when=["checkpoint_loading", "load_state_dict_errors", "model_compatibility", "debugging_inference"] -->

**Version**: 1.0
**Last Updated**: 2025-10-18
**Status**: Active

## ⚠️ CRITICAL: Read This First

**This protocol prevents the most common checkpoint loading errors that cause hours of debugging.**

### Common Failure Modes

1. **KeyError: 'state_dict'**: Checkpoint uses 'model_state_dict' or is a raw state dict
2. **RuntimeError: Error(s) in loading state_dict**: Key prefix mismatch ('model.' vs no prefix)
3. **AttributeError: 'NoneType' has no attribute 'shape'**: Missing weight in state dict
4. **Signature mismatch**: Decoder/head architecture changed between training and inference
5. **Silent failures**: Model loads but predictions are garbage (wrong architecture)

### Quick Reference Decision Tree

```
Loading a checkpoint?
├─ For training/fine-tuning?
│  └─ Use Lightning's automatic loading (checkpoint_path in Trainer)
│
├─ For inference in UI?
│  └─ Use InferenceEngine.load_checkpoint() (handles all validation)
│
├─ For catalog building?
│  └─ Use new V2 modules:
│     1. Try load_metadata() first (YAML, fast)
│     2. Fall back to get_metadata_from_wandb() (API, medium)
│     3. Last resort: load_checkpoint() with validation (slow)
│
└─ For custom script?
   └─ Use validate_checkpoint_structure() before torch.load()
```

---

## 1. Architecture Overview

### Checkpoint Structure Hierarchy

```
Raw Checkpoint File (.ckpt)
    ↓ (torch.load)
Checkpoint Dict
    ├─ state_dict OR model_state_dict OR <raw weights>
    ├─ epoch (optional)
    ├─ global_step (optional)
    ├─ cleval_metrics (optional)
    ├─ callbacks (optional)
    └─ hyper_parameters (optional)
        ↓ (validate_checkpoint_structure)
Validated StateDictStructure
    ├─ has_wrapper: bool
    ├─ wrapper_key: "state_dict" | "model_state_dict" | None
    ├─ has_model_prefix: bool
    ├─ decoder_pattern: DecoderKeyPattern
    └─ head_pattern: HeadKeyPattern
        ↓ (extract signatures)
Typed Metadata (CheckpointMetadataV1)
```

### Module Responsibilities

| Module | Purpose | When to Use |
|--------|---------|-------------|
| [state_dict_models.py](../../../ui/apps/inference/services/checkpoint/state_dict_models.py) | Pydantic validation for state dict structure | **Always** before accessing state_dict keys |
| [inference_engine.py](../../../ui/apps/inference/services/checkpoint/inference_engine.py) | State dict analysis and signature extraction | Legacy checkpoints without metadata |
| [metadata_loader.py](../../../ui/apps/inference/services/checkpoint/metadata_loader.py) | YAML metadata loading | Primary path (fastest) |
| [wandb_client.py](../../../ui/apps/inference/services/checkpoint/wandb_client.py) | Wandb API fallback | When YAML missing but run ID available |

---

## 2. Loading Patterns (By Use Case)

### Pattern A: Training/Fine-tuning (Lightning)

**Use Case**: Resume training, fine-tune model

```python
from lightning.pytorch import Trainer

# ✅ CORRECT: Lightning handles everything
trainer = Trainer(
    ...
)
trainer.fit(
    model,
    ckpt_path="outputs/exp/checkpoints/epoch=10.ckpt",
)
```

**DO NOT**:
- ❌ Manually load with torch.load() for training
- ❌ Modify state dict keys manually
- ❌ Load checkpoint in model's __init__()

**Lightning automatically handles**:
- State dict unwrapping
- Optimizer state restoration
- LR scheduler state restoration
- Callback state restoration

---

### Pattern B: Inference (UI/API)

**Use Case**: Load checkpoint for inference in UI or API

```python
from ui.apps.inference.services.checkpoint import build_catalog
from ui.utils.inference.engine import InferenceEngine

# Step 1: Build catalog (validates all checkpoints)
catalog = build_catalog(outputs_dir=Path("outputs"))

# Step 2: Select checkpoint entry
entry = catalog.entries[0]

# Step 3: Load for inference (fully validated)
engine = InferenceEngine.from_checkpoint(
    checkpoint_path=entry.checkpoint_path,
    config_path=entry.config_path,
    device="cuda",
)

# ✅ Safe to use
predictions = engine.predict(images)
```

**Critical AI Cues**:
- 🔴 **NEVER** skip catalog validation step
- 🔴 **NEVER** use torch.load() directly in inference
- 🟡 **ALWAYS** check entry.has_metadata before assuming config accuracy
- 🟢 **USE** InferenceEngine.from_checkpoint() for all inference loading

---

### Pattern C: Catalog Building (Metadata Extraction)

**Use Case**: Build checkpoint catalog for UI dropdown

```python
from pathlib import Path
from ui.apps.inference.services.checkpoint import (
    build_catalog,
    load_metadata,
    get_wandb_client,
    extract_run_id_from_checkpoint,
)
from ui.apps.inference.services.checkpoint.state_dict_models import (
    validate_checkpoint_structure,
)

checkpoint_path = Path("outputs/exp/checkpoints/epoch=10.ckpt")

# Tier 1: Try YAML metadata (FAST: ~5-10ms)
metadata = load_metadata(checkpoint_path)
if metadata:
    # ✅ Validated, use directly
    architecture = metadata.model.architecture
    hmean = metadata.metrics.hmean

# Tier 2: Try Wandb API (MEDIUM: ~100-500ms, cached)
if metadata is None:
    run_id = extract_run_id_from_checkpoint(checkpoint_path)
    if run_id:
        client = get_wandb_client()
        metadata = client.get_metadata_from_wandb(run_id, checkpoint_path)

# Tier 3: Legacy inference (SLOW: ~2-5s)
if metadata is None:
    import torch

    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # ⚠️ CRITICAL: Validate structure BEFORE accessing keys
    structure = validate_checkpoint_structure(checkpoint_data)

    # ✅ Safe access with validation
    raw_state_dict = structure.get_raw_state_dict(checkpoint_data)

    # Use decoder/head patterns for architecture detection
    decoder_type = structure.decoder_pattern.decoder_type if structure.decoder_pattern else "unknown"
    head_type = structure.head_pattern.head_type if structure.head_pattern else "unknown"
```

**Critical AI Cues**:
- 🔴 **ALWAYS** try YAML → Wandb → torch.load() in that order
- 🔴 **NEVER** skip `validate_checkpoint_structure()` when using torch.load()
- 🟡 **LOG** which tier was used for performance monitoring
- 🟢 **CACHE** Wandb responses (already handled by WandbClient)

---

### Pattern D: State Dict Inspection (Advanced)

**Use Case**: Debug checkpoint loading errors, inspect model architecture

```python
import torch
from ui.apps.inference.services.checkpoint.state_dict_models import (
    validate_checkpoint_structure,
    safe_get_shape,
)

checkpoint_path = Path("outputs/exp/checkpoints/epoch=10.ckpt")

# Load checkpoint
checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# Validate structure
try:
    structure = validate_checkpoint_structure(checkpoint_data)
except ValueError as exc:
    print(f"❌ Invalid checkpoint: {exc}")
    # Handle error appropriately
    raise

# Access validated information
print(f"Has wrapper: {structure.has_wrapper}")
print(f"Wrapper key: {structure.wrapper_key}")
print(f"Has model prefix: {structure.has_model_prefix}")

# Detect architecture
if structure.decoder_pattern:
    print(f"Decoder type: {structure.decoder_pattern.decoder_type}")
    print(f"Decoder prefix: {structure.decoder_pattern.prefix}")

if structure.head_pattern:
    print(f"Head type: {structure.head_pattern.head_type}")
    print(f"Head prefix: {structure.head_pattern.prefix}")

# Safe weight access
raw_dict = structure.get_raw_state_dict(checkpoint_data)
prefix = "model." if structure.has_model_prefix else ""

# Get weight shape safely
weight_key = f"{prefix}encoder.model.layer1.0.conv1.weight"
weight = raw_dict.get(weight_key)

if weight:
    shape = safe_get_shape(weight)
    if shape:
        print(f"Weight shape: {shape.dims}")
        print(f"Out channels: {shape.out_channels}")
        print(f"In channels: {shape.in_channels}")
```

**Critical AI Cues**:
- 🔴 **NEVER** access state_dict keys without validating structure first
- 🔴 **NEVER** assume key existence (use .get() with None default)
- 🟡 **USE** safe_get_shape() instead of weight.shape directly
- 🟢 **LOG** validation errors before re-raising

---

## 3. State Dict Key Patterns (Reference)

### Encoder Patterns

```python
# ResNet
"encoder.model.layer1.0.conv1.weight"  # Shape: [64, 64, 3, 3]
"encoder.model.layer2.0.conv1.weight"  # Shape: [128, 64, 3, 3]
"encoder.model.layer3.0.conv1.weight"  # Shape: [256, 128, 3, 3]
"encoder.model.layer4.0.conv1.weight"  # Shape: [512, 256, 3, 3]

# MobileNetV3
"encoder.model.conv_stem.weight"       # Shape: [16, 3, 3, 3] (small_050)
                                       # Shape: [24, 3, 3, 3] (small_075)

# EfficientNet
"encoder.model.features.0.0.weight"    # Shape: [32, 3, 3, 3] (b0)
                                       # Shape: [40, 3, 3, 3] (b3)
```

### Decoder Patterns

```python
# PAN Decoder
"decoder.bottom_up.0.depthwise.weight"   # Shape: [256, 1, 3, 3]
"decoder.bottom_up.0.pointwise.weight"   # Shape: [256, 256, 1, 1]
"decoder.lateral_convs.0.0.weight"       # Shape: [256, 2048, 1, 1]
"decoder.lateral_convs.1.0.weight"       # Shape: [256, 1024, 1, 1]

# FPN Decoder
"decoder.fusion.0.weight"                # Shape: [256, 1024, 1, 1]
"decoder.lateral_convs.0.0.weight"       # Shape: [256, 2048, 1, 1]

# UNet Decoder
"decoder.inners.0.weight"                # Shape: [256, 512, 1, 1]
"decoder.outers.0.0.weight"              # Shape: [64, 256, 3, 3]
```

### Head Patterns

```python
# DB Head
"head.binarize.0.weight"                 # Shape: [1, 256, 3, 3]

# CRAFT Head
"head.craft.0.weight"                    # Shape: [2, 256, 3, 3]
```

### Prefix Variations

```python
# With 'model.' prefix (standard Lightning)
"model.encoder.model.layer1.0.weight"

# Without 'model.' prefix (some custom checkpoints)
"encoder.model.layer1.0.weight"
```

---

## 4. Common Error Patterns and Solutions

### Error 1: KeyError: 'state_dict'

**Symptom**:
```python
KeyError: 'state_dict'
```

**Cause**: Checkpoint uses 'model_state_dict' or is a raw state dict

**Solution**:
```python
# ❌ BAD
state_dict = checkpoint_data["state_dict"]

# ✅ GOOD
structure = validate_checkpoint_structure(checkpoint_data)
state_dict = structure.get_raw_state_dict(checkpoint_data)
```

---

### Error 2: RuntimeError: Error(s) in loading state_dict

**Symptom**:
```python
RuntimeError: Error(s) in loading state_dict for MyModel:
size mismatch for encoder.model.layer1.0.weight:
copying a param with shape torch.Size([64, 64, 3, 3]) from checkpoint,
the shape in current model is torch.Size([64, 3, 7, 7]).
```

**Cause**: Model architecture mismatch (wrong encoder, decoder, or head)

**Solution**:
```python
# ✅ Validate architecture before loading
structure = validate_checkpoint_structure(checkpoint_data)

# Check decoder type
if structure.decoder_pattern.decoder_type != expected_decoder:
    raise ValueError(
        f"Checkpoint has {structure.decoder_pattern.decoder_type}, "
        f"but model expects {expected_decoder}"
    )

# Check head type
if structure.head_pattern.head_type != expected_head:
    raise ValueError(
        f"Checkpoint has {structure.head_pattern.head_type}, "
        f"but model expects {expected_head}"
    )
```

---

### Error 3: AttributeError: 'NoneType' has no attribute 'shape'

**Symptom**:
```python
AttributeError: 'NoneType' object has no attribute 'shape'
```

**Cause**: Missing weight in state dict

**Solution**:
```python
# ❌ BAD
weight = state_dict[key]
shape = weight.shape  # Crashes if key missing

# ✅ GOOD
weight = state_dict.get(key)
if weight:
    shape = safe_get_shape(weight)
    if shape:
        # Use shape.dims, shape.out_channels, etc.
        pass
```

---

### Error 4: Silent Failure (Wrong Architecture)

**Symptom**: Model loads successfully but predictions are garbage

**Cause**: Checkpoint architecture doesn't match config (e.g., loading PAN weights into FPN model)

**Solution**:
```python
# ✅ Validate before loading
structure = validate_checkpoint_structure(checkpoint_data)

# Log architecture mismatch warnings
if structure.decoder_pattern.decoder_type != config["model"]["decoder"]["name"]:
    LOGGER.warning(
        "Decoder mismatch: checkpoint has %s, config specifies %s",
        structure.decoder_pattern.decoder_type,
        config["model"]["decoder"]["name"],
    )
```

---

## 5. DO NOTs (Critical Anti-Patterns)

### 🔴 NEVER: Modify state_dict keys manually

```python
# ❌ EXTREMELY DANGEROUS
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace("old_prefix", "new_prefix")
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)
```

**Why**: Breaks reproducibility, hard to debug, causes silent failures

**Instead**: Update model architecture to match checkpoint or train new checkpoint

---

### 🔴 NEVER: Skip validation when using torch.load()

```python
# ❌ BRITTLE
checkpoint_data = torch.load(path)
state_dict = checkpoint_data["state_dict"]  # May crash
```

**Why**: Assumes specific structure, fails on edge cases

**Instead**: Always use `validate_checkpoint_structure()`

---

### 🔴 NEVER: Assume weight shapes without checking

```python
# ❌ CRASHES ON MISSING WEIGHTS
weight = state_dict["encoder.layer1.weight"]
out_channels = weight.shape[0]
```

**Why**: Weight may be None, may not have .shape

**Instead**: Use `safe_get_shape()` utility

---

### 🔴 NEVER: Load checkpoints in model __init__()

```python
# ❌ ANTI-PATTERN
class MyModel(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        checkpoint = torch.load(checkpoint_path)  # Bad!
        self.load_state_dict(checkpoint["state_dict"])
```

**Why**: Violates separation of concerns, hard to test, breaks Lightning

**Instead**: Load checkpoints externally (Lightning Trainer or InferenceEngine)

---

## 6. Validation Checklist

Before committing checkpoint loading code:

- [ ] Used `validate_checkpoint_structure()` before accessing state_dict
- [ ] Checked both `has_wrapper` and `wrapper_key` before unwrapping
- [ ] Used `safe_get_shape()` for all weight shape access
- [ ] Logged decoder/head pattern mismatches as warnings
- [ ] Tested with all 3 checkpoint formats:
  - [ ] Standard (state_dict wrapper)
  - [ ] Alternative (model_state_dict wrapper)
  - [ ] Raw (no wrapper)
- [ ] Tested with both prefix patterns:
  - [ ] With 'model.' prefix
  - [ ] Without 'model.' prefix
- [ ] Added error handling with clear messages
- [ ] Documented any assumptions in comments

---

## 7. Testing Patterns

### Unit Test Template

```python
import pytest
import torch
from pathlib import Path
from ui.apps.inference.services.checkpoint.state_dict_models import (
    validate_checkpoint_structure,
)

class TestCheckpointLoading:
    """Test checkpoint loading with various formats."""

    @pytest.fixture
    def mock_state_dict(self):
        """Create mock state dict with realistic structure."""
        return {
            "model.encoder.model.layer1.0.weight": torch.randn(64, 64, 3, 3),
            "model.decoder.bottom_up.0.weight": torch.randn(256, 256, 3, 3),
            "model.head.binarize.0.weight": torch.randn(1, 256, 3, 3),
        }

    def test_standard_wrapper(self, mock_state_dict):
        """Test checkpoint with 'state_dict' wrapper."""
        checkpoint_data = {
            "state_dict": mock_state_dict,
            "epoch": 10,
        }

        structure = validate_checkpoint_structure(checkpoint_data)

        assert structure.has_wrapper is True
        assert structure.wrapper_key == "state_dict"
        assert structure.has_model_prefix is True

    def test_alternative_wrapper(self, mock_state_dict):
        """Test checkpoint with 'model_state_dict' wrapper."""
        checkpoint_data = {
            "model_state_dict": mock_state_dict,
            "epoch": 10,
        }

        structure = validate_checkpoint_structure(checkpoint_data)

        assert structure.has_wrapper is True
        assert structure.wrapper_key == "model_state_dict"

    def test_raw_state_dict(self, mock_state_dict):
        """Test raw state dict (no wrapper)."""
        structure = validate_checkpoint_structure(mock_state_dict)

        assert structure.has_wrapper is False
        assert structure.wrapper_key is None

    def test_architecture_detection(self, mock_state_dict):
        """Test decoder and head architecture detection."""
        checkpoint_data = {"state_dict": mock_state_dict}
        structure = validate_checkpoint_structure(checkpoint_data)

        assert structure.decoder_pattern.decoder_type == "pan_decoder"
        assert structure.head_pattern.head_type == "db_head"
```

---

## 8. Migration Guide (Legacy to V2)

### Old Pattern (Brittle)

```python
# ❌ OLD: checkpoint_catalog.py
def _load_checkpoint(path):
    checkpoint_data = torch.load(path, map_location="cpu")
    state_dict = checkpoint_data.get("state_dict") or checkpoint_data.get("model_state_dict")
    if state_dict is None:
        state_dict = checkpoint_data  # Assumes it's raw
    return state_dict
```

### New Pattern (Validated)

```python
# ✅ NEW: Using V2 modules
from ui.apps.inference.services.checkpoint.state_dict_models import validate_checkpoint_structure

def load_checkpoint_validated(path):
    checkpoint_data = torch.load(path, map_location="cpu", weights_only=False)

    # Validate structure
    structure = validate_checkpoint_structure(checkpoint_data)

    # Get raw state dict with validation
    state_dict = structure.get_raw_state_dict(checkpoint_data)

    return state_dict, structure
```

---

## 9. References

- [State Dict Models](../../../ui/apps/inference/services/checkpoint/state_dict_models.py)
- [Inference Engine](../../../ui/apps/inference/services/checkpoint/inference_engine.py)
- [Checkpoint Catalog V2 Design](../../03_references/architecture/checkpoint_catalog_v2_design.md)
- [PyTorch Checkpoint Documentation](https://pytorch.org/docs/stable/notes/serialization.html)

---

## Appendix: Full Example Script

See [scripts/examples/safe_checkpoint_loading.py](../../../scripts/examples/safe_checkpoint_loading.py) for a complete working example.

---

**Last Updated**: 2025-10-18
**Maintained By**: AI Handbook Team
**Questions**: See [AI Handbook Index](../../index.md)
