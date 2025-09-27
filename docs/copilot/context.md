# Core Project Context

## 🚨 ENVIRONMENT SETUP - READ THIS FIRST

### Required Environment
- **Python**: >= 3.10
- **Package Manager**: UV (not pip/conda/poetry)
- **Virtual Environment**: `.venv/` in workspace root
- **VS Code**: Auto-configured to use `./.venv/bin/python`

### Quick Start Commands
```bash
# Install all dependencies
uv sync --group dev

# Run tests
uv run pytest tests/ -v

# Format code
uv run black . && uv run isort .

# Lint code
uv run flake8 .
```

### VS Code Integration
- Python interpreter automatically set to workspace `.venv`
- Terminal automatically activates virtual environment
- All commands should use `uv run` prefix

---

## Architecture Overview

### Current Structure (Pre-Refactor)
```
ocr/
├── models/
│   ├── architecture.py      # Main OCRModel class
│   ├── encoder/            # Backbone encoders (TIMM)
│   ├── decoder/            # Feature decoders (U-Net)
│   ├── head/               # Task heads (DB head)
│   └── loss/               # Loss functions
├── datasets/               # Data loading and transforms
├── lightning_modules/      # PyTorch Lightning training
├── metrics/               # CLEval evaluation metrics
└── utils/                 # Utilities
```

### Target Structure (Post-Refactor)
```
src/ocr_framework/
├── core/
│   ├── base_model.py       # Abstract base classes
│   ├── registry.py         # Architecture registry
│   └── utils.py
├── architectures/
│   ├── dbnet/             # DBNet implementation
│   ├── east/              # EAST implementation
│   └── custom/            # Experiment architectures
├── datasets/
├── training/
├── evaluation/
└── utils/
```

## Key Components

### OCRModel (Main Model)
- **Purpose**: Orchestrates encoder → decoder → head → loss pipeline
- **Config**: Uses Hydra instantiate for all components
- **Methods**:
  - `forward()`: Main inference pipeline
  - `get_optimizers()`: Returns optimizer and scheduler
  - `get_polygons_from_maps()`: Post-processing for polygons

### Decoder Options
- `unet` *(default)* – strong baseline with skip connections, good balance of quality and speed.
- `fpn_decoder` – lightweight feature pyramid tuned for receipts; fewer parameters, stable on CPU inference.
- `pan_decoder` – path aggregation emphasising low-level detail; boosts recall on densely packed text.
- `dbnetpp_decoder` – bi-directional pyramid used in DBNet++; highest accuracy, higher memory cost.

### Factory Functions
```python
# Pattern used throughout codebase
def get_encoder_by_cfg(config):
    return instantiate(config)

def get_decoder_by_cfg(config):
    return instantiate(config)

def get_head_by_cfg(config):
    return instantiate(config)

def get_loss_by_cfg(config):
    return instantiate(config)
```

### Configuration Pattern
```yaml
# configs/preset/models/model_example.yaml
defaults:
  - /preset/models/decoder/unet
  - /preset/models/encoder/timm_backbone
  - /preset/models/head/db_head
  - /preset/models/loss/db_loss

model:
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
  scheduler:
    _target_: torch.optim.lr_scheduler.StepLR
    step_size: 100
```

## Development Patterns

### 1. Component Instantiation
```python
from hydra.utils import instantiate

# In config
component:
  _target_: ocr.models.encoder.TimmBackbone
  backbone: resnet50
  pretrained: true

# In code
encoder = instantiate(config.component)
```

### 2. Abstract Base Classes (Future)
```python
from abc import ABC, abstractmethod

class BaseEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```

### 3. Registry Pattern (Future)
```python
class ArchitectureRegistry:
    def register(self, name: str, model_class: Type):
        self._architectures[name] = model_class

    def get_architecture(self, name: str):
        return self._architectures[name]
```

### 4. Benchmarking Toolkit
- `scripts/decoder_benchmark.py` orchestrates sequential decoder evaluations. It composes `configs/benchmark/decoder.yaml`, applies per-decoder overrides, runs Lightning training/evaluation, and emits a CSV summary in `outputs/decoder_benchmark/`.
- Configure decoder candidates inside `configs/benchmark/decoder.yaml` (`benchmark.decoders`). Each entry accepts `key`, `name` (registry identifier), optional `params`, and optional per-run `trainer_overrides`.
- New decoder presets are available under `configs/preset/models/decoder/{unet,fpn,pan}.yaml`. The benchmark config defaults to running the UNet, FPN, and PAN options registered in `ocr/models/architectures/shared_decoders.py`.
- Quick dry run:
    ```bash
    uv run python scripts/decoder_benchmark.py benchmark.skip_training=true benchmark.limit_val_batches=0.05
    ```
    Use full training by omitting `skip_training` and adjusting `benchmark.trainer_overrides.max_epochs`.

## Common Imports
```python
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
```

## Error Handling Patterns
```python
try:
    result = some_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    raise
```

## Testing Patterns
```python
import pytest
from unittest.mock import Mock, patch

def test_component_initialization():
    with patch('module.instantiate') as mock_instantiate:
        # Test instantiation logic
        pass
```

## File Path Handling
```python
from pathlib import Path

# Always use Path objects
config_path = Path("configs") / "model.yaml"
data_path = Path("data") / "images" / "train"

# Check existence
if config_path.exists():
    # Load config
    pass
```</content>
<parameter name="filePath">/home/vscode/workspace/upstage-receipt-text-detection-dbnet-baseline/docs/copilot/context.md
