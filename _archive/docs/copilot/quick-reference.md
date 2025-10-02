# Configuration & Validation Policy

- **Hydra** is the single source of truth for all configuration composition and runtime parameters. All config schemas are maintained in `/configs/schemas/` and must be kept up to date for train, predict, test, and model configs.
- **UI schemas** in `/ui/schemas/` are strictly for Streamlit/Pydantic validation and command building. They must not be used for config management or composition.
- **Pydantic** is only allowed for validating user input in the UI and (optionally) validating the final Hydra config dict at runtime. See `docs/development/pydantic-policy.md` for details.

- Never use Pydantic for config composition or management. Always use Hydra for config overrides and composition.

## Validation vs Configuration


# Quick Reference Guide

## Common Patterns and Snippets

### 1. Hydra Configuration
```python
# config.yaml
component:
  _target_: ocr.models.encoder.TimmBackbone
  backbone: resnet50
  pretrained: true
  features_only: true
  out_indices: [2, 3, 4, 5]

# Usage in code
from hydra.utils import instantiate
from omegaconf import DictConfig

def create_component(cfg: DictConfig):
    return instantiate(cfg.component)
```

### 2. PyTorch Lightning Module
```python
import pytorch_lightning as pl
import torch

class OCRLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = instantiate(cfg.model)
        self.metric = instantiate(cfg.metric)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch['image'], gt=batch['polygons'])
        self.log('train_loss', outputs['loss'])
        return outputs['loss']

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch['image'], gt=batch['polygons'])
        preds = self.model.get_polygons_from_maps(batch['polygons'], outputs['maps'])
        metric_result = self.metric(preds, batch['polygons'])
        self.log('val_f1', metric_result['f1'])
        return metric_result

    def configure_optimizers(self):
        return self.model.get_optimizers()
```

### 3. Dataset Implementation
```python
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image

class OCRDataset(Dataset):
    def __init__(self, image_dir: str, annotation_file: str, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        # Filter valid images
        self.valid_images = [
            img for img in self.annotations['images'].keys()
            if (self.image_dir / img).exists()
        ]

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_name = self.valid_images[idx]
        image = Image.open(self.image_dir / img_name).convert('RGB')

        # Get polygons
        polygons = self.annotations['images'][img_name].get('words', {})
        polygons = [np.array(list(word['points'].values())) for word in polygons.values()]

        if self.transform:
            transformed = self.transform(image=np.array(image), polygons=polygons)
            return {
                'image': transformed['image'],
                'polygons': transformed['polygons'],
                'image_name': img_name
            }

        return {
            'image': torch.from_numpy(np.array(image)).permute(2, 0, 1).float(),
            'polygons': polygons,
            'image_name': img_name
        }
```

### 4. Model Architecture Pattern
```python
import torch.nn as nn
from typing import Dict, Any

class DBNetHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        prob_map = torch.sigmoid(self.conv3(x))
        return {'maps': prob_map}

    def get_polygons_from_maps(self, gt, pred):
        # Post-processing logic here
        return polygons
```

### 5. Factory Functions
```python
# encoder/__init__.py
from .timm_backbone import TimmBackbone
from hydra.utils import instantiate

def get_encoder_by_cfg(config):
    """Factory function for encoders"""
    return instantiate(config)

# Usage in config
encoder:
  _target_: ocr.models.encoder.TimmBackbone
  backbone: resnet50
  pretrained: true
```

### 6. Loss Function Pattern
```python
import torch
import torch.nn as nn

class DBLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        # pred: {'maps': probability_maps}
        # target: ground truth maps

        prob_maps = pred['maps']
        gt_maps = target

        bce = self.bce_loss(prob_maps, gt_maps)
        dice = self.dice_loss(prob_maps, gt_maps)

        total_loss = self.alpha * bce + self.beta * dice

        return total_loss, {
            'bce_loss': bce.item(),
            'dice_loss': dice.item(),
            'total_loss': total_loss.item()
        }
```

### 7. Configuration Validation
```python
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Any

def validate_config(cfg: DictConfig) -> List[str]:
    """Validate configuration and return error messages"""
    errors = []

    # Check required fields
    required_fields = ['model', 'data', 'training']
    for field in required_fields:
        if field not in cfg:
            errors.append(f"Missing required field: {field}")

    # Validate model config
    if 'model' in cfg:
        if 'encoder' not in cfg.model:
            errors.append("Model config missing encoder")
        if 'decoder' not in cfg.model:
            errors.append("Model config missing decoder")

    return errors

def setup_config() -> DictConfig:
    """Load and validate configuration"""
    cfg = OmegaConf.load('config.yaml')
    errors = validate_config(cfg)

    if errors:
        raise ValueError(f"Configuration errors: {errors}")

    return cfg
```

### 8. Logging and Debugging
```python
import logging
from rich.console import Console
from rich.logging import RichHandler
from icecream import ic

# Setup rich logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

# Usage
logger.info("Starting training...")
ic(cfg.model.backbone)  # Debug config values
```

### 9. Testing Patterns
```python
import pytest
import torch
from unittest.mock import Mock, patch

@pytest.fixture
def sample_batch():
    return {
        'image': torch.randn(2, 3, 224, 224),
        'polygons': [torch.randn(5, 2) for _ in range(2)]
    }

def test_model_forward(sample_batch):
    model = OCRModel(cfg)
    output = model(sample_batch['image'])

    assert 'maps' in output
    assert output['maps'].shape[0] == sample_batch['image'].shape[0]

@patch('hydra.utils.instantiate')
def test_component_instantiation(mock_instantiate):
    mock_component = Mock()
    mock_instantiate.return_value = mock_component

    result = get_encoder_by_cfg({'_target_': 'TestEncoder'})

    assert result == mock_component
    mock_instantiate.assert_called_once()
```

### 10. Utility Functions
```python
from pathlib import Path
from typing import Union, List
import json

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists and return Path object"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_json_config(path: Union[str, Path]) -> dict:
    """Load JSON configuration file"""
    with open(path, 'r') as f:
        return json.load(f)

def save_json_config(data: dict, path: Union[str, Path]) -> None:
    """Save data as JSON file"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def calculate_polygon_area(polygon: np.ndarray) -> float:
    """Calculate area of polygon using shoelace formula"""
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

## Reusable Utility Scripts

### Path Management (`ocr/utils/path_utils.py`)
Use these utilities for consistent path resolution across all scripts:

**Core Classes:**
- `OCRPathConfig`: Configuration class for all project paths
- `OCRPathResolver`: Central path resolution manager
- `PathUtils`: Legacy class for backward compatibility

**Key Functions:**
```python
from ocr.utils.path_utils import (
    get_project_root, get_data_path, get_config_path, get_outputs_path,
    get_images_path, get_annotations_path, get_logs_path,
    get_checkpoints_path, get_submissions_path, setup_project_paths
)

# Get standardized paths
project_root = get_project_root()
data_dir = get_data_path()
logs_dir = get_logs_path()
checkpoints_dir = get_checkpoints_path()

# Setup project paths with custom config
resolver = setup_project_paths({
    "project_root": "/custom/path",
    "data_dir": "custom_data"
})
```

**Usage Guidelines:**
- Always use `get_project_root()` instead of hardcoded paths
- Use path utilities for all file I/O operations
- Call `setup_project_paths()` early in scripts to ensure directories exist
- Use `OCRPathResolver` for complex path resolution needs

### Logging Utilities
Standardize logging across all scripts using consistent patterns from existing utilities.

### W&B Integration (`ocr/utils/wandb_utils.py`)
Use these functions for consistent experiment tracking:

```python
from ocr.utils.wandb_utils import generate_run_name, finalize_run, log_validation_images

# Generate descriptive run names
run_name = generate_run_name(config)

# Log validation images with GT/predicted boxes
log_validation_images(images, gt_bboxes, pred_bboxes, epoch=5)

# Finalize run with final metrics
finalize_run(final_validation_loss)
```

**Best Practices:**
- Always use `generate_run_name()` for consistent naming
- Call `finalize_run()` at the end of training to update run names with final scores
- Use `log_validation_images()` for visual debugging of model predictions

### OCR Utilities (`ocr/utils/ocr_utils.py`)
Common OCR-specific helper functions:

```python
from ocr.utils.ocr_utils import draw_boxes

# Visualize ground truth and predicted bounding boxes
annotated_image = draw_boxes(
    image_path,
    det_polys=predicted_boxes,
    gt_polys=ground_truth_boxes
)
```

### Development Guidelines
- **Import Order**: Always import utility functions from the appropriate modules
- **Path Resolution**: Never hardcode paths - use path utilities
- **Logging**: Use consistent logging patterns for better debugging
- **Reusability**: When creating new utilities, add them to appropriate modules and document here
```</content>
<parameter name="filePath">/home/vscode/workspace/upstage-receipt-text-detection-dbnet-baseline/docs/copilot/quick-reference.md
