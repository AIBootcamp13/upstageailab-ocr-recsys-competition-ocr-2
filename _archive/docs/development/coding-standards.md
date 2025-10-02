# Coding Standards and Best Practices

## Python Style Guide

### 1. Code Formatting
- **Formatter**: Black with 88 character line length
- **Import sorting**: isort with black compatibility
- **Docstring format**: Google style

```python
# Correct formatting (Black will handle this)
def complex_function(param1, param2, param3, param4, param5):
    """Do something complex with multiple parameters.

    Args:
        param1: First parameter description
        param2: Second parameter description
        param3: Third parameter description
        param4: Fourth parameter description
        param5: Fifth parameter description

    Returns:
        Description of return value
    """
    result = param1 + param2 + param3 + param4 + param5
    return result
```

### 2. Import Organization
```python
# Standard library imports
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party imports
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import albumentations as A
from hydra.utils import instantiate
from omegaconf import DictConfig

# Local imports
from ocr.models.encoder import get_encoder_by_cfg
from ocr.utils.path_utils import get_project_root
from ..core.base_model import BaseOCRModel
```

### 3. Type Hints
- **Required**: All public functions and methods
- **Optional**: Private functions (single underscore prefix)
- **Comprehensive**: Use Union, Optional, Literal where appropriate

```python
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import torch
import numpy as np

def process_image(
    image_path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True
) -> torch.Tensor:
    """Process image for model input.

    Args:
        image_path: Path to image file
        target_size: Optional resize dimensions (H, W)
        normalize: Whether to normalize pixel values

    Returns:
        Processed image tensor
    """
    # Implementation
    pass

def validate_polygons(polygons: List[np.ndarray]) -> List[np.ndarray]:
    """Filter and validate polygon coordinates.

    Args:
        polygons: List of polygon coordinate arrays

    Returns:
        List of valid polygons
    """
    # Implementation
    pass
```

## Architecture Patterns

### 1. Abstract Base Classes
```python
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn

class BaseEncoder(nn.Module, ABC):
    """Abstract base class for all encoders."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder.

        Args:
            x: Input tensor

        Returns:
            Encoded features
        """
        pass

    @property
    @abstractmethod
    def output_channels(self) -> int:
        """Number of output channels."""
        pass
```

### 2. Factory Functions
```python
from hydra.utils import instantiate
from typing import Dict, Any

def get_encoder_by_cfg(config: Dict[str, Any]) -> BaseEncoder:
    """Factory function for encoder instantiation.

    Args:
        config: Hydra configuration dictionary

    Returns:
        Instantiated encoder

    Raises:
        ValueError: If encoder type is not supported
    """
    if '_target_' not in config:
        raise ValueError("Config must contain '_target_' field")

    encoder = instantiate(config)

    if not isinstance(encoder, BaseEncoder):
        raise TypeError(f"Encoder must inherit from BaseEncoder, got {type(encoder)}")

    return encoder
```

### 3. Configuration Classes
```python
from dataclasses import dataclass
from typing import Optional, List
from omegaconf import MISSING

@dataclass
class EncoderConfig:
    """Configuration for encoder components."""

    backbone: str = "resnet50"
    pretrained: bool = True
    features_only: bool = True
    out_indices: Optional[List[int]] = None

    # Hydra integration
    _target_: str = "ocr.models.encoder.TimmBackbone"
```

## Error Handling and Logging

### 1. Exception Handling
```python
import logging
from typing import Union
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_path: Union[str, Path]) -> dict:
    """Load configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = Path(config_path)

    try:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Validate config
        required_keys = ['model', 'data']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        logger.info(f"Successfully loaded config from {config_path}")
        return config

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file {config_path}: {e}")
        raise ValueError(f"Invalid config file format: {e}") from e
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise
```

### 2. Logging Standards
```python
import logging
from rich.logging import RichHandler
from rich.console import Console

# Setup rich logging for development
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)

# Usage patterns
logger.debug("Detailed debug information")
logger.info("General information about program execution")
logger.warning("Warning about potential issues")
logger.error("Error that doesn't stop execution")
logger.critical("Critical error that may stop execution")
```

## Testing Standards

### 1. Unit Test Structure
```python
import pytest
import torch
from unittest.mock import Mock, patch
from pathlib import Path

class TestOCRModel:
    """Test cases for OCRModel class."""

    @pytest.fixture
    def sample_config(self):
        """Provide sample configuration for testing."""
        return {
            'encoder': {'_target_': 'ocr.models.encoder.TimmBackbone'},
            'decoder': {'_target_': 'ocr.models.decoder.UNetDecoder'},
            'head': {'_target_': 'ocr.models.head.DBHead'},
            'loss': {'_target_': 'ocr.models.loss.DBLoss'}
        }

    @pytest.fixture
    def sample_batch(self):
        """Provide sample batch for testing."""
        return {
            'image': torch.randn(2, 3, 224, 224),
            'polygons': [torch.randn(5, 4, 2) for _ in range(2)]
        }

    def test_model_initialization(self, sample_config):
        """Test that model initializes correctly."""
        with patch('hydra.utils.instantiate') as mock_instantiate:
            mock_instantiate.return_value = Mock()

            model = OCRModel(sample_config)

            assert hasattr(model, 'encoder')
            assert hasattr(model, 'decoder')
            assert hasattr(model, 'head')
            assert hasattr(model, 'loss')

    def test_forward_pass(self, sample_config, sample_batch):
        """Test forward pass produces expected outputs."""
        model = OCRModel(sample_config)

        with patch.object(model, 'encoder') as mock_encoder, \
             patch.object(model, 'decoder') as mock_decoder, \
             patch.object(model, 'head') as mock_head, \
             patch.object(model, 'loss') as mock_loss:

            # Setup mocks
            mock_encoder.return_value = torch.randn(2, 256, 28, 28)
            mock_decoder.return_value = torch.randn(2, 128, 56, 56)
            mock_head.return_value = {'maps': torch.randn(2, 1, 224, 224)}
            mock_loss.return_value = (torch.tensor(0.5), {'total_loss': 0.5})

            result = model(sample_batch['image'])

            assert 'maps' in result
            assert 'loss' in result
            assert 'loss_dict' in result
```

### 2. Test Organization
```
tests/
├── unit/              # Unit tests (fast, isolated)
│   ├── test_models.py
│   ├── test_datasets.py
│   └── test_utils.py
├── integration/       # Integration tests (slower, multi-component)
│   ├── test_training_pipeline.py
│   └── test_evaluation_pipeline.py
├── manual/           # Manual verification tests
│   ├── test_visualization.py
│   └── test_data_validation.py
├── debug/            # Debug utilities and helpers
│   ├── debug_model.py
│   └── debug_data.py
└── wandb/            # W&B integration tests
    ├── test_logging.py
    └── test_artifacts.py
```

## Performance and Optimization

### 1. Memory Management
```python
# Use context managers for large objects
@torch.no_grad()
def inference_batch(model, batch):
    """Run inference without gradient computation."""
    return model(batch)

# Use mixed precision where appropriate
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def training_step(model, batch):
    with autocast():
        output = model(batch)
        loss = output['loss']

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. Efficient Data Loading
```python
from torch.utils.data import DataLoader

# Optimize DataLoader settings
dataloader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,  # Adjust based on CPU cores
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
    prefetch_factor=2  # Prefetch batches
)
```

## Documentation Standards

### 1. Module Docstrings
```python
"""OCR Model Architectures.

This module contains the main OCR model implementations including
DBNet, EAST, and other text detection architectures. All models
follow a modular encoder-decoder-head pattern for maximum flexibility.

Classes:
    OCRModel: Main model orchestrating the detection pipeline
    DBNetModel: Differentiable Binarization Network implementation
    EASTModel: Efficient and Accurate Scene Text detection

Examples:
    >>> model = OCRModel(cfg)
    >>> output = model(batch)
    >>> polygons = model.get_polygons_from_maps(gt, pred)
"""

__version__ = "1.0.0"
__author__ = "OCR Team"
```

### 2. Function Documentation
```python
def postprocess_predictions(
    probability_maps: torch.Tensor,
    threshold: float = 0.5,
    min_area: int = 10
) -> List[np.ndarray]:
    """Post-process probability maps to extract text polygons.

    This function applies thresholding, morphological operations,
    and contour extraction to convert probability maps into
    polygon coordinates representing detected text regions.

    Args:
        probability_maps: Binary probability maps (B, 1, H, W)
        threshold: Binarization threshold (0-1)
        min_area: Minimum polygon area to keep

    Returns:
        List of polygon coordinates for each image in batch.
        Each polygon is a numpy array of shape (N, 2) where N
        is the number of vertices.

    Raises:
        ValueError: If probability_maps has invalid shape

    Examples:
        >>> maps = torch.randn(2, 1, 224, 224)
        >>> polygons = postprocess_predictions(maps, threshold=0.3)
        >>> len(polygons)  # Number of images
        2
    """
```

## Configuration Management

### 1. Hydra Best Practices
```python
# Use MISSING for required fields
@dataclass
class ModelConfig:
    backbone: str = MISSING
    pretrained: bool = True
    freeze_backbone: bool = False

# Use interpolation for derived values
model:
  batch_size: 8
  num_epochs: 100
  steps_per_epoch: ${eval:'${data.train_size} // ${model.batch_size}'}

# Group related configurations
defaults:
  - model: dbnet
  - data: icdar
  - training: default
  - _self_
```

### 2. Configuration Validation
```python
def validate_model_config(cfg: DictConfig) -> List[str]:
    """Validate model configuration.

    Args:
        cfg: Model configuration

    Returns:
        List of validation error messages
    """
    errors = []

    if cfg.backbone not in ['resnet50', 'resnet101', 'efficientnet_b0']:
        errors.append(f"Unsupported backbone: {cfg.backbone}")

    if cfg.learning_rate <= 0:
        errors.append("Learning rate must be positive")

    return errors
```</content>
<parameter name="filePath">/home/vscode/workspace/upstage-receipt-text-detection-dbnet-baseline/docs/development/coding-standards.md
