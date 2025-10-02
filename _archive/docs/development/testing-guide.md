# Testing Guide

## Overview

This guide covers the testing strategy and best practices for the OCR framework. The testing suite is organized into multiple categories to ensure comprehensive coverage of all components.

## Testing Structure

```
tests/
├── unit/              # Unit tests for individual components
│   ├── test_models.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   └── test_utils.py
├── integration/       # Integration tests for component interaction
│   ├── test_training_pipeline.py
│   ├── test_evaluation_pipeline.py
│   └── test_data_pipeline.py
├── manual/            # Manual/visual verification tests
│   ├── test_visualization.py
│   ├── test_data_validation.py
│   └── test_model_outputs.py
├── debug/             # Debugging and development tests
│   ├── test_config_validation.py
│   ├── test_experiment_setup.py
│   └── test_reproducibility.py
└── wandb/             # Weights & Biases integration tests
    ├── test_logging.py
    ├── test_artifact_upload.py
    └── test_experiment_tracking.py
```

## Unit Testing

### 1. Model Components

```python
# tests/unit/test_models.py
import pytest
import torch
from ocr.models.encoder import TimmBackbone
from ocr.models.head import DBNetHead


class TestTimmBackbone:
    """Test TimmBackbone component."""

    @pytest.fixture
    def backbone_config(self):
        return {
            "backbone": "resnet18",
            "pretrained": True,
            "freeze_backbone": False
        }

    def test_initialization(self, backbone_config):
        """Test backbone initialization."""
        backbone = TimmBackbone(**backbone_config)
        assert backbone is not None
        assert hasattr(backbone, 'backbone')

    def test_forward_pass(self, backbone_config):
        """Test forward pass with sample input."""
        backbone = TimmBackbone(**backbone_config)
        x = torch.randn(1, 3, 512, 512)

        with torch.no_grad():
            output = backbone(x)

        assert output.shape[0] == 1  # batch size
        assert len(output.shape) == 4  # [B, C, H, W]

    @pytest.mark.parametrize("backbone_name", ["resnet18", "resnet50"])
    def test_different_backbones(self, backbone_name):
        """Test different backbone architectures."""
        config = {
            "backbone": backbone_name,
            "pretrained": False,
            "freeze_backbone": False
        }
        backbone = TimmBackbone(**config)
        assert backbone is not None


class TestDBNetHead:
    """Test DBNetHead component."""

    def test_output_shapes(self):
        """Test that output shapes match expectations."""
        head = DBNetHead(in_channels=256, out_channels=1)
        x = torch.randn(2, 256, 32, 32)

        with torch.no_grad():
            output = head(x)

        # DBNet outputs probability map and threshold map
        assert len(output) == 2
        assert output[0].shape == (2, 1, 32, 32)  # probability map
        assert output[1].shape == (2, 1, 32, 32)  # threshold map
```

### 2. Loss Functions

```python
# tests/unit/test_losses.py
import pytest
import torch
from ocr.models.loss import DBLoss, DiceLoss


class TestDBLoss:
    """Test DBNet loss function."""

    def test_loss_calculation(self):
        """Test loss calculation with sample data."""
        loss_fn = DBLoss()

        # Mock predictions and targets
        pred_maps = torch.randn(2, 1, 32, 32)
        pred_thresholds = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()

        loss = loss_fn(pred_maps, pred_thresholds, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # Loss should be non-negative
        assert not torch.isnan(loss)  # Loss should not be NaN

    def test_loss_components(self):
        """Test individual loss components."""
        loss_fn = DBLoss()

        # Test with perfect predictions
        pred_maps = torch.ones(1, 1, 32, 32)
        pred_thresholds = torch.zeros(1, 1, 32, 32)
        targets = torch.ones(1, 1, 32, 32)

        loss = loss_fn(pred_maps, pred_thresholds, targets)
        assert loss.item() < 0.1  # Should be close to zero


class TestDiceLoss:
    """Test Dice loss function."""

    def test_perfect_overlap(self):
        """Test dice loss with perfect overlap."""
        loss_fn = DiceLoss()
        pred = torch.ones(1, 1, 32, 32)
        target = torch.ones(1, 1, 32, 32)

        loss = loss_fn(pred, target)
        assert abs(loss.item()) < 1e-6  # Should be essentially zero

    def test_no_overlap(self):
        """Test dice loss with no overlap."""
        loss_fn = DiceLoss()
        pred = torch.ones(1, 1, 32, 32)
        target = torch.zeros(1, 1, 32, 32)

        loss = loss_fn(pred, target)
        assert loss.item() > 0.5  # Should be high loss
```

### 3. Metrics

```python
# tests/unit/test_metrics.py
import pytest
import torch
from ocr.metrics import CLEvalMetric


class TestCLEvalMetric:
    """Test CLEval metric implementation."""

    @pytest.fixture
    def sample_predictions(self):
        """Sample predictions for testing."""
        return [
            {
                "polygons": [[10, 10, 50, 10, 50, 30, 10, 30]],
                "text": "HELLO"
            }
        ]

    @pytest.fixture
    def sample_targets(self):
        """Sample ground truth for testing."""
        return [
            {
                "polygons": [[10, 10, 50, 10, 50, 30, 10, 30]],
                "text": "HELLO"
            }
        ]

    def test_perfect_match(self, sample_predictions, sample_targets):
        """Test metric with perfect predictions."""
        metric = CLEvalMetric()

        # Add samples
        for pred, target in zip(sample_predictions, sample_targets):
            metric.update(pred, target)

        # Compute metrics
        results = metric.compute()

        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert results["f1"] > 0.95  # Should be near perfect

    def test_empty_predictions(self):
        """Test metric with empty predictions."""
        metric = CLEvalMetric()

        pred = {"polygons": [], "text": ""}
        target = {
            "polygons": [[10, 10, 50, 10, 50, 30, 10, 30]],
            "text": "HELLO"
        }

        metric.update(pred, target)
        results = metric.compute()

        assert results["precision"] == 0.0
        assert results["recall"] == 0.0
```

## Integration Testing

### 1. Training Pipeline

```python
# tests/integration/test_training_pipeline.py
import pytest
import tempfile
import torch
from pathlib import Path
from ocr.lightning_modules import OCRLightningModule
from ocr.datasets import OCRDataset
from pytorch_lightning import Trainer


class TestTrainingPipeline:
    """Test complete training pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def minimal_config(self):
        """Minimal configuration for testing."""
        return {
            "model": {
                "encoder": {"backbone": "resnet18", "pretrained": False},
                "decoder": {"type": "unet"},
                "head": {"type": "db_head"},
                "loss": {"type": "db_loss"}
            },
            "data": {
                "batch_size": 2,
                "num_workers": 0
            },
            "training": {
                "max_epochs": 1,
                "learning_rate": 0.001
            }
        }

    def test_model_initialization(self, minimal_config):
        """Test lightning module initialization."""
        model = OCRLightningModule(minimal_config)
        assert model is not None
        assert hasattr(model, 'model')

    def test_forward_pass(self, minimal_config):
        """Test forward pass through lightning module."""
        model = OCRLightningModule(minimal_config)
        x = torch.randn(2, 3, 256, 256)

        with torch.no_grad():
            output = model(x)

        assert output is not None
        assert len(output.shape) == 4

    @pytest.mark.slow
    def test_training_step(self, minimal_config, temp_dir):
        """Test complete training step."""
        # Create minimal dataset
        # Note: In real tests, you'd create actual mock data
        dataset = OCRDataset(...)  # Mock dataset

        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            num_workers=0
        )

        # Initialize model and trainer
        model = OCRLightningModule(minimal_config)
        trainer = Trainer(
            max_epochs=1,
            default_root_dir=str(temp_dir),
            enable_checkpointing=False,
            logger=False
        )

        # Run training
        trainer.fit(model, dataloader)

        # Verify training completed
        assert trainer.state.finished
```

### 2. Data Pipeline

```python
# tests/integration/test_data_pipeline.py
import pytest
from pathlib import Path
from ocr.datasets import OCRDataset
from ocr.datasets.transforms import OCRTransforms


class TestDataPipeline:
    """Test data loading and preprocessing pipeline."""

    @pytest.fixture
    def sample_data_dir(self):
        """Directory with sample data."""
        return Path("tests/fixtures/sample_data")

    def test_dataset_initialization(self, sample_data_dir):
        """Test dataset initialization."""
        transforms = OCRTransforms(is_training=False)
        dataset = OCRDataset(
            data_dir=sample_data_dir,
            transforms=transforms
        )

        assert len(dataset) > 0
        assert hasattr(dataset, '__getitem__')

    def test_data_loading(self, sample_data_dir):
        """Test loading individual data samples."""
        transforms = OCRTransforms(is_training=False)
        dataset = OCRDataset(
            data_dir=sample_data_dir,
            transforms=transforms
        )

        # Load first sample
        sample = dataset[0]

        # Verify sample structure
        assert "image" in sample
        assert "polygons" in sample
        assert "text" in sample

        # Verify data types
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["polygons"], list)

    def test_batch_collation(self, sample_data_dir):
        """Test batch collation function."""
        from ocr.datasets.db_collate_fn import db_collate_fn

        transforms = OCRTransforms(is_training=False)
        dataset = OCRDataset(
            data_dir=sample_data_dir,
            transforms=transforms
        )

        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            collate_fn=db_collate_fn
        )

        # Load batch
        batch = next(iter(dataloader))

        # Verify batch structure
        assert "images" in batch
        assert "polygons" in batch
        assert batch["images"].shape[0] == 2  # batch size
```

## Manual Testing

### 1. Visualization Tests

```python
# tests/manual/test_visualization.py
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
from ocr.utils.visualization import visualize_predictions


class TestVisualization:
    """Manual tests for visualization functions."""

    @pytest.fixture
    def sample_predictions(self):
        """Sample predictions for visualization."""
        return {
            "image": torch.randn(3, 512, 512),
            "polygons": [
                [100, 100, 200, 100, 200, 150, 100, 150],
                [250, 200, 350, 200, 350, 250, 250, 250]
            ],
            "text": ["HELLO", "WORLD"]
        }

    def test_prediction_visualization(self, sample_predictions):
        """Test prediction visualization."""
        fig = visualize_predictions(sample_predictions)

        assert fig is not None
        assert len(fig.axes) > 0

        # Save visualization for manual inspection
        output_path = Path("tests/outputs/visualization_test.png")
        output_path.parent.mkdir(exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)

        assert output_path.exists()

    def test_overlay_visualization(self, sample_predictions):
        """Test overlay visualization."""
        from ocr.utils.visualization import create_overlay

        overlay = create_overlay(
            sample_predictions["image"],
            sample_predictions["polygons"]
        )

        assert overlay.shape == sample_predictions["image"].shape
```

### 2. Data Validation Tests

```python
# tests/manual/test_data_validation.py
import pytest
from pathlib import Path
from ocr.utils.data_validation import validate_dataset


class TestDataValidation:
    """Manual tests for data validation."""

    @pytest.fixture
    def test_data_dir(self):
        """Test data directory."""
        return Path("tests/fixtures/validation_data")

    def test_dataset_validation(self, test_data_dir):
        """Test comprehensive dataset validation."""
        validation_report = validate_dataset(test_data_dir)

        # Check report structure
        assert "total_images" in validation_report
        assert "total_annotations" in validation_report
        assert "invalid_polygons" in validation_report
        assert "missing_images" in validation_report

        # Verify no critical issues
        assert validation_report["missing_images"] == 0
        assert validation_report["corrupted_images"] == 0

    def test_annotation_format_validation(self, test_data_dir):
        """Test annotation format validation."""
        from ocr.utils.data_validation import validate_annotation_format

        annotation_file = test_data_dir / "annotations.json"

        if annotation_file.exists():
            issues = validate_annotation_format(annotation_file)

            # Should not have format errors
            assert len(issues) == 0
```

## Debug Testing

### 1. Configuration Validation

```python
# tests/debug/test_config_validation.py
import pytest
from omegaconf import OmegaConf
from ocr.utils.config_validation import validate_config


class TestConfigValidation:
    """Debug tests for configuration validation."""

    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = {
            "model": {
                "encoder": {"backbone": "resnet50"},
                "decoder": {"type": "unet"},
                "head": {"type": "db_head"},
                "loss": {"type": "db_loss"}
            },
            "data": {
                "batch_size": 8,
                "image_size": [512, 512]
            },
            "training": {
                "learning_rate": 0.001,
                "max_epochs": 100
            }
        }

        issues = validate_config(config)
        assert len(issues) == 0

    def test_invalid_config(self):
        """Test validation of invalid configuration."""
        config = {
            "model": {
                "encoder": {"backbone": "invalid_backbone"},
                "decoder": {"type": "invalid_decoder"}
            }
        }

        issues = validate_config(config)
        assert len(issues) > 0

        # Check specific issues
        issue_messages = [issue["message"] for issue in issues]
        assert any("backbone" in msg.lower() for msg in issue_messages)

    @pytest.mark.parametrize("backbone", ["resnet18", "resnet50", "efficientnet_b0"])
    def test_supported_backbones(self, backbone):
        """Test all supported backbones."""
        config = {
            "model": {
                "encoder": {"backbone": backbone},
                "decoder": {"type": "unet"},
                "head": {"type": "db_head"},
                "loss": {"type": "db_loss"}
            }
        }

        issues = validate_config(config)
        backbone_issues = [i for i in issues if "backbone" in i["field"].lower()]
        assert len(backbone_issues) == 0
```

### 2. Reproducibility Tests

```python
# tests/debug/test_reproducibility.py
import pytest
import torch
import numpy as np
from ocr.utils.reproducibility import set_seed, get_deterministic_config


class TestReproducibility:
    """Debug tests for reproducibility."""

    def test_seed_setting(self):
        """Test random seed setting."""
        set_seed(42)

        # Generate random numbers
        torch_vals_1 = torch.randn(10)
        np_vals_1 = np.random.randn(10)

        # Reset seed and generate again
        set_seed(42)
        torch_vals_2 = torch.randn(10)
        np_vals_2 = np.random.randn(10)

        # Values should be identical
        assert torch.allclose(torch_vals_1, torch_vals_2)
        assert np.allclose(np_vals_1, np_vals_2)

    def test_deterministic_config(self):
        """Test deterministic configuration."""
        config = get_deterministic_config()

        # Check PyTorch settings
        assert config["torch_deterministic"] is True
        assert config["torch_benchmark"] is False

        # Check CUDA settings if available
        if torch.cuda.is_available():
            assert config["cuda_deterministic"] is True

    def test_model_reproducibility(self):
        """Test model training reproducibility."""
        set_seed(42)

        # Create model
        model = torch.nn.Linear(10, 1)

        # Generate same input multiple times
        inputs = torch.randn(5, 10)

        # Get outputs
        with torch.no_grad():
            output1 = model(inputs)

        # Reset and try again
        set_seed(42)
        model2 = torch.nn.Linear(10, 1)
        with torch.no_grad():
            output2 = model2(inputs)

        # Outputs should be identical
        assert torch.allclose(output1, output2)
```

## Weights & Biases Integration Testing

### 1. Logging Tests

```python
# tests/wandb/test_logging.py
import pytest
from unittest.mock import patch, MagicMock
from ocr.utils.logging import WandbLogger


class TestWandbLogging:
    """Test Weights & Biases logging integration."""

    @pytest.fixture
    def mock_wandb(self):
        """Mock wandb module."""
        with patch("ocr.utils.logging.wandb") as mock_wandb:
            yield mock_wandb

    def test_logger_initialization(self, mock_wandb):
        """Test logger initialization."""
        config = {"project": "ocr-test", "name": "test-run"}

        logger = WandbLogger(config)

        mock_wandb.init.assert_called_once_with(
            project="ocr-test",
            name="test-run"
        )

    def test_log_metrics(self, mock_wandb):
        """Test logging metrics."""
        logger = WandbLogger({})

        metrics = {"loss": 0.5, "accuracy": 0.85}

        logger.log_metrics(metrics, step=10)

        mock_wandb.log.assert_called_once_with(metrics, step=10)

    def test_log_artifacts(self, mock_wandb):
        """Test artifact logging."""
        logger = WandbLogger({})

        artifact_path = "models/checkpoint.pth"
        artifact_name = "model-checkpoint"

        logger.log_artifact(artifact_path, artifact_name)

        mock_wandb.log_artifact.assert_called_once()
        args = mock_wandb.log_artifact.call_args[0]
        assert artifact_name in args[0]
```

## Test Configuration and Fixtures

### 1. Shared Fixtures

```python
# tests/conftest.py
import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf


@pytest.fixture(scope="session")
def test_data_dir():
    """Root directory for test data."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_image():
    """Sample image tensor for testing."""
    return torch.randn(3, 512, 512)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    config_path = Path(__file__).parent / "fixtures" / "sample_config.yaml"
    return OmegaConf.load(config_path)


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, size=10):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "image": torch.randn(3, 256, 256),
                "polygons": [[10, 10, 50, 10, 50, 30, 10, 30]],
                "text": "SAMPLE"
            }

    return MockDataset()
```

### 2. Test Configuration

```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --strict-markers
    --strict-config
    --disable-warnings
    --tb=short
    -v
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    manual: marks tests as manual/visual tests
    gpu: marks tests that require GPU
```

## Running Tests

### 1. Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_models.py

# Run specific test class
pytest tests/unit/test_models.py::TestTimmBackbone

# Run specific test method
pytest tests/unit/test_models.py::TestTimmBackbone::test_initialization

# Run tests with coverage
pytest --cov=ocr --cov-report=html
```

### 2. Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run only manual tests
pytest -m manual

# Skip slow tests
pytest -m "not slow"

# Run tests requiring GPU
pytest -m gpu
```

### 3. Test Configuration

```bash
# Run tests in parallel
pytest -n auto

# Run tests with verbose output
pytest -v

# Run tests and stop on first failure
pytest -x

# Run tests with detailed tracebacks
pytest --tb=long
```

## Continuous Integration

### 1. GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]

    - name: Run tests
      run: |
        pytest --cov=ocr --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

### 2. Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8

  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        args: [--tb=short, --strict-markers]
```

This testing guide provides comprehensive coverage for all aspects of the OCR framework, ensuring reliability and maintainability through automated testing.</content>
<parameter name="filePath">/home/vscode/workspace/upstage-receipt-text-detection-dbnet-baseline/docs/development/testing-guide.md
