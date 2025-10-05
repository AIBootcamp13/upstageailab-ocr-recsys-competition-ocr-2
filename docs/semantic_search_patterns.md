# Semantic Search Patterns for the OCR Project

This document outlines effective search patterns and queries for exploring the OCR receipt text detection project. The project uses a modular architecture with DBNet as the baseline model and leverages Hydra for configuration management.

## Architecture Overview

The project implements a modular OCR framework with plug-and-play components:

- **Encoders**: Extract features from input images (e.g., ResNet backbones)
- **Decoders**: Process encoder features to produce higher-resolution maps (e.g., U-Net)
- **Heads**: Generate final predictions (e.g., DBHead for text detection)
- **Losses**: Compute training objectives (e.g., DBLoss)
- **Metrics**: Evaluate model performance

The components are managed through a central registry system (`ocr/models/core/registry.py`) that enables flexible architecture composition.

## Key Search Patterns

### Finding Architecture Components

```bash
# Find all DBNet-related code
pattern="DBNet|dbnet"
include="**/*.py"

# Find specific architecture implementations
pattern="class.*DBNet"
include="**/*.py"

# Find all architecture registrations
pattern="register.*architecture"
include="**/*.py"
```

### Exploring the Registry System

```bash
# Find the registry implementation
file="ocr/models/core/registry.py"

# Find all registered components
pattern="registry\.register"
include="**/*.py"

# Find architecture presets
pattern="register_architecture"
include="**/*.py"
```

### Understanding Model Configuration

```bash
# Find the main OCR model class
file="ocr/models/architecture.py"

# Find base classes for components
file="ocr/models/core/base_classes.py"

# Find command builder utilities
pattern="CommandBuilder"
include="**/*.py"
```

### Navigation by Functionality

```bash
# Find training scripts
file="runners/train.py"

# Find configuration files
path="configs/"

# Find UI components
pattern="run_command_builder|command_builder"
include="**/*.py"
```

## Codebase Structure

```
ocr/
├── models/                 # Core model components
│   ├── architectures/      # Architecture registration (dbnet.py, dbnetpp.py)
│   ├── core/              # Registry and base classes
│   ├── encoder/           # Feature extractors
│   ├── decoder/           # Feature processing modules
│   ├── head/              # Prediction heads
│   └── loss/              # Loss functions
runners/                   # Execution scripts (train.py, test.py, predict.py)
configs/                   # Hydra configuration files
ui/                        # Streamlit UI components
├── utils/
│   └── command/           # Command building utilities
tests/                     # Unit tests
scripts/                   # Utility scripts
```

## Effective Search Queries

### 1. Finding Implementation Details
- Query: "class.*DBHead" to find the DBNet head implementation
- Query: "def forward" to find forward pass implementations
- Query: "BaseEncoder|BaseDecoder|BaseHead|BaseLoss" to find base classes

### 2. Understanding Configuration
- Query: "hydra.main" to find Hydra configuration entry points
- Query: "get_registry|ComponentRegistry" to find the registry system
- Query: "CommandBuilder" to find the command construction utilities

### 3. Locating Test Files
- Query: "test_.*dbnet" to find DBNet-specific tests
- Query: "conftest.py" for test fixtures
- Query: "tests/test_architecture.py" for architecture tests

### 4. Finding UI Components
- Query: "run_ui.py" for the main UI entry point
- Query: "command_builder.py" for the command builder UI
- Query: "ui/utils/command/" for command-related utilities

## Specific Examples

### Finding How DBNet is Constructed:
1. Search for `register_dbnet_components` in `ocr/models/architectures/dbnet.py`
2. Look at `OCRModel` class in `ocr/models/architecture.py` to see how components are instantiated
3. Check the registry implementation in `ocr/models/core/registry.py`

### Understanding the Training Process:
1. Examine `runners/train.py` for the main training loop
2. Look at `configs/` directory for configuration files
3. Check `get_pl_modules_by_cfg` function to see how models are built from config

### Finding Model Components:
1. To find encoders: Search for files in `ocr/models/encoder/` or patterns like `register_encoder`
2. To find decoders: Search for files in `ocr/models/decoder/` or patterns like `register_decoder`
3. To find heads: Search for files in `ocr/models/head/` or patterns like `register_head`
4. To find losses: Search for files in `ocr/models/loss/` or patterns like `register_loss`

This modular structure allows for easy experimentation with different components while maintaining a consistent interface through the abstract base classes.
