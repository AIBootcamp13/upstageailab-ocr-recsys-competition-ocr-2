# AI Coding Agent Instructions for OCR Receipt Text Detection Project

## Project Overview
This is a modular OCR system for receipt text detection using PyTorch Lightning and Hydra. The architecture enables plug-and-play component swapping through a custom registry system, supporting rapid experimentation with different encoders, decoders, and heads.

## Architecture Patterns

### Modular Component System
- **Registry-based architecture**: Components are registered and assembled via Hydra configs
- **Abstract interfaces**: All components inherit from base classes (`BaseEncoder`, `BaseDecoder`, `BaseHead`, `BaseLoss`)
- **Factory pattern**: `ModelFactory` assembles complete models from registered components
- **Example**: `configs/model/encoder/resnet50.yaml` defines encoder configs that get instantiated via `_target_`

### Data Flow Contracts
- **Strict validation**: Use Pydantic v2 models for runtime data validation
- **Dataset contract**: `__getitem__` returns `DatasetSample` with exact shapes/types
- **Transform contract**: Albumentations pipeline preserves polygon coordinates and probability maps
- **Key validation**: Check `docs/pipeline/data_contracts.md` before modifying data structures

## Critical Developer Workflows

### Training & Experimentation
```bash
# Start context logging for experiments
make context-log-start LABEL="experiment_name"

# Run training with config overrides
uv run python runners/train.py model.encoder.name=resnet50 data.batch_size=8

# Analyze results with ablation tools
uv run python ablation_study/collect_results.py --project "receipt-text-recognition-ocr-project"
```

### Code Quality & Validation
```bash
# Auto-fix formatting and linting
make quality-fix

# Run comprehensive checks
make quality-check

# Validate configurations
uv run python scripts/agent_tools/validate_config.py --config-name train
```

### UI Development & Testing
```bash
# Start different UIs
make serve-ui                    # Command builder
make serve-evaluation-ui         # Results viewer
make serve-inference-ui          # OCR inference
make serve-resource-monitor      # System monitoring
```

## Project-Specific Conventions

### Configuration Management
- **Hydra-first**: All experiments driven by YAML configs in `configs/`
- **Override pattern**: Use command-line overrides like `model.optimizer.lr=0.0005`
- **Registry integration**: New components auto-discoverable via `architectures/registry.py`

### Naming & Structure
- **Modules**: `snake_case` (e.g., `ocr_framework`, `data_loader.py`)
- **Classes**: `PascalCase` with prefixes (`BaseEncoder`, `OCRLightningModule`)
- **Abstract bases**: Start with `Base` (e.g., `BaseDecoder`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_IMAGE_SIZE`)

### Data Handling
- **Polygon format**: List of `(N, 2)` numpy arrays for text regions
- **Image format**: `(H, W, 3)` numpy arrays, uint8 or float32
- **Probability maps**: `(H, W)` float32 arrays in range [0, 1]
- **Validation**: Runtime checks prevent shape/type errors

## Integration Points

### Experiment Tracking
- **Primary**: Weights & Biases for metrics, artifacts, and comparisons
- **Backup**: CSV logging for programmatic analysis
- **Analysis**: `ablation_study/` tools for automated result comparison

### UI Components
- **Streamlit apps**: Multiple specialized UIs in `ui/` directory
- **Command builder**: Interactive config generation
- **Resource monitor**: GPU/CPU usage tracking
- **Evaluation viewer**: Results visualization and analysis

### Documentation System
- **AI Handbook**: `docs/ai_handbook/` is single source of truth
- **Context bundles**: Curated file sets for common tasks
- **Protocol library**: Standardized approaches for development tasks

## Common Patterns & Examples

### Adding New Components
```python
# Register in architectures/registry.py
@register_encoder('custom_encoder')
class CustomEncoder(BaseEncoder):
    def __init__(self, config):
        super().__init__(config)
        # Implementation

# Use in Hydra config
encoder:
  _target_: ocr_framework.architectures.encoders.CustomEncoder
  custom_param: value
```

### Data Pipeline Debugging
```python
# Validate dataset outputs
from ocr.datasets import ValidatedOCRDataset
dataset = ValidatedOCRDataset(...)
sample = dataset[0]  # Pydantic validation ensures contract compliance

# Check transform pipeline
transforms = DBTransforms(config)
result = transforms(sample)  # Validates input/output contracts
```

### Experiment Documentation
```python
# Always document experiments
# Copy template: cp docs/ai_handbook/04_experiments/TEMPLATE.md docs/ai_handbook/04_experiments/YYYY-MM-DD_experiment.md

# Log context: make context-log-start LABEL="experiment_name"
# Document hypothesis, config, and results
```

## Key Files & Directories
- `docs/ai_handbook/index.md`: Complete project documentation index
- `docs/pipeline/data_contracts.md`: Data structure specifications
- `configs/`: Hydra configuration hierarchy
- `ablation_study/`: Experiment analysis tools
- `ui/`: Streamlit applications
- `scripts/agent_tools/`: AI-safe automation scripts

## Quality Assurance
- **Pre-commit hooks**: Automatic formatting and linting
- **Runtime validation**: Pydantic contracts prevent data errors
- **Type hints**: Comprehensive typing for IDE support
- **Test coverage**: pytest suite with automated validation

Remember: Check `docs/ai_handbook/index.md` for task-specific context bundles and protocols before making changes.
