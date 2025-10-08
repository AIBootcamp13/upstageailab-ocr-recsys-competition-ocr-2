# 2025-10-08 Hydra Logging and Extras Configuration Implementation

## Summary
Implemented comprehensive Hydra logging configuration with Rich formatting and added extras configuration for managing miscellaneous settings. This establishes proper logging infrastructure and centralized configuration for experiment metadata. Also fixed dataset limiting functionality that was broken due to configuration restructuring.

## Changes Made

### 1. Created Hydra Configuration Structure
- **Added**: `configs/hydra/default.yaml` - Main Hydra configuration with structured logging
  - Configures dynamic output directories based on task name and timestamp
  - Sets up structured logging with file output in experiment directories
  - Designed for compatibility with standard Hydra logging (colorlog can be added later if needed)
- **Added**: `configs/extras/default.yaml` - Miscellaneous settings management
  - `ignore_warnings`: Controls Python warning suppression
  - `enforce_tags`: Requires user-provided experiment tags
  - `print_config`: Enables pretty-printing of configuration tree with Rich

### 2. Enhanced Logger Configuration
- **Added**: `configs/logger/csv.yaml` - CSV logging capability for experiments
  - Uses Lightning's built-in CSVLogger
  - Saves logs to `${paths.output_dir}/csv/` directory
  - Provides structured experiment logging alongside WandB
- **Updated**: `configs/logger/default.yaml` to include CSV logger in defaults
  - Now includes both WandB and CSV loggers by default
  - Maintains backward compatibility with existing configurations

### 3. Updated Base Configuration
- **Updated**: `configs/base.yaml` to include new configuration groups
  - Added `hydra: default` and `extras: default` to defaults list
  - Added `task_name: "train"` for Hydra logging context
  - Ensures all configurations inherit proper logging and extras settings

### 4. Enhanced Path Management
- **Updated**: `configs/paths/default.yaml` to include `output_dir`
  - Added `output_dir: 'outputs/${exp_name}'` for CSV logger compatibility
  - Maintains consistent path structure across all logging components

### 5. Cleaned Up Training Configuration
- **Updated**: `configs/train.yaml` to remove redundant Hydra overrides
  - Removed `override hydra/hydra_logging: disabled` and `override hydra/job_logging: disabled`
  - Now properly inherits Rich logging configuration from `hydra/default.yaml`
  - Streamlines configuration composition

### 6. Fixed Dataset Limiting Functionality
- **Fixed**: Dataset sample limiting that was broken after configuration restructuring
- **Updated**: `ocr/datasets/__init__.py` to properly access data config for limiting parameters
- **Updated**: `ocr/lightning_modules/__init__.py` to pass data config to dataset creation
- **Fixed**: `ocr/lightning_modules/ocr_pl.py` to handle Subset objects when accessing dataset annotations
- **Result**: `val_num_samples: 8` in synthetic_debug now correctly limits validation to 8 samples

## Benefits Achieved

1. **Structured Logging**: Organized logging output with automatic directory management for better experiment tracking
2. **Structured Experiment Logging**: CSV logging provides machine-readable experiment data alongside WandB visualization
3. **Centralized Configuration**: Extras configuration provides a single place to manage miscellaneous experiment settings
4. **Improved Developer Experience**: Automatic configuration of logging directories and formats
5. **Enhanced Experiment Tracking**: Multiple logging backends ensure comprehensive experiment documentation
6. **Template Alignment**: Configuration structure now aligns with lightning-hydra-template best practices

## Configuration Usage

### Basic Training with Enhanced Logging
```bash
uv run python runners/train.py experiment=synthetic_debug
```

### Custom Logging Configuration
```bash
# Override logging settings
uv run python runners/train.py \
  experiment=synthetic_debug \
  extras.print_config=false \
  hydra.run.dir=outputs/custom_experiment
```

### CSV-Only Logging (Disable WandB)
```bash
uv run python runners/train.py \
  experiment=synthetic_debug \
  logger.wandb.mode=disabled
```

## Files Changed

### New Files
- `configs/hydra/default.yaml` - Hydra logging configuration
- `configs/extras/default.yaml` - Miscellaneous settings
- `configs/logger/csv.yaml` - CSV logger configuration

### Modified Files
- `configs/base.yaml` - Added hydra and extras defaults
- `configs/logger/default.yaml` - Included CSV logger
- `configs/paths/default.yaml` - Added output_dir
- `configs/train.yaml` - Removed redundant overrides
- `ocr/datasets/__init__.py` - Fixed dataset limiting to work with config structure
- `ocr/lightning_modules/__init__.py` - Updated to pass data config for dataset limiting
- `ocr/lightning_modules/ocr_pl.py` - Fixed dataset access for Subset objects

## Validation
All YAML configurations have been validated for syntax correctness and Hydra composition compatibility.
