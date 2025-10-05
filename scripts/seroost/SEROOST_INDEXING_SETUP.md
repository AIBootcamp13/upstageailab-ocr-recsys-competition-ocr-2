# Seroost Indexing Setup for OCR Project

This document provides a complete overview of the Seroost indexing configuration for the OCR project.

## Overview

This setup provides comprehensive indexing for the OCR project with appropriate inclusion and exclusion rules to optimize search performance and exclude unnecessary files.

## Components

### 1. Configuration File (`seroost_config.json`)
- Defines which files and directories to include/exclude during indexing
- Optimized for the OCR project structure
- Includes source code, documentation, and configuration files
- Excludes build artifacts, dependencies, and large data files

### 2. Setup Script (`setup_seroost_indexing.py`)
- Python script that configures and runs the Seroost indexing
- Provides graceful handling when Seroost is not installed
- Points to the correct project directory and configuration

### 3. Shell Scripts
- `run_seroost_indexing.sh`: Runs the indexing using the existing Seroost installation
- `install_and_run_seroost.sh`: Installs Seroost and runs the indexing setup

### 4. Test Script (`test_seroost_config.py`)
- Validates that the configuration includes expected patterns
- Verifies both include and exclude patterns are present

## Included Files

The indexing configuration includes:

- **Source Code**: Python files (`*.py`), with specific inclusion of key directories like `ocr/`, `ui/`, `runners/`, etc.
- **Configuration**: YAML, TOML, JSON files including the entire `configs/` directory
- **Documentation**: Markdown files and the entire `docs/` directory
- **Scripts**: Shell scripts, Python scripts in the `scripts/` directory
- **Tests**: Test files in the `tests/` directory
- **Other Text Files**: Various text-based configuration and documentation files

## Excluded Files

The indexing configuration excludes:

- **Virtual Environments**: Directories like `.venv/`, `venv/`, `env/`
- **Cache Directories**: `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, etc.
- **Build Artifacts**: `build/`, `dist/`, `*.egg-info/`, compiled files
- **Logs**: All `.log` files and `logs/` directories
- **Data Files**: Large data files, images, video, audio files
- **Model Checkpoints**: `*.pth`, `*.ckpt`, and output directories like `outputs/`
- **Dependencies**: `node_modules/`, and other dependency directories
- **Deprecated/Archive**: `DEPRECATED/`, `_deprecated/`, `_archive/` directories
- **Large Data Sets**: `data/datasets/`, `data/pseudo_label/`, JSONL files

## Usage

### Option 1: Install and Run Everything
```bash
./install_and_run_seroost.sh
```

### Option 2: Run with Existing Installation
```bash
./run_seroost_indexing.sh
```

### Option 3: Manual Execution
```bash
python setup_seroost_indexing.py
```

## Verification

To verify the configuration is valid:
```bash
python test_seroost_config.py
```

## Customization

To customize the configuration:

1. Modify `seroost_config.json` to adjust include/exclude patterns
2. Test the configuration: `python test_seroost_config.py`
3. Run the indexing: `python setup_seroost_indexing.py`

## Benefits

- **Performance**: Excludes unnecessary files to speed up indexing and searching
- **Relevance**: Focuses on code, documentation, and configuration files
- **Maintainability**: Builds on existing `.gitignore` and `.gitingestignore` patterns
- **Completeness**: Includes all relevant project components

## Troubleshooting

- If Seroost is not installed, the setup script will provide installation instructions
- Ensure the project directory has appropriate read permissions
- Large repositories may take several minutes to index completely
- Monitor disk space during indexing as temporary files are created
