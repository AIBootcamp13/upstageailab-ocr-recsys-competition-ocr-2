# Qwen Code Assistant Entry Point

Welcome! This document serves as your entry point for working with the Qwen Code Assistant on this OCR project.

## Project Overview

This is an OCR (Optical Character Recognition) project focused on receipt text detection using DBNet architecture. The system is built with PyTorch Lightning and Hydra for modular experimentation. The project emphasizes:

- Modular architecture with plug-and-play components (encoders, decoders, heads, losses)
- Configuration-driven approach using Hydra
- Type hints for all public APIs
- UV for package management

## Key Directories and Files

### Documentation
- `docs/ai_handbook/index.md` - Your primary source of truth with comprehensive project documentation
- `docs/ai_handbook/01_onboarding/01_setup_and_tooling.md` - Start here if new to the project
- `docs/ai_handbook/02_protocols/command_builder_testing_guide.md` - Testing guide for Command Builder module

### Setup Scripts
- `scripts/setup/` - Contains numbered setup scripts for environment setup, linting, and aliases
- `pyproject.toml` - Project dependencies and build configuration
- `uv.lock` - Lock file for UV package manager

### Utilities
- `run_ui.py` - Streamlit applications for evaluation viewer, inference, command builder, and resource monitor
- `.env.template` - Environment variable template

### Command Builder Module (New Modular Structure)
- `ui/utils/command/` - Modular command building utilities
  - `models.py` - Data models for command parameters
  - `quoting.py` - Hydra/shell quoting utilities
  - `builder.py` - Command construction logic
  - `executor.py` - Command execution and process management
  - `validator.py` - Command validation
- **Important**: Use `from ui.utils.command import CommandBuilder` for new development (not the old location)

## Working Guidelines

### AI Cue Markers
Look for HTML comments like `<!-- ai_cue:priority=high -->` in documentation files. These markers indicate priority levels and scenarios that should trigger loading those documents first.

### Safe Operations
- Always use `uv run` prefix for Python commands
- Follow the documented protocols in the AI handbook
- Use pre-commit hooks for quality checks
- Reference the command registry for authorized scripts

### Context Bundles
For specific tasks, refer to the following context bundles in the AI handbook:
- New features: Coding Standards → Architecture → Hydra & Registry
- Debugging: Debugging Workflow → Command Registry → Experiment Logs
- Utilities: Utility Adoption Guide → Existing Utility Functions
- Streamlit apps: Command Registry → run_ui.py → Streamlit maintenance bundle

## Getting Started

1. Read the AI handbook at `docs/ai_handbook/index.md` for comprehensive project information
2. Follow the setup instructions in `docs/ai_handbook/01_onboarding/01_setup_and_tooling.md`
3. Use the setup scripts in `scripts/setup/` in numerical order (00, 01, 02)
4. Run `uv run python run_ui.py --help` to see available UI options

## Quality Standards

- Ruff for formatting and linting
- Pytest for testing with coverage
- Comprehensive docstrings and type hints
- Pre-commit hooks for automated quality checks

For detailed information on any aspect of the project, navigate to the complete AI handbook at `docs/ai_handbook/index.md` which serves as the single source of truth for all project knowledge.
