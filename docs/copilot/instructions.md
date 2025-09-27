# Copilot Instructions for DBNet OCR Project

## ⚠️ CRITICAL: Environment Setup
**ALWAYS use UV for package management and environment handling.**

### Environment Requirements
- **Python Version**: >= 3.10 (configured in pyproject.toml)
- **Package Manager**: UV (not pip, conda, or poetry)
- **Virtual Environment**: `.venv/` in workspace root
- **VS Code Interpreter**: `./.venv/bin/python` (configured in .vscode/settings.json)

### Environment Commands
```bash
# Install dependencies
uv sync --group dev

# Run commands
uv run python script.py
uv run pytest tests/
uv run black .
uv run isort .
uv run flake8 .

# Add new dependencies
uv add package-name
uv add --dev dev-package-name
```

### VS Code Configuration
- **Python Interpreter**: Automatically set to `./.venv/bin/python`
- **Terminal**: Automatically activates virtual environment
- **Testing**: pytest with `-v --tb=short` arguments
- **Formatting**: black with 88 character line length
- **Linting**: flake8 with max-line-length=88

### NEVER Use These Commands
- ❌ `pip install` (use `uv add`)
- ❌ `python -m venv` (use `uv sync`)
- ❌ `conda activate` (use uv environment)
- ❌ `source .venv/bin/activate` (terminal auto-activates)

## Context Loading Strategy
Include these files based on your task type:

### For General Development
- `docs/copilot/context.md` - Core architecture and patterns
- `docs/maintenance/project-state.md` - Current status and priorities

### For Data/Model Tasks
- `docs/copilot/data-context.md` - Dataset formats and evaluation
- `docs/copilot/quick-reference.md` - Common implementation patterns

### For New Features
- `docs/development/coding-standards.md` - Implementation guidelines
- `docs/development/naming-conventions.md` - Naming patterns

### For Debugging
- `docs/development/debugging.md` - Debugging workflows
- Recent debug logs in `docs/plans/debug-logs/`

## Key Architecture Patterns
- **Factory Functions**: Use Hydra instantiate for component creation
- **Abstract Bases**: Implement abstract classes for extensibility
- **Registry Pattern**: Architecture registry for plug-and-play
- **Configuration First**: All behavior driven by config files

## Quality Standards
- **Type Hints**: All functions and methods
- **Docstrings**: Google/NumPy format
- **Testing**: Unit tests for all core components
- **Linting**: Black formatting, flake8 linting

## Development Tools
- **icecream**: For debugging (`ic()` instead of `print()`)
- **rich**: For beautiful console output
- **hydra**: For configuration management
- **pytest**: For testing with coverage

## File Organization
```
src/ocr_framework/          # Main package (future)
├── core/                   # Abstract bases and registry
├── architectures/          # Plug-and-play architectures
├── datasets/              # Data loading and processing
├── training/              # Lightning modules
├── evaluation/            # Metrics and evaluation
└── utils/                 # Utilities

tests/                     # Test suite
├── unit/                  # Unit tests
├── integration/           # Integration tests
├── manual/                # Manual tests
├── debug/                 # Debug utilities
└── wandb/                 # W&B integration tests

configs/                   # Hydra configurations
├── architectures/         # Architecture configs
└── experiments/           # Experiment configs
```

## Common Patterns
- Use `Path` objects for file operations
- Handle exceptions with proper logging
- Use dataclasses for configuration objects
- Implement `__repr__` for debugging</content>
<parameter name="filePath">/home/vscode/workspace/upstage-receipt-text-detection-dbnet-baseline/docs/copilot/instructions.md
