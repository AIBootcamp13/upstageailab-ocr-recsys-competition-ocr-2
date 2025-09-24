# DBNet OCR Project - Copilot Instructions

## Project Overview
Receipt text detection using DBNet architecture with PyTorch Lightning. Modular design supporting plug-and-play architectures for experimentation.

## Key Guidelines
- **Configuration**: Use Hydra for all configurations
- **Dependencies**: Use UV for dependency management
- **Architecture**: Follow modular encoder/decoder/head/loss pattern
- **Code Style**: Type hints, docstrings, snake_case naming
- **Testing**: Comprehensive unit tests with pytest

## Development Workflow
1. Check `docs/maintenance/project-state.md` for current status
2. Review active plans in `docs/plans/`
3. Include relevant context files for your task
4. Follow coding standards from `docs/development/`
5. Update project state after significant changes

## Context Files
- `docs/copilot/context.md` - Core project understanding
- `docs/copilot/data-context.md` - Data processing and evaluation
- `docs/copilot/quick-reference.md` - Common patterns and utilities
- `docs/maintenance/project-state.md` - Current project status

## Architecture Patterns
- Use factory functions for component instantiation
- Implement abstract base classes for extensibility
- Separate configuration from implementation
- Use registry pattern for plug-and-play architectures

## Quality Standards
- All public functions need type hints and docstrings
- Unit test coverage for core components
- Integration tests for end-to-end pipelines
- Use icecream for debugging, rich for console output