# **filename: docs/ai_handbook/02_protocols/development/01_coding_standards.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=code_changes,standards,development -->

# **Protocol: Coding Standards & Naming Conventions**

This protocol provides the standardized approach for maintaining coding standards and naming conventions across the project to ensure consistency, readability, and maintainability.

## **Overview**

This protocol establishes the coding standards, formatting requirements, and naming conventions that all contributors must follow. These standards ensure code consistency across the project, improve readability, and reduce maintenance overhead. The standards are enforced through automated tools and CI/CD validation.

## **Prerequisites**

- Familiarity with Python development and PEP 8 standards
- Understanding of the project's architecture (see: `docs/ai_handbook/03_references/architecture/01_architecture.md`)
- Access to development environment with required dependencies
- Knowledge of type hinting and modern Python practices

## **Procedure**

### **Step 1: Code Formatting & Linting Setup**
Configure your development environment with the required tools:
- Install Ruff: `uv add --dev ruff`
- Set up pre-commit hooks: `pre-commit install`
- Configure IDE to use Ruff formatter and linter

### **Step 2: Apply Coding Standards**
Follow these standards when writing or modifying code:

**Code Formatting:**
- Use Ruff Formatter (or Black with line length 88)
- All formatting is handled automatically via pre-commit hooks
- Manual application: `uv run ruff check . --fix && uv run ruff format .`

**Import Organization:**
- Imports organized by Ruff into three groups: standard library, third-party, and local application imports
- One import per line
- Alphabetical ordering within each group

### **Step 3: Type Hinting Implementation**
Apply comprehensive type hinting:
- All public functions, methods, and class attributes must include type hints
- Use specific types from typing module (Dict, List, Optional, Tuple, Union)
- Private methods should also be typed for internal consistency

### **Step 4: Naming Convention Application**
Apply consistent naming conventions following PEP 8:

**Modules & Packages:** snake_case (e.g., ocr_framework, data_loader.py)
**Classes:** PascalCase (e.g., OCRLightningModule, TimmBackbone)
- Abstract Base Classes start with Base (e.g., BaseEncoder)
- Exceptions end with Error (e.g., ConfigurationError)
**Functions & Methods:** snake_case (e.g., validate_polygons, training_step)
- Private methods prefixed with single underscore (e.g., _preprocess_image)
**Variables & Attributes:** snake_case (e.g., learning_rate, self.batch_size)
**Constants:** UPPER_SNAKE_CASE (e.g., MAX_IMAGE_SIZE, DEFAULT_THRESHOLD)

## **Validation**

Run the following validation checks:

```bash
# Code quality validation
uv run ruff check docs/ai_handbook/02_protocols/development/01_coding_standards.md
uv run ruff format --check docs/ai_handbook/02_protocols/development/01_coding_standards.md

# Type checking (if applicable)
uv run mypy --ignore-missing-imports docs/ai_handbook/02_protocols/development/01_coding_standards.md

# Template compliance
python scripts/validate_templates.py docs/ai_handbook/_templates docs/ai_handbook
```

## **Troubleshooting**

### **Common Issues**
- **Formatting Failures**: Run `uv run ruff format .` to auto-fix formatting issues
- **Linting Errors**: Use `uv run ruff check . --fix` to auto-fix common linting issues
- **Type Checking Failures**: Review type hints and update imports as needed
- **Pre-commit Hook Failures**: Ensure all dependencies are installed with `uv sync`

### **Debugging Steps**
1. Check Ruff version: `uv run ruff --version`
2. Validate configuration: `uv run ruff check --show-settings`
3. Test individual files: `uv run ruff check path/to/file.py`
4. Review CI/CD logs for specific error details

## **Related Documents**

- `docs/ai_handbook/02_protocols/governance/18_documentation_governance_protocol.md` - Documentation standards
- `docs/ai_handbook/02_protocols/development/03_debugging_workflow.md` - Debugging procedures
- `docs/ai_handbook/02_protocols/development/05_modular_refactor.md` - Refactoring guidelines
- `docs/ai_handbook/03_references/architecture/01_architecture.md` - Project architecture
- `docs/ai_handbook/_templates/development.md` - Development template

---

*This document follows the development protocol template. Last updated: October 13, 2025*
