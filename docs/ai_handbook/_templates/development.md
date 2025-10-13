# **filename: docs/ai_handbook/_templates/development.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=code_changes,refactoring,debugging -->

# **Protocol: {{protocol_title}}**

This protocol provides the standardized approach for {{development_focus}} development activities in the project.

## **Overview**

{{overview_content}}

## **Prerequisites**

- Familiarity with the project's architecture (see: `docs/ai_handbook/03_references/architecture/01_architecture.md`)
- Understanding of coding standards (see: `docs/ai_handbook/02_protocols/development/01_coding_standards.md`)
- Access to development environment with required dependencies

## **Procedure**

### **Step 1: Preparation**
{{step1_content}}

### **Step 2: Implementation**
{{step2_content}}

### **Step 3: Testing & Validation**
{{step3_content}}

### **Step 4: Documentation Update**
{{step4_content}}

## **Validation**

Run the following validation checks:

```bash
# Code quality validation
uv run ruff check {{target_files}}
uv run ruff format --check {{target_files}}
uv run mypy {{target_files}}

# Functional testing
uv run pytest {{test_files}} -v
```

## **Troubleshooting**

### **Common Issues**
- **Import Errors**: Ensure all dependencies are installed with `uv sync`
- **Type Checking Failures**: Review type hints and update as needed
- **Test Failures**: Check test setup and fixtures

### **Debugging Steps**
1. Run tests in isolation: `uv run pytest {{specific_test}} -v -s`
2. Check logs for detailed error information
3. Validate configuration files if applicable

## **Related Documents**

- `docs/ai_handbook/02_protocols/development/01_coding_standards.md` - Coding standards
- `docs/ai_handbook/02_protocols/development/03_debugging_workflow.md` - Debugging procedures
- `docs/ai_handbook/02_protocols/development/05_modular_refactor.md` - Refactoring guidelines

---

*This document follows the development protocol template. Last updated: {{last_updated}}*
