<!-- This is a Markdown file for GitHub Copilot instructions -->

# Co-Instructions for AI Agents

## Primary Directive
**Start here for all tasks:** Navigate to `docs/ai_handbook/index.md` for the complete, authoritative knowledge base.

## Quick Context
This is an OCR project using DBNet architecture for receipt text detection. The system is built with PyTorch Lightning and Hydra for modular experimentation.

## Key Principles
- **Single Source of Truth:** All documentation lives in `docs/ai_handbook/`
- **Modular Architecture:** Plug-and-play components (encoders, decoders, heads, losses)
- **Configuration-Driven:** Use Hydra configs for all instantiations
- **UV Package Management:** Never use pip/conda directly
- **Type Hints Required:** All public APIs must have complete type annotations

### AI Cue Markers
- Protocol files may include HTML comments such as `<!-- ai_cue:priority=high -->` or `<!-- ai_cue:use_when=debugging -->`.
- Treat `priority` as the relative importance for triage; `use_when` lists scenarios that should trigger loading that document first.
- When scanning docs, prioritise sections with `priority=high` before moving to medium/low markers.

## Essential Starting Points
1. **New to Project:** Read `docs/ai_handbook/01_onboarding/01_setup_and_tooling.md`
2. **Adding Features:** Follow `docs/ai_handbook/02_protocols/01_coding_standards.md`
3. **Debugging Issues:** Use `docs/ai_handbook/02_protocols/03_debugging_workflow.md`
4. **Architecture Questions:** Check `docs/ai_handbook/03_references/01_architecture.md`

## Safe Operations
- Use `uv run` prefix for all Python commands
- Refer to `docs/ai_handbook/02_protocols/02_command_registry.md` for approved scripts
- Update `docs/ai_handbook/05_changelog/` after significant changes

## Context Bundles for Common Tasks
- **Model Development:** Architecture + Coding Standards + Hydra Registry
- **Data Pipeline:** Data Overview + Evaluation Metrics
- **Experimentation:** Experiment Template + Debugging Workflow

## Quality Standards
- Ruff for formatting and linting
- Pytest for testing with coverage
- Pre-commit hooks for automated quality checks
- Comprehensive docstrings and type hints

## Emergency Contacts
If stuck, reference the Command Registry or create an issue with full context from the handbook.
