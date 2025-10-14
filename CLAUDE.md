# Claude Code Protocol

**Purpose**: Remind Claude to consult project documentation before and after making changes.

**Key Principle**: Documentation exists to prevent mistakes. Use it proactively.

---

## Quick Reference

**Primary Documentation Index**: [`docs/ai_handbook/index.md`](docs/ai_handbook/index.md)

All documentation paths, context bundles, and usage scenarios are maintained in the AI Handbook. Refer to it for:
- Task-specific context bundles (feature development, debugging, training, etc.)
- Complete protocol library with AI cue markers
- Architecture references and guides
- Command registry for approved scripts
- Naming conventions and file organization (defined in [`index.json`](docs/ai_handbook/index.json))

---

## Before Making Changes

1. **Check the relevant protocol** in [`docs/ai_handbook/02_protocols/`](docs/ai_handbook/02_protocols/)
   - Development: Coding standards, debugging, refactoring
   - Configuration: Hydra troubleshooting, testing
   - Components: Training, preprocessing
   - Governance: Documentation updates

2. **Review architecture docs** in [`docs/ai_handbook/03_references/architecture/`](docs/ai_handbook/03_references/architecture/) if modifying:
   - System structure (`01_architecture.md`)
   - Configuration system (`02_hydra_and_registry.md`)
   - Data structures → also check [`docs/pipeline/data_contracts.md`](docs/pipeline/data_contracts.md)

3. **Check performance guides** in [`docs/ai_handbook/03_references/guides/`](docs/ai_handbook/03_references/guides/) if optimizing:
   - Cache system: `cache-management-guide.md`
   - Mixed precision: `fp16-training-guide.md`
   - Profiling: `performance_profiler_usage.md`
   - Presets: [`configs/data/performance_preset/README.md`](configs/data/performance_preset/README.md)

---

## After Making Changes

1. **Update changelog**: [`docs/CHANGELOG.md`](docs/CHANGELOG.md)
2. **Create dated entry**: `docs/ai_handbook/05_changelog/YYYY-MM/DD_description.md`
3. **Create bug report** (if fixing a bug): `docs/bug_reports/BUG_YYYY_NNN_DESCRIPTION.md`
4. **Update protocols** (if new patterns emerged)

---

## Common Scenarios

See [`docs/ai_handbook/index.md`](docs/ai_handbook/index.md) Section 2 for detailed context bundles:

- **New Feature**: Coding Standards → Architecture → Hydra Registry
- **Debugging**: Debugging Workflow → Command Registry → Past Experiments
- **Training Run**: Training Protocol → `propose_next_run.py`
- **Configuration**: Hydra Reference → Config Resolution Troubleshooting
- **Performance**: Cache Guide → FP16 Guide → Profiler Usage

---

## Critical Reminders

- **Data structures**: Always check [`docs/pipeline/data_contracts.md`](docs/pipeline/data_contracts.md) before modifying
- **Naming conventions**: Defined in [`docs/ai_handbook/index.json`](docs/ai_handbook/index.json) schema section
- **Performance features**: Disabled by default, use presets to enable (see [`configs/data/performance_preset/`](configs/data/performance_preset/))
- **AI cue markers**: Protocol files contain `<!-- ai_cue:priority=high -->` for context loading guidance

---

## When Documentation is Missing

1. Check if it should exist per [`index.json`](docs/ai_handbook/index.json) schema
2. Create in appropriate location following naming conventions
3. Update index if needed
4. Document in changelog

---

**Remember**: The AI Handbook ([`docs/ai_handbook/index.md`](docs/ai_handbook/index.md)) is the single source of truth. This file is just a reminder to use it.
