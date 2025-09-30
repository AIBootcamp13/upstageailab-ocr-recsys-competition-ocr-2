# **filename: docs/ai_handbook/refactor_plan.md**


# **AI Documentation Refactor & Enhancement Plan**

This document outlines a phased plan to refactor the project's documentation into a centralized, efficient, and scalable knowledge base for AI agents, incorporating best practices from both internal and external assessments. Our guiding principle is a systematic, numbered hierarchy for clarity and navigation.

## **Phase 1: Foundation & Structure (1 Week)**

**Goal:** Establish the new, authoritative directory structure and the central index file.

1. **Create New Directory:**
   * Initialize the new documentation root: docs/ai_handbook/.
2. **Build Subdirectories:**
   * Create the following folder structure inside docs/ai_handbook/:
     * 01_onboarding/ (For project overview, architecture, and setup)
     * 02_protocols/ (For task-oriented, step-by-step guides)
     * 03_references/ (For stable, factual information)
     * 04_experiments/ (For logging experiment runs and results)
     * 05_changelog/ (For tracking significant project state changes)
3. **Draft the Central Index (index.md):**
   * Create docs/ai_handbook/index.md. This file is the primary entry point.
   * Populate it with a project overview, a global Table of Contents, and sections for "Context Bundles" and the "Command Registry."

## **Phase 2: Content Migration & Consolidation (1-2 Weeks)**

**Goal:** Migrate existing documentation into the new structure, eliminating redundancy.

1. **Content Migration Map:**

| Old File(s) | New File | Purpose |
| :---- | :---- | :---- |
| docs/copilot/context.md, architecture-overview.md, component-diagrams.md | 03_references/01_architecture.md | The single source of truth for the model and system architecture. |
| docs/copilot/instructions.md, .github/copilot-instructions.md | index.md (Quick Start section) | The new instruction is simple: "Start at index.md." |
| docs/development/coding-standards.md, naming-conventions.md | 02_protocols/01_coding_standards.md | Step-by-step guide for developers on writing and contributing code. |
| docs/copilot/data-context.md | 01_onboarding/02_data_overview.md | High-level context about the dataset, formats, and challenges. |
| docs/development/pydantic-policy.md, registry-compatibility.md | 03_references/02_hydra_and_registry.md | Detailed reference on how Hydra and the component registry work. |
| docs/copilot/quick-reference.md | 03_references/03_utility_functions.md | API-style reference for key reusable functions. |
| docs/maintenance/project-state.md | 05_changelog/YYYY-MM-DD_state.md | Project state updates become entries in a versioned changelog. |
| docs/plans/* | _archive/docs/plans/ | Archive historical planning documents. |

2. **Archive Old Documentation:**
   * Move the old docs/ subdirectories into an _archive/ folder to prevent confusion.

## **Phase 3: Lightweight Tooling & Automation (Ongoing)**

**Goal:** Create lightweight, AI-callable tools for safe, autonomous execution.

1. **Create a Command Registry:**
   * Create 02_protocols/02_command_registry.md.
   * Document a curated list of safe-to-run scripts, including purpose, arguments, expected output, and a note on dry-run modes.
2. **Develop Script Wrappers:**
   * Create a new scripts/agent_tools/ directory.
   * Add simple Python wrappers for common tasks (e.g., validate_config.py, list_checkpoints.py).
   * Ensure all relevant scripts support --dry-run, limit_batches, or similar flags for safe inspection.
3. **Update index.md:**
   * Link to the new command_registry.md from the main index for easy discoverability.

## **Phase 4: Process & Cadence (Ongoing)**

**Goal:** Implement routines to keep the documentation accurate and in sync with the codebase.

1. **Define Context Bundles:**
   * In index.md, create task-specific "Context Bundles" that list the essential documents for recurring workflows.
2. **Establish Versioning Routine:**
   * Formalize the process: When the project state is updated, a new entry is created in 05_changelog/. The index.md is updated to point to the latest entry.
3. **Implement Review Cadence:**
   * Schedule a recurring monthly task to audit the documentation. The checklist includes validating links in index.md, updating the command registry, and pruning obsolete information.
4. **Dedicated Experiment & Run Docs:**
   * Create 04_experiments/TEMPLATE.md to standardize how new experiment results are logged.
   * This ensures that configuration, results, learnings, and links to outputs (W&B, etc.) are captured consistently.
