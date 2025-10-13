# **filename: docs/ai_handbook/02_protocols/development/05_modular_refactor.md**
<!-- ai_cue:priority=medium -->
<!-- ai_cue:use_when=refactor,architecture,modularity -->

# **Protocol: Modular Refactoring**

## **Overview**
This protocol provides the authoritative workflow for modular refactors to increase project health and development velocity. The focus is on creating clean interfaces, independent components, and maintainable architecture through systematic refactoring.

## **Prerequisites**
- Understanding of project architecture and component relationships
- Familiarity with Hydra configuration system and registry patterns
- Knowledge of base classes in `ocr/models/core/base_classes.py`
- Access to testing framework and validation scripts
- Clear identification of refactoring motivation and scope

## **Procedure**

### **Step 1: Analyze & Plan the Refactor**
Establish clear objectives and target architecture before making changes:

**Clarify Motivation:**
- Capture the "why" (tech debt, new feature, performance regression)
- Identify code smells: large files, mixed responsibilities, tight coupling, duplicated logic

**Review Architecture:**
- Consult [Architecture](../../references/architecture/01_architecture.md)
- Understand component relationships and dependencies
- Define scope and target module boundaries

**Baseline Current Behavior:**
- Run verification tests: unit tests, smoke tests (`trainer.fast_dev_run=true`)
- Document current functionality for regression testing
- Establish success criteria for the refactor

### **Step 2: Execute Incrementally**
Implement changes in small, testable steps while preserving functionality:

**Work on Dedicated Branch:**
- Create feature branch to isolate changes
- Keep commit history clean and focused

**Extract, Don't Rewrite:**
- Move working code into new module structure
- Delay cleanup or optimization until behavior is preserved
- Maintain backward compatibility during transition

**Commit & Test Frequently:**
- Run focused tests after each extraction
- Commit only when each step is stable
- Use temporary adapters if needed for gradual migration

### **Step 3: Validate Continuously**
Ensure changes maintain system integrity throughout the process:

**Run Targeted Tests:**
- Execute unit tests for touched modules
- Run integration tests for affected components
- Validate configuration loading and model instantiation

**Monitor Baseline Parity:**
- Re-run baseline verification from Step 1
- Catch regressions early with frequent validation
- Compare performance metrics if applicable

### **Step 4: Finalize & Document**
Complete the refactor and update all references:

**Update References & Configs:**
- Point Hydra configs to new module locations
- Update registry registrations in `ocr/models/core/registry.py`
- Modify imports across the codebase

**Clean Up Legacy Code:**
- Remove superseded files and functions once new implementation is stable
- Eliminate temporary compatibility wrappers
- Update any remaining aliases to point to new locations

**Document Changes:**
- Update AI Handbook references and architecture docs
- Add changelog entry in `docs/ai_handbook/05_changelog/`
- Update any relevant component documentation

## **Validation**
- [ ] **Analysis:** Scope, motivation, and target architecture documented
- [ ] **Planning:** New module layout and registry/config updates drafted
- [ ] **Implementation:** Components inherit from correct base classes, registries updated, Hydra configs adjusted
- [ ] **Testing:** Unit tests and smoke tests pass, no regressions detected
- [ ] **Documentation:** Handbook and changelog updated, obsolete code removed
- [ ] **Compatibility:** Temporary adapters reviewed, minimal bridging layer remains

## **Troubleshooting**
- If tests fail after extraction, check import paths and dependencies
- When components don't inherit properly, verify base class compatibility
- If registry updates cause instantiation errors, validate configuration syntax
- For complex refactors, consider smaller incremental changes
- When performance degrades, profile before and after changes

## **Related Documents**
- [Architecture Reference](../../references/architecture/01_architecture.md) - System architecture overview
- [Utility Adoption](04_utility_adoption.md) - Code reuse and DRY principles
- [Refactoring Guide](10_refactoring_guide.md) - Advanced refactoring techniques
- [Coding Standards](01_coding_standards.md) - Development best practices
- [Configuration Management](../../configuration/01_hydra_config_system.md) - Hydra configuration patterns
