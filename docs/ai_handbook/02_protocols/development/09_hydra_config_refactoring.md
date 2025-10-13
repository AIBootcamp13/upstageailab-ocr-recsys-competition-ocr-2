# **filename: docs/ai_handbook/02_protocols/development/09_hydra_config_refactoring.md**
<!-- ai_cue:priority=medium -->
<!-- ai_cue:use_when=hydra,configuration,refactoring -->

# **Protocol: Hydra Configuration Refactoring**

## **Overview**
This protocol provides a safe, systematic process for refactoring Hydra configuration structures. Due to Hydra's declarative and compositional nature, refactoring requires careful planning to avoid breaking the entire configuration system, which lacks intermediate testable states.

## **Prerequisites**
- Deep understanding of Hydra configuration system and composition rules
- Knowledge of project's current config structure and interdependencies
- Access to Hydra debugging tools and validation commands
- Familiarity with git branching for isolated changes
- Understanding of package headers and resolver conventions

## **Procedure**

### **Step 1: Plan Offline - Define Target State**
Create comprehensive refactoring plan without touching live configuration files:

**Define Clear Objective:**
- State goal explicitly (e.g., "Decompose monolithic db.yaml into separate data and dataloader config groups")
- Follow established patterns like lightning-hydra-template

**Audit Current State:**
- List all affected `.yaml` files (created, moved, modified, deleted)
- Identify code references in runners/ and scripts/ directories
- Map interdependencies and defaults list connections

**Design Target Structure:**
- Create before/after Mermaid diagrams of `configs/` structure
- Write complete contents for all new/modified files
- Define proper package headers for Hydra resolver:

**Root Config Group Files:**
```yaml
# @package _global_
# Main data configuration
```

**Nested Config Files:**
```yaml
# @package _group_.model.decoder
# UNet decoder configuration
```

**Specify New Usage:**
- Define updated defaults lists for main experiment configs
- Provide new command-line invocation examples
- Establish validation command for testing

### **Step 2: Execute Transactionally**
Implement the complete refactoring plan in a single, focused operation:

**Create Dedicated Branch:**
```bash
git checkout -b feature/refactor-hydra-configs
```

**Apply All Changes:**
- Create, modify, and move files exactly as planned
- Execute as single comprehensive set of actions
- Avoid partial or incremental changes

### **Step 3: Validate with Hydra Tools**
Verify new configuration structure using systematic validation approach:

**Run Validation Command:**
```bash
uv run python runners/train.py --config-name train data.limit_val_batches=1
```

**Debug with Hydra Tools:**
- View final composed config: `--cfg job`
- See Hydra's search path: `hydra.verbose=true`
- Compare output against planned structure

**Address Failures:**
- Don't guess fixes - use Hydra tools to identify discrepancies
- Compare actual vs. planned configuration composition
- Verify package headers and resolver paths

### **Step 4: Finalize and Document**
Complete successful refactor and update project references:

**Commit Changes:**
- Validation command succeeds with new structure
- Commit complete refactor as single operation

**Update Documentation:**
- Modify Command Registry for changed script invocations
- Create changelog entry detailing the refactor
- Update any configuration documentation

## **Validation**
- Offline plan is comprehensive and covers all affected files
- All configuration files follow proper package header conventions
- Validation command succeeds with new structure
- Hydra composition matches planned target state
- No silent errors or unexpected configuration behavior

## **Troubleshooting**
- If validation fails, use `--cfg job` to identify composition discrepancies
- When package headers are incorrect, check resolver paths with `hydra.verbose=true`
- For complex refactors, consider smaller incremental changes
- If silent errors occur, verify all interdependencies are properly mapped
- When context becomes contaminated, restart with fresh plan document

## **Related Documents**
- [Configuration Management](../../configuration/01_hydra_config_system.md) - Hydra configuration patterns
- [Modular Refactor](05_modular_refactor.md) - General refactoring principles
- [Command Registry](02_command_registry.md) - Available validation and testing tools
- [Coding Standards](01_coding_standards.md) - Development best practices
