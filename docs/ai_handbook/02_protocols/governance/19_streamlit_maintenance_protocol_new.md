# **filename: docs/ai_handbook/02_protocols/governance/19_streamlit_maintenance_protocol_new.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=streamlit,maintenance,governance -->

# **Protocol: Streamlit Maintenance**

This protocol establishes the governance framework for Streamlit-based UI modules to ensure consistent maintenance, reliability, and compatibility across all Streamlit applications in the repository.

## **Overview**

This protocol codifies the recurring maintenance loop for Streamlit-based UI modules. Follow it whenever you touch a Streamlit app without a large architectural refactor. The canonical implementation is the OCR inference experience, but the guardrails apply to any app living under `ui/apps/`.

## **Prerequisites**

- Active project environment with `uv` package manager
- Understanding of Streamlit application structure
- Access to UI configuration files and schemas
- Knowledge of validation scripts and testing procedures

## **Governance Rules**

### **Rule 1: Scope and Ownership**
All Streamlit surfaces live in `ui/`. Each runnable app sits under `ui/apps/<app_name>/` with a thin façade. The module consumes UI configuration, runtime defaults, schemas, and UI metadata from designated locations.

### **Rule 2: Trigger Conditions**
Run this protocol whenever Streamlit dependency upgrades, model changes, configuration updates, UI regressions, or periodic maintenance occur.

### **Rule 3: Environment Standards**
Always activate project environment, ensure CUDA visibility, and confirm required backing assets exist before maintenance.

### **Rule 4: Documentation Requirements**
Append changelog entries, notify AI assistants, and update documentation for any behavioral changes.

### **Rule 5: Anti-Patterns**
Never hard-code paths, bypass schema validation, edit without testing, or leave mock inference as default.

## **Procedure**

### **Step 1: Assessment**
Record current git status, export metadata, and identify the scope of maintenance needed based on trigger conditions.

### **Step 2: Implementation**
Follow the maintenance checklist: sanity sweep, data health checks, real inference validation, configuration integrity, and documentation updates.

### **Step 3: Validation**
Re-run the Streamlit app, exercise all features, inspect logs, and run targeted smoke tests to ensure functionality.

### **Step 4: Monitoring**
Verify AI_DOCS markers, run validation scripts, and ensure documentation hooks remain synchronized.

## **Compliance Validation**

```bash
# Streamlit maintenance compliance check
uv run ruff check ui/apps/inference ui/utils/inference_engine.py
uv run python -m compileall ui
uv run python scripts/agent_tools/validate_yaml.py configs/ui/<app_name>.yaml
uv run python scripts/agent_tools/validate_manifest.py
```

## **Enforcement**

### **Automated Checks**
- Pre-commit hooks for linting and syntax validation
- CI/CD pipeline checks for Streamlit applications
- Automated schema validation requirements

### **Manual Review**
- Code review requirements for UI changes
- Manual testing of all Streamlit features
- Exception approval for urgent maintenance

## **Troubleshooting**

### **Common Governance Issues**
- **Mock Inference Fallback**: Inspect logs in `ui/utils/inference_engine.py` and verify catalogue health
- **Schema Mismatches**: Update compatibility schemas and add new families as needed
- **Configuration Drift**: Sync YAML configs with schemas and UI metadata
- **Dependency Issues**: Regenerate lockfile and run pre-commit hooks

### **Escalation Path**
1. Consult UI maintainer for immediate issues
2. Escalate to development team for architectural changes
3. Document exceptions and update this protocol as needed

## **Related Documents**

- `docs/ai_handook/02_protocols/governance/18_documentation_governance_protocol.md` - Documentation governance
- `docs/ai_handbook/02_protocols/governance/20_bug_fix_protocol.md` - Issue resolution
- `docs/ai_handbook/02_protocols/components/12_streamlit_refactoring_protocol.md` - Refactoring procedures
- `docs/ai_handbook/_templates/governance.md` - Governance template

---

*This document follows the governance protocol template. Last updated: October 13, 2025*
