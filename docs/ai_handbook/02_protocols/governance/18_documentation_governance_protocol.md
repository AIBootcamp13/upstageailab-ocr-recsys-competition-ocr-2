# **filename: docs/ai_handbook/02_protocols/governance/18_documentation_governance_protocol.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=governance,compliance,standards,documentation -->

# **Protocol: Documentation Governance**

This protocol establishes the governance framework for documentation management to ensure consistency, quality, and compliance across the AI Handbook project.

## **Overview**

This protocol defines the rules for managing the `docs/ai_handbook/` directory structure. All contributors MUST adhere to this protocol for file creation, modification, and deletion tasks. The master schema is defined in `docs/ai_handbook/index.json` and serves as the authoritative source for directory structure and naming conventions.

## **Prerequisites**

- Understanding of project documentation structure
- Access to `docs/ai_handbook/index.json` schema
- Knowledge of Markdown formatting standards
- Familiarity with validation scripts

## **Governance Rules**

### **Rule 1: Adhere to the Central Schema**
Before adding or moving a file, you MUST consult the "schema" section within `docs/ai_handbook/index.json`. This schema defines the required naming convention, purpose, and allowed content for each top-level directory.

### **Rule 2: Do Not Mix Artifact Types**
Documentation directories (01_onboarding, 02_protocols, etc.) MUST ONLY contain Markdown (.md) documents.

* **PROHIBITED:** You MUST NOT save generated outputs (images, logs, checkpoints, JSON files) inside the `docs/ai_handbook/` directories. All generated outputs belong in the project's top-level `outputs/` directory.
* **PROHIBITED:** You MUST NOT store planning documents, work-in-progress notes, or temporary files within the handbook. These belong in a separate project management directory.

### **Rule 3: Follow Naming and Numbering Conventions**
All new documents within a numbered directory (e.g., 01_onboarding, 02_protocols) MUST follow the `NN_descriptive_name.md` format, where NN is the next sequential number in that directory.

### **Rule 4: Update the Manifest**
After creating a new document, you MUST add a corresponding entry to `docs/ai_handbook/index.json`. Ensure you provide a unique id, title, path, and other relevant metadata.

### **Rule 5: Validate Your Changes**
After modifying the handbook, run the validation script to ensure integrity.

## **Procedure**

### **Step 1: Assessment**
Before making changes, review the current schema in `docs/ai_handbook/index.json` and identify the appropriate directory and naming convention for your content.

### **Step 2: Implementation**
Create or modify documentation following the established naming conventions and directory structure. Ensure content follows the appropriate template format.

### **Step 3: Validation**
Run validation scripts to ensure compliance with governance rules and formatting standards.

### **Step 4: Monitoring**
Regularly review documentation health through automated validation reports and manual audits.

## **Compliance Validation**

```bash
# Documentation governance compliance check
uv run python scripts/agent_tools/validate_manifest.py

# Template validation
python scripts/validate_templates.py docs/ai_handbook/_templates docs/ai_handbook
```

## **Enforcement**

### **Automated Checks**
- Pre-commit hooks for format compliance
- CI/CD pipeline validation for documentation changes
- Automated template validation requirements

### **Manual Review**
- Code review requirements for documentation changes
- Documentation review process for new content
- Exception approval process for urgent changes

## **Troubleshooting**

### **Common Governance Issues**
- **Schema Non-compliance**: Review `docs/ai_handbook/index.json` and update accordingly
- **Naming Convention Violations**: Use the next sequential number in the target directory
- **Mixed Artifact Types**: Move generated outputs to appropriate directories

### **Escalation Path**
1. Consult team lead for immediate clarification
2. Escalate to documentation governance committee for policy changes
3. Document exceptions and rationale in the change log

## **Related Documents**

- `docs/ai_handbook/02_protocols/governance/19_streamlit_maintenance_protocol.md` - Maintenance procedures
- `docs/ai_handbook/02_protocols/governance/20_bug_fix_protocol.md` - Issue resolution
- `docs/ai_handbook/_templates/governance.md` - Governance template
- `docs/ai_handbook/index.json` - Master schema definition

---

*This document follows the governance protocol template. Last updated: October 13, 2025*
