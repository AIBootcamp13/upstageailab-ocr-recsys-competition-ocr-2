# **filename: docs/ai_handbook/_templates/governance.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=governance,compliance,standards -->

# **Protocol: {{protocol_title}}**

This protocol establishes the governance framework for {{governance_focus}} to ensure consistency, quality, and compliance across the project.

## **Overview**

{{overview_content}}

## **Prerequisites**

- Understanding of project governance structure
- Access to relevant governance documentation
- Knowledge of applicable standards and requirements

## **Governance Rules**

### **Rule 1: {{rule1_title}}**
{{rule1_content}}

### **Rule 2: {{rule2_title}}**
{{rule2_content}}

### **Rule 3: {{rule3_title}}**
{{rule3_content}}

## **Procedure**

### **Step 1: Assessment**
{{step1_content}}

### **Step 2: Implementation**
{{step2_content}}

### **Step 3: Validation**
{{step3_content}}

### **Step 4: Monitoring**
{{step4_content}}

## **Compliance Validation**

```bash
# Governance compliance check
uv run python scripts/validate_governance.py {{governance_area}}

# Automated validation
uv run python scripts/check_compliance.py {{validation_target}}
```

## **Enforcement**

### **Automated Checks**
- Pre-commit hooks for format compliance
- CI/CD pipeline validation
- Automated testing requirements

### **Manual Review**
- Code review requirements
- Documentation review process
- Exception approval process

## **Troubleshooting**

### **Common Governance Issues**
- **Non-compliance**: Follow remediation procedures outlined in related documents
- **Ambiguous Requirements**: Consult governance leads for clarification
- **Process Bottlenecks**: Use exception processes for urgent changes

### **Escalation Path**
1. Consult team lead for immediate issues
2. Escalate to governance committee for policy changes
3. Document exceptions and rationale

## **Related Documents**

- `docs/ai_handbook/02_protocols/governance/18_documentation_governance_protocol.md` - Documentation governance
- `docs/ai_handbook/02_protocols/governance/19_streamlit_maintenance_protocol.md` - Maintenance procedures
- `docs/ai_handbook/02_protocols/governance/20_bug_fix_protocol.md` - Issue resolution

---

*This document follows the governance protocol template. Last updated: {{last_updated}}*
