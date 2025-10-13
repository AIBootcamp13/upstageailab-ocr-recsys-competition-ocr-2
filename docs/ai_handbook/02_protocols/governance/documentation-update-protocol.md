# **filename: docs/ai_handbook/02_protocols/governance/documentation-update-protocol.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=documentation,updates,governance -->

# **Protocol: Documentation Update**

This protocol establishes the governance framework for documentation updates to ensure consistent maintenance, timely updates, and quality standards across all project documentation.

## **Overview**

This protocol provides clear guidelines for instructing AI agents about which documentation files to update and when. Use this protocol to ensure consistent documentation maintenance across the project and maintain high-quality, up-to-date documentation.

## **Prerequisites**

- Understanding of project documentation hierarchy
- Knowledge of change types and their documentation requirements
- Access to validation tools and testing procedures
- Familiarity with documentation standards and quality checks

## **Governance Rules**

### **Rule 1: Document Hierarchy**
Follow the established hierarchy: primary documentation (README.md), component-specific docs, operational documentation, and development documentation with appropriate update frequencies.

### **Rule 2: Update Triggers**
Immediate updates for breaking changes and major features, after-testing updates for bug fixes and enhancements, weekly reviews for outdated information, and pre-release reviews for comprehensive validation.

### **Rule 3: Agent Instruction Format**
Use structured format for documentation update requests including target document, change type, context, update scope, validation criteria, and detailed description.

### **Rule 4: Quality Standards**
All documentation must include entry points, prerequisites, validation steps, troubleshooting information, and concrete examples.

### **Rule 5: Agent Response Protocol**
Agents must confirm understanding, identify changes, provide summaries, validate updates, and report completion with recommendations.

## **Procedure**

### **Step 1: Assessment**
Evaluate the change type, impact, and documentation requirements based on the established hierarchy and update triggers.

### **Step 2: Implementation**
Use the structured request format to specify target documents, change types, context, scope, and validation criteria.

### **Step 3: Validation**
Run quality checks including code example testing, prerequisite validation, cross-reference checking, and link accessibility verification.

### **Step 4: Monitoring**
Conduct regular reviews and emergency updates for critical issues following established protocols.

## **Compliance Validation**

```bash
# Documentation update compliance check
python scripts/validate_templates.py docs/ai_handbook/_templates docs/ai_handbook
uv run python scripts/agent_tools/validate_manifest.py
# Check for broken links and validate examples
```

## **Enforcement**

### **Automated Checks**
- Pre-commit hooks for documentation validation
- CI/CD pipeline checks for link integrity and format compliance
- Automated quality checks for examples and prerequisites

### **Manual Review**
- Documentation review requirements for all updates
- Quality assurance checklist completion
- Exception approval for urgent documentation updates

## **Troubleshooting**

### **Common Governance Issues**
- **Missing Prerequisites**: Ensure all requirements are documented before usage instructions
- **Outdated Examples**: Test all code examples and update as needed
- **Broken Cross-references**: Validate all internal and external links
- **Incomplete Validation**: Include verification steps for all procedures

### **Escalation Path**
1. Consult documentation maintainer for complex updates
2. Escalate to governance committee for standard changes
3. Use emergency protocol for critical documentation issues

## **Related Documents**

- `docs/ai_handbook/02_protocols/governance/18_documentation_governance_protocol.md` - Documentation governance
- `docs/ai_handbook/02_protocols/governance/19_streamlit_maintenance_protocol.md` - Maintenance procedures
- `docs/ai_handbook/_templates/governance.md` - Governance template
- `docs/ai_handbook/index.json` - Master schema definition

---

*This document follows the governance protocol template. Last updated: October 13, 2025*
