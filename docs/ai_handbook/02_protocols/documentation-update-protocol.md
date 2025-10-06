# Documentation Update Protocol

This protocol provides clear guidelines for instructing AI agents about which documentation files to update and when. Use this protocol to ensure consistent documentation maintenance across the project.

## Document Hierarchy & Update Guidelines

### 1. Primary Documentation (Update Immediately)

**README.md** - Main project documentation
- **When to update:** Major feature additions, breaking changes, new setup requirements
- **What to include:** Installation instructions, quick start guide, feature overview
- **Agent instruction format:** "Update README.md to document [feature/change]"

### 2. Component-Specific Documentation (Update After Implementation)

**docs/** directory files:
- **docs/setup/** - Installation and configuration guides
  - Update when: New dependencies, environment setup changes, configuration options
- **docs/project/** - Architecture and design documents
  - Update when: New components, architectural changes, design decisions
- **docs/ai_handbook/** - AI/ML specific documentation
  - Update when: New models, training procedures, evaluation methods

**ui/README.md** - UI-specific documentation
- **When to update:** New UI features, interface changes, workflow updates
- **Agent instruction format:** "Update ui/README.md to document [UI feature/change]"

### 3. Operational Documentation (Update After Testing)

**docs/submission-quick-reference.md** - Competition submission guide
- **When to update:** Changes to submission process, new evaluation metrics, competition requirements

**docs/generating-submissions.md** - Technical submission generation
- **When to update:** Changes to submission format, new output requirements

### 4. Development Documentation (Update During Development)

**docs/agentic-workflow-example.md** - AI agent workflow examples
- **When to update:** New agent capabilities, workflow improvements, automation examples

**docs/semantic_search_patterns.md** - Search and discovery patterns
- **When to update:** New search features, indexing improvements, query patterns

## Agent Instruction Format

When requesting documentation updates, use this structured format:

```
DOCUMENTATION UPDATE REQUEST

Target Document: [specific file path]
Change Type: [new feature|bug fix|enhancement|breaking change]
Context: [brief description of what changed]
Update Scope: [what sections to update/add]
Validation: [how to verify the documentation is correct]

Description:
[Detailed description of changes and why documentation needs updating]
```

## Examples

### Example 1: New Feature Documentation
```
DOCUMENTATION UPDATE REQUEST

Target Document: ui/README.md
Change Type: new feature
Context: Added checkpoint catalog functionality to inference UI
Update Scope: Inference UI section - add checkpoint catalog features
Validation: Verify checkpoint catalog description matches implementation

Description:
The inference UI now includes automatic checkpoint discovery and cataloging.
Update the Inference UI section to document:
- Automatic checkpoint scanning
- Metadata display features
- Catalog browsing workflow
```

### Example 2: Breaking Change Documentation
```
DOCUMENTATION UPDATE REQUEST

Target Document: README.md
Change Type: breaking change
Context: Updated PyTorch Lightning version requirement
Update Scope: Requirements section, installation instructions
Validation: Ensure version compatibility is clearly stated

Description:
PyTorch Lightning updated to 2.1+ with breaking changes.
Update requirements and installation docs accordingly.
```

### Example 3: Test Documentation
```
DOCUMENTATION UPDATE REQUEST

Target Document: tests/integration/test_checkpoint_fixes.py
Change Type: new test
Context: Added comprehensive checkpoint catalog testing
Update Scope: Test file moved to appropriate directory
Validation: Test runs successfully and documents catalog functionality

Description:
Created comprehensive test suite for checkpoint catalog fixes.
Moved test file to tests/integration/ and updated any references.
```

## Documentation Standards

### Content Requirements
- **Entry Points:** Every major feature must have clear usage examples
- **Prerequisites:** List all requirements before usage instructions
- **Validation:** Include verification steps for each procedure
- **Troubleshooting:** Add common issues and solutions
- **Examples:** Provide concrete code/command examples

### Update Triggers
- ✅ **Immediate:** Breaking changes, security issues, new major features
- ✅ **After Testing:** Bug fixes, enhancements, new minor features
- ✅ **Weekly Review:** Check for outdated information, missing examples
- ✅ **Before Release:** Comprehensive documentation review

### Quality Checks
- [ ] All code examples are tested and working
- [ ] Prerequisites are complete and accurate
- [ ] Cross-references between documents are valid
- [ ] Table of contents matches document structure
- [ ] External links are accessible
- [ ] Screenshots/images are up to date (if applicable)

## Agent Response Protocol

When an agent updates documentation, they should:

1. **Confirm understanding** of the update request
2. **Identify target document** and specific sections to modify
3. **Provide before/after summary** of changes
4. **Validate** the documentation (run examples, check links)
5. **Report completion** with any follow-up recommendations

## Emergency Documentation Updates

For critical issues requiring immediate documentation updates:

```
URGENT DOCUMENTATION UPDATE

Priority: [HIGH|CRITICAL]
Impact: [user safety|data loss|security|functionality]
Deadline: [timeframe]
Stakeholders: [who needs to be notified]

[Standard update request format]
```

This protocol ensures consistent, timely, and high-quality documentation maintenance across the project.
