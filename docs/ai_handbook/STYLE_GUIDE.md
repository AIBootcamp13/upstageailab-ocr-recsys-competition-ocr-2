# Documentation Style Guide

## Purpose

This style guide defines the standards and conventions for all documentation in the AI Agent Handbook. All contributors must follow these guidelines to ensure consistency, clarity, and maintainability across all documentation files.

## Document Structure

### Required Sections

All documentation files must contain the following sections in order:

1. **Title Header**: Must follow the format `# # **filename: path/to/file.md**`
2. **AI Cue Markers**: Must include both priority and use_when markers
3. **Main Content**: Varies by document type (see templates)
4. **Footer**: Standard footer with update information

### Document Types and Required Sections

#### Base Protocol Documents
- Overview
- Prerequisites
- Procedure
- Validation
- Troubleshooting
- Related Documents

#### Development Protocol Documents
- Overview
- Prerequisites
- Procedure
- Validation
- Troubleshooting
- Related Documents

#### Configuration Protocol Documents
- Overview
- Prerequisites
- Procedure
- Configuration Structure
- Validation
- Troubleshooting
- Related Documents

#### Governance Protocol Documents
- Overview
- Prerequisites
- Governance Rules
- Procedure
- Compliance Validation
- Enforcement
- Troubleshooting
- Related Documents

#### Component Protocol Documents
- Overview
- Prerequisites
- Component Architecture
- Procedure
- API Reference
- Validation
- Troubleshooting
- Related Documents

#### Reference Documents
- Overview
- Key Concepts
- Detailed Information
- Examples
- Configuration Options
- Best Practices
- Troubleshooting
- Related References

## Markdown Formatting Standards

### Headers
- Use level 2 headers (`##`) for all main sections: `## **Section Title**`
- Use level 3 headers (`###`) for subsections: `### Subsection Title`
- Use level 4 headers (`####`) for sub-subsections: `#### Sub-subsection Title`
- All main section headers must be bold: `## **Bold Section Title**`
- Section titles must use Title Case (capitalize major words)

### Lists
- Use hyphens (`-`) for unordered lists
- Use numbers (`1.`, `2.`, etc.) for ordered lists
- Maintain consistent indentation (2 spaces per level)
- Include a blank line before and after each list

### Code Blocks
- Use triple backticks (```) for code blocks
- Specify language for syntax highlighting: ```python, ```bash, etc.
- Indent code properly within blocks
- Use inline code with single backticks: `code`

### Links and References
- Use relative paths for internal links: `[Link Text](./path/to/file.md)`
- Use absolute URLs for external links
- Cross-reference other documents with full path: `[Document Name](./02_protocols/development/example.md)`

## Content Standards

### Terminology Consistency
- Use project-specific terminology consistently:
  - "AI Agent Handbook" (not "Documentation" or "Manual")
  - "Command Registry" (capitalized when referring to the specific system)
  - "Hydra" (for the configuration framework)
  - "PyTorch Lightning" (full name, not "PTL")
- Define acronyms on first use in each document
- Use active voice wherever possible
- Write in present tense when describing functionality

### Cross-Reference Validation
- All internal links must point to existing files
- Use descriptive link text that indicates the destination
- Verify cross-references during content updates
- Include both forward and backward references where appropriate

### Example Code Standards
- Include comments explaining complex code examples
- Use realistic, project-relevant examples
- Follow the same coding standards as the project codebase
- Test code examples when possible

## AI Cue Markers

All documentation files must include AI cue markers in the following format:

```markdown
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=when_working_on_project -->
```

Valid priority values:
- `critical` (highest priority)
- `high`
- `medium`
- `low`

Valid use_when values should describe when an AI agent should prioritize this document.

## Validation Requirements

### Template Validation
- All documents must follow the appropriate template structure
- Required sections must be present and non-empty
- Template placeholders must be replaced with actual content

### Content Validation
- All links must be valid and point to existing resources
- All referenced files and commands must exist
- Content must be accurate and up-to-date
- No template placeholders (`{{variable}}`) should remain in final documents

## File Naming Conventions

### Numbering System
- Use sequential numbering for documents within directories
- Maintain consistent numbering when adding new documents
- Update cross-references when document numbers change

### Naming Format
- Use lowercase letters, numbers, hyphens, and underscores
- Separate words with hyphens: `example-document-name.md`
- Include descriptive names that indicate content purpose
- Use the format: `NN_document_title.md` where NN is the sequence number

## Quality Standards

### Completeness
- All required sections must contain meaningful content
- Procedures must be complete and actionable
- Troubleshooting sections should address common issues
- Validation steps must be verifiable

### Clarity
- Use clear, concise language
- Avoid jargon without explanation
- Provide context for complex topics
- Include visual aids when helpful

### Consistency
- Maintain consistent formatting throughout the handbook
- Use identical terminology across all documents
- Follow the same structural patterns
- Apply the same style conventions universally

## Maintenance

### Update Procedures
- Update the "Last Updated" timestamp when making changes
- Maintain version history in changelog documents
- Review cross-references when moving or renaming files
- Validate all changes using the standardization script

### Review Process
- All documentation changes require validation
- Cross-reference accuracy must be verified
- Template compliance must be confirmed
- Style guide adherence should be checked
