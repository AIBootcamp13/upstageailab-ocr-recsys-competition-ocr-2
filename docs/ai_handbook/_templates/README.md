# AI Handbook Protocol Templates

This directory contains standardized templates for creating consistent documentation in the AI Handbook.

## Available Templates

### Base Template (`base.md`)
The foundation template that all other templates extend. Contains the core structure:
- Overview
- Prerequisites
- Procedure
- Validation
- Troubleshooting
- Related Documents

### Specialized Templates

#### Development (`development.md`)
For development-related protocols (code changes, refactoring, debugging)
- **Priority**: High
- **Use When**: code_changes, refactoring, debugging

#### Configuration (`configuration.md`)
For configuration and setup protocols
- **Priority**: Medium
- **Use When**: configuration, setup, deployment

#### Governance (`governance.md`)
For governance, compliance, and standards protocols
- **Priority**: High
- **Use When**: governance, compliance, standards

#### Components (`components.md`)
For component development and integration protocols
- **Priority**: Medium
- **Use When**: component_development, integration, architecture

#### References (`references.md`)
For reference documentation and lookup guides
- **Priority**: Low
- **Use When**: reference, lookup, documentation

## How to Use Templates

### 1. Choose the Appropriate Template
Select the template that best matches your documentation type:

- **New protocol**: Use the most specific template (development, configuration, etc.)
- **Reference material**: Use the references template
- **General documentation**: Use the base template

### 2. Copy and Customize
1. Copy the chosen template file
2. Replace all `{{placeholder}}` values with actual content
3. Update the filename header to match the actual file path
4. Fill in AI cue values appropriately

### 3. Template Placeholders
Replace these placeholders with actual content:
- `{{protocol_title}}` - Clear, descriptive title
- `{{protocol_description}}` - Brief description of the protocol's purpose
- `{{overview_content}}` - Detailed explanation of what the protocol covers
- `{{prerequisites_content}}` - Requirements and dependencies
- `{{procedure_content}}` - Step-by-step instructions
- `{{validation_content}}` - How to verify the protocol worked
- `{{troubleshooting_content}}` - Common issues and solutions
- `{{related_documents_content}}` - Links to related documentation

### 4. AI Cues
Set appropriate AI cue values:
- **priority**: `high`, `medium`, `low`
- **use_when**: Comma-separated list of contexts (e.g., `code_changes,debugging`)

### 5. Validation
Before committing, run the template validator:

```bash
python scripts/validate_templates.py docs/ai_handbook/_templates docs/ai_handbook
```

## Template Structure Standards

### Required Sections
All templates must include these sections:
- Overview
- Prerequisites
- Procedure
- Validation
- Troubleshooting
- Related Documents

### Formatting Standards
- Use Markdown headers (`#`, `##`, `###`)
- Include code blocks with appropriate language tags
- Use tables for configuration options
- Include cross-references to related documents

### AI Optimization
- Include AI cue comments at the top
- Use descriptive section headers
- Provide clear, actionable content
- Include examples and code snippets

## Creating New Templates

When creating new specialized templates:

1. Start from the base template
2. Add template-specific sections and placeholders
3. Update the validation script to include the new template type
4. Document the new template in this README
5. Test the template with sample content

## Validation Rules

The template validator checks for:
- Required AI cues (`priority`, `use_when`)
- Required sections based on template type
- Unresolved template placeholders
- Correct filename headers
- Proper markdown formatting

## Examples

See existing protocols in `../02_protocols/` for examples of templates in use.
