# **filename: docs/ai_handbook/02_protocols/development/12_bug_fix_protocol.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=bug_fixes,critical_issues,documentation -->

# **Protocol: Bug Fix Documentation**

## **Overview**
This protocol provides comprehensive guidelines for documenting bug fixes, ensuring consistent reporting, proper categorization, and maintainable documentation structure. All bug fixes must follow this protocol to maintain project quality and debugging traceability.

## **Prerequisites**
- Access to project repository and documentation
- Understanding of the bug report template (`docs/bug_reports/BUG_REPORT_TEMPLATE.md`)
- Familiarity with the project's development workflow
- Access to bug reports directory (`docs/bug_reports/`)

## **Procedure**

### **Step 1: Assess Bug Severity**
Determine the appropriate documentation level based on bug impact:

**Critical Bugs** (require full bug report):
- System crashes or data corruption
- Security vulnerabilities
- Complete loss of functionality
- Breaking changes affecting multiple users

**Standard Bugs** (require bug report):
- Functional errors with clear reproduction steps
- Performance degradation
- UI/UX issues affecting usability
- Configuration errors

**Quick Fixes** (use QUICK_FIXES.md):
- Minor patches (< 5 lines changed)
- Hotfixes applied during development
- Cosmetic fixes
- Dependency updates

### **Step 2: Generate Bug ID**
Use the standardized BUG-YYYY-NNN format:
- `BUG-YYYY-NNN` where YYYY is the current year and NNN is sequential number
- Check existing bug reports to determine the next available number
- Example: `BUG-2025-011` for the 11th bug report in 2025

### **Step 3: Create Bug Report**
Create a new bug report file following the template:

**Location**: `docs/bug_reports/BUG-YYYY-NNN_descriptive_name.md`
**Template**: Use `docs/bug_reports/BUG_REPORT_TEMPLATE.md`

**Required Sections**:
- Bug ID, Date, Reporter, Severity, Status
- Summary, Environment, Steps to Reproduce
- Expected vs Actual Behavior
- Root Cause Analysis, Resolution
- Testing, Prevention, Files Changed
- Impact Assessment

### **Step 4: Update Related Documentation**
Update all relevant documentation files:

**Changelog Entry** (`docs/CHANGELOG.md`):
```markdown
#### Bug Fixes
- **BUG-2025-011**: Fixed inference UI coordinate transformation bug causing annotation misalignment for EXIF-oriented images ([BUG-2025-011_inference_ui_coordinate_transformation.md](bug_reports/BUG-2025-011_inference_ui_coordinate_transformation.md))
```

**Quick Fixes Log** (`docs/QUICK_FIXES.md`) - if applicable:
```markdown
## 2025-10-19 10:30 BUG - Inference UI coordinate transformation

**Issue**: OCR annotations misaligned for EXIF-oriented images
**Fix**: Removed incorrect inverse transformations in InferenceEngine
**Files**: ui/utils/inference/engine.py
**Impact**: minimal
**Test**: ui
```

### **Step 5: Verify Documentation**
Ensure all documentation is consistent and properly linked:

- [ ] Bug report follows template format
- [ ] Bug ID is unique and properly formatted
- [ ] File location follows naming convention
- [ ] Changelog references bug report correctly
- [ ] All file paths are accurate
- [ ] Cross-references are working

## **File Organization**

### **Bug Reports Directory Structure**
```
docs/bug_reports/
├── BUG_REPORT_TEMPLATE.md          # Template for new reports
├── BUG-2025-001_*.md              # Year-based sequential numbering
├── BUG-2025-002_*.md
└── BUG-2025-011_*.md              # Current highest number
```

### **Naming Convention**
- **Format**: `BUG-YYYY-NNN_descriptive_name.md`
- **Year**: Current year (YYYY)
- **Number**: Sequential within year (001, 002, etc.)
- **Description**: Brief, descriptive name using underscores
- **Examples**:
  - `BUG-2025-011_inference_ui_coordinate_transformation.md`
  - `BUG-2025-012_pydantic_validation_error.md`

### **Cross-Reference Requirements**
- Changelog entries must reference bug reports
- Bug reports should reference related issues
- Quick fixes should reference bug reports when applicable

## **Examples**

### **Critical Bug Report Structure**
```markdown
## 🐛 Bug Report Template

**Bug ID:** BUG-2025-011
**Date:** October 19, 2025
**Reporter:** Development Team
**Severity:** Critical
**Status:** Fixed

### Summary
[Brief description of the bug and fix]

### Environment
[Technical context and affected components]

### Steps to Reproduce
[Clear reproduction steps]

### Root Cause Analysis
[Technical analysis of the problem]

### Resolution
[How the bug was fixed]

### Testing
[Validation performed]

### Files Changed
[Affected files list]
```

### **Changelog Integration**
```markdown
#### Bug Fixes
- **BUG-2025-011**: Fixed inference UI coordinate transformation bug causing annotation misalignment for EXIF-oriented images ([BUG-2025-011_inference_ui_coordinate_transformation.md](bug_reports/BUG-2025-011_inference_ui_coordinate_transformation.md))
```

## **Quality Assurance**
- **Template Compliance**: All bug reports must use the standard template
- **ID Uniqueness**: Bug IDs must be unique across the project
- **Location Consistency**: Bug reports must be in `docs/bug_reports/` directory
- **Cross-References**: All related documentation must be properly linked
- **Content Completeness**: All required sections must be filled out

## **Automation Support**
Use available tools for bug fix documentation:

```bash
# Validate bug report format
python scripts/agent_tools/validate_bug_report.py --file docs/bug_reports/BUG-2025-011_inference_ui_coordinate_transformation.md

# Generate changelog entry
python scripts/agent_tools/generate_changelog_entry.py --bug-id BUG-2025-011
```

## **Common Pitfalls**
- **Wrong Location**: Bug reports should not go in changelog directories
- **Inconsistent Naming**: Always use BUG-YYYY-NNN format
- **Missing Cross-References**: Ensure changelog and quick fixes reference bug reports
- **Incomplete Documentation**: Fill out all template sections
- **Duplicate IDs**: Check existing reports before assigning new IDs</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/ai_handbook/02_protocols/development/12_bug_fix_protocol.md
