# Post-Debugging Session Framework

**Version:** 1.1
**Last Updated:** 2025-10-08
**Purpose:** Standardized workflow for organizing, documenting, and integrating results from successful debugging sessions

## Table of Contents

- [Overview](#overview)
- [Session Completion Checklist](#session-completion-checklist)
- [File Organization Workflow](#file-organization-workflow)
- [Workflow Phases](#workflow-phases)
- [Automation Scripts](#automation-scripts)
- [Success Criteria](#success-criteria)
- [Usage Instructions](#usage-instructions)
- [Framework Maintenance](#framework-maintenance)

## Overview

This framework provides a systematic approach to post-debugging session work, ensuring that:
- Loose documents and scripts are properly organized
- Session results are thoroughly documented
- Project documentation stays current
- Future sessions can easily reference past work

**Key Benefits:**
- Eliminates manual record-keeping inconsistencies
- Provides precise instructions for agents
- Ensures comprehensive knowledge transfer
- Maintains project documentation accuracy
- Enables faster future debugging sessions

## Session Completion Checklist

### Immediate Post-Session (Within 1 Hour)
- [ ] Run artifact organization script
- [ ] Rename files to follow naming conventions
- [ ] Update session status in README.md
- [ ] Create or update resolution summary

### Short-term Documentation (Within 24 Hours)
- [ ] Document all findings and insights gained
- [ ] Update relevant project documentation
- [ ] Create session handover document
- [ ] Archive or remove temporary files

### Integration (Within 1 Week)
- [ ] Update project README if major changes
- [ ] Add references to new documentation
- [ ] Review and update related docs
- [ ] Share key insights with team

## File Organization Workflow

### Step 1: Identify Loose Files
```bash
# Find all debugging-related files in workspace
find . -name "*debug*" -o -name "*test*" -o -name "*profile*" -o -name "*cache*" | grep -v __pycache__ | grep -v .git
```

### Step 2: Categorize Files
**Documentation Files:**
- Analysis reports, findings, recommendations
- Move to: `docs/ai_handbook/04_experiments/YYYY-MM-DD_experiment/`

**Scripts:**
- Investigation scripts, profiling tools
- Move to: `docs/ai_handbook/04_experiments/YYYY-MM-DD_experiment/scripts/`

**Artifacts:**
- Cache dumps, performance profiles, test outputs
- Move to: `docs/ai_handbook/04_experiments/YYYY-MM-DD_experiment/artifacts/`

**Logs:**
- Session logs, command outputs
- Move to: `docs/ai_handbook/04_experiments/YYYY-MM-DD_experiment/logs/`

### Step 3: Rename Files (Standard Convention)
```bash
# Automated renaming script
rename_files_by_convention() {
    local session_dir=$1

    # Process each file type
    for file in "$session_dir"/*; do
        if [[ -f "$file" ]]; then
            filename=$(basename "$file")
            extension="${filename##*.}"
            base_name="${filename%.*}"

            # Skip if already properly named
            if [[ $filename =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_ ]]; then
                continue
            fi

            # Add date prefix and sequential number
            date_prefix=$(date +"%Y-%m-%d")
            sequential_num=$(get_next_sequential_num "$session_dir" "$extension")

            new_name="${date_prefix}_${sequential_num}_${base_name}.${extension}"
            mv "$file" "$(dirname "$file")/$new_name"
        fi
    done
}
```

**Naming Convention:**
- `YYYY-MM-DD_NN_descriptive_name.extension`
- `NN`: Sequential number (01, 02, 03...)
- `descriptive_name`: Clear, specific, no abbreviations

## Workflow Phases

### Phase 1: File Organization & Naming Convention Enforcement

#### 1.1 Identify Loose Files
**Search Pattern:** Look for files in workspace root, debug folders, and temporary directories
```bash
# Find potential loose files
find . -name "*.py" -o -name "*.md" -o -name "*.log" -o -name "*.json" | grep -v -E "(docs/|__pycache__|node_modules|\.git)"
```

**Common Locations to Check:**
- Workspace root directory
- `debug/` or `tmp/` folders
- `logs/` directory
- Desktop or Downloads (if applicable)

#### 1.2 Apply Naming Convention
**Standard Format:** `YYYY-MM-DD_index_descriptive_name.ext`

**Examples:**
- `2025-10-08_01_debug_cache_keys.py`
- `2025-10-08_02_performance_test_results.json`
- `2025-10-08_03_memory_analysis.log`

**Index Assignment Rules:**
- `00`: Session summary/overview
- `01-09`: Primary artifacts (scripts, configs)
- `10-49`: Secondary artifacts (logs, data dumps)
- `50-79`: Analysis and documentation
- `80-99`: Archives and deprecated files

#### 1.3 Create Organized Folder Structure
```
docs/ai_handbook/04_experiments/YYYY-MM-DD_debug_session_name/
├── 00_session_summary.md
├── 01_debug_findings.md
├── 02_root_cause_analysis.md
├── 03_solution_implementation.md
├── 04_test_results.md
├── 05_performance_impact.md
├── artifacts/
│   ├── logs/
│   │   ├── YYYY-MM-DD_HH-MM-SS_debug_run.log
│   │   └── YYYY-MM-DD_HH-MM-SS_performance_test.log
│   ├── data/
│   │   ├── cache_inspection_data.json
│   │   └── memory_snapshots/
│   └── configs/
│       └── debug_config_overrides.yaml
└── scripts/
    ├── 01_reproduce_issue.py
    ├── 02_validate_fix.py
    └── 03_performance_test.py
```

#### 1.4 File Movement Operations
```bash
# Create session directory
SESSION_DIR="docs/ai_handbook/04_experiments/$(date +%Y-%m-%d)_${SESSION_NAME}"
mkdir -p "$SESSION_DIR/{artifacts/{logs,data,configs},scripts}"

# Move and rename files
mv debug_cache_keys.py "$SESSION_DIR/scripts/01_debug_cache_keys.py"
mv performance_test.log "$SESSION_DIR/artifacts/logs/$(date +%Y-%m-%d_%H-%M-%S)_performance_test.log"
mv cache_data.json "$SESSION_DIR/artifacts/data/cache_inspection_data.json"
```

### Phase 2: Session Documentation

#### 2.1 Create Session Summary (00_session_summary.md)
**Required Sections:**
```markdown
# Debug Session Summary: [Issue Title]

**Date:** YYYY-MM-DD
**Duration:** X hours
**Status:** ✅ Resolved
**Priority:** [High/Medium/Low]

## Problem Statement
[Clear description of the issue]

## Root Cause
[Technical explanation of what was wrong]

## Solution
[How the issue was fixed]

## Impact
- Performance improvement: [metrics]
- Code changes: [files modified]
- Test coverage: [new tests added]

## Files Changed
- `path/to/file.py`: [description of changes]
- `path/to/config.yaml`: [configuration updates]

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance tests show improvement
- [ ] Manual validation complete

## Next Steps
[Future improvements or related issues]
```

#### 2.2 Document Technical Insights (01_debug_findings.md)
**Content Requirements:**
- Step-by-step debugging process
- Tools and techniques used
- Key observations and hypotheses tested
- Code snippets showing the issue and fix
- Performance metrics before/after

#### 2.3 Root Cause Analysis (02_root_cause_analysis.md)
**Analysis Framework:**
```markdown
## Root Cause Analysis

### What Was Expected
[Expected behavior and assumptions]

### What Actually Happened
[Observed behavior with evidence]

### Why It Happened
[Technical explanation with code references]

### Contributing Factors
- [Factor 1]: [Impact and evidence]
- [Factor 2]: [Impact and evidence]

### Prevention Measures
[How to avoid similar issues in future]
```

#### 2.4 Solution Documentation (03_solution_implementation.md)
**Implementation Details:**
- Code changes with before/after diffs
- Configuration updates
- Test additions/modifications
- Performance validation results

### Phase 3: Project Documentation Updates

#### 3.1 Update Changelog
**Location:** `docs/ai_handbook/05_changelog/YYYY-MM/`

**Entry Format:**
```markdown
## YYYY-MM-DD - [Session Name]

### Changes
- **Fixed:** [Issue description] ([commit hash])
  - Root cause: [brief explanation]
  - Impact: [performance/code metrics]
  - Files: `path/to/changed/file.py`

### Technical Details
- [Key technical insights]
- [Performance improvements]
- [Code quality improvements]

### Testing
- Added [test coverage]
- Validated [performance metrics]
```

#### 3.2 Update Relevant Documentation
**Check and Update:**
- **README.md:** If user-facing changes
- **API Documentation:** If public interfaces changed
- **Configuration Docs:** If config options added/modified
- **Troubleshooting Guide:** Add known issues and solutions
- **Performance Guide:** Update performance characteristics

#### 3.3 Update Code Comments and Docstrings
**Requirements:**
- Add explanatory comments for complex fixes
- Update docstrings to reflect new behavior
- Document configuration parameters
- Add type hints where applicable

### Phase 4: Quality Assurance & Validation

#### 4.1 Documentation Completeness Check
- [ ] All files properly named and organized
- [ ] Session summary covers all required sections
- [ ] Technical insights clearly documented
- [ ] Code changes explained with context
- [ ] Performance impact quantified

#### 4.2 Cross-Reference Validation
- [ ] Changelog entries match session documentation
- [ ] Code comments reference relevant docs
- [ ] Test files include documentation links
- [ ] Configuration changes documented

#### 4.3 Knowledge Transfer Verification
- [ ] Another developer could understand the issue from docs
- [ ] Solution approach is clearly explained
- [ ] Future maintainers can avoid similar issues

## Automation Scripts

### File Organization Script
```python
#!/usr/bin/env python3
"""
Post-debug session file organizer.
Automatically finds, renames, and organizes loose files.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

def organize_debug_session(session_name: str, base_date: str = None):
    """Organize loose files into proper session structure."""
    if base_date is None:
        base_date = datetime.now().strftime("%Y-%m-%d")

    # Create session directory
    session_dir = Path(f"docs/ai_handbook/04_experiments/{base_date}_{session_name}")
    session_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    for subdir in ["artifacts/logs", "artifacts/data", "scripts"]:
        (session_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Find and organize loose files
    loose_files = find_loose_files()
    organized_files = organize_files(loose_files, session_dir, base_date)

    return organized_files

def find_loose_files():
    """Find files that should be organized."""
    # Implementation to find loose .py, .md, .log, .json files
    pass

def organize_files(files, session_dir, base_date):
    """Move and rename files according to convention."""
    # Implementation to organize files
    pass
```

### Documentation Generator Script
```python
#!/usr/bin/env python3
"""
Generate post-debug documentation template.
"""

def generate_session_summary(session_data):
    """Generate session summary markdown."""
    template = f"""
# Debug Session Summary: {session_data['title']}

**Date:** {session_data['date']}
**Duration:** {session_data['duration']}
**Status:** ✅ Resolved

## Problem Statement
{session_data['problem']}

## Root Cause
{session_data['root_cause']}

## Solution
{session_data['solution']}

## Impact
{session_data['impact']}
"""
    return template
```

## Success Criteria

### Organization Quality
- [ ] All loose files properly renamed and organized
- [ ] Folder structure follows established conventions
- [ ] File naming is consistent and descriptive

### Documentation Quality
- [ ] Session summary is comprehensive and clear
- [ ] Technical insights are well-documented
- [ ] Code changes are explained with context
- [ ] Performance impact is quantified

### Knowledge Transfer
- [ ] Documentation enables future debugging of similar issues
- [ ] Solution approach is reproducible
- [ ] Prevention measures are documented

### Maintenance Readiness
- [ ] Code includes appropriate comments and documentation
- [ ] Tests cover the fixed functionality
- [ ] Configuration changes are documented

## Usage Instructions

1. **After successful debug:** Run this framework immediately
2. **File organization first:** Organize before documenting
3. **Document thoroughly:** Include all technical details
4. **Update project docs:** Ensure knowledge is preserved
5. **Validate completeness:** Use checklists to verify quality

## Framework Maintenance

- **Review quarterly:** Update based on lessons learned
- **Version control:** Track framework improvements
- **Template updates:** Refine based on usage feedback
- **Automation:** Add scripts for common operations

---

**Framework Author:** AI Assistant
**Version:** 1.1
**Last Updated:** 2025-10-08
**Review Date:** 2026-01-08

**Purpose:** This framework addresses the core problem of inconsistent manual record-keeping in technical debugging sessions. By providing precise instructions and automation scripts, it ensures that complex technical changes are properly documented, organized, and integrated into the project knowledge base.</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/ai_handbook/post_debugging_session_framework.md
