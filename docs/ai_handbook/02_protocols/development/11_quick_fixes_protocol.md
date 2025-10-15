# **filename: docs/ai_handbook/02_protocols/development/11_quick_fixes_protocol.md**
<!-- ai_cue:priority=medium -->
<!-- ai_cue:use_when=bug_fixes,hotfixes,patches -->

# **Protocol: Quick Fixes Logging**

## **Overview**
This protocol provides a minimal, structured format for recording quick fixes, patches, and hotfixes. Designed for rapid logging of numerous small changes without verbose documentation overhead.

## **Prerequisites**
- Access to project repository and documentation
- Understanding of basic Markdown formatting
- Familiarity with the project's development workflow
- Access to the QUICK_FIXES.md log file

## **Procedure**

### **Step 1: Prepare Fix Information**
Gather the required information for the fix entry:
- Fix type (bug, compat, config, dep, doc, perf, sec, ui)
- Brief, descriptive title
- One-line problem description
- One-line solution description
- Comma-separated list of affected files
- Impact level (minimal/major/none)
- Testing performed (unit/ui/manual/integration/none)

### **Step 2: Log the Fix**
Use one of the following methods to record the fix:

**Via Makefile (recommended):**
```bash
make quick-fix-log TYPE=<type> TITLE="<title>" ISSUE="<issue>" FIX="<fix>" FILES="<files>" [IMPACT=<impact>] [TEST=<test>]
```

**Via Python script:**
```bash
python scripts/agent_tools/quick_fix_log.py <type> "<title>" --issue "<issue>" --fix "<fix>" --files "<files>" [--impact <impact>] [--test <test>]
```

**Manual entry in docs/QUICK_FIXES.md:**
```markdown
## YYYY-MM-DD HH:MM TYPE - Brief Title

**Issue**: One-line problem description
**Fix**: One-line solution description
**Files**: file1.py, file2.py
**Impact**: minimal/major/none
**Test**: unit/ui/manual/none
```

### **Step 3: Verify Entry**
Confirm the entry was added correctly to docs/QUICK_FIXES.md with proper formatting and complete information.

## **Entry Format**

### **Template**
```markdown
## YYYY-MM-DD HH:MM [TYPE] - Brief Title

**Issue**: One-line problem description
**Fix**: One-line solution description
**Files**: file1.py, file2.py
**Impact**: minimal/major/none
**Test**: unit/ui/manual/none
```

### **Types**
- `BUG` - Bug fix
- `COMPAT` - Compatibility fix
- `CONFIG` - Configuration fix
- `DEP` - Dependency fix
- `DOC` - Documentation fix
- `PERF` - Performance fix
- `SEC` - Security fix
- `UI` - User interface fix

### **Impact Levels**
- `minimal` - No breaking changes, safe rollback
- `major` - Breaking changes or significant behavior change
- `none` - Cosmetic or no functional impact

### **Test Types**
- `unit` - Unit tests pass
- `ui` - UI tests pass
- `manual` - Manual verification completed
- `integration` - Integration tests pass
- `none` - No testing required

## **Examples**

### **Bug Fix Example**
```markdown
## 2025-10-15 14:30 BUG - Pydantic replace() compatibility

**Issue**: TypeError when building preprocessing config in Streamlit UI
**Fix**: Replace dataclass replace() with Pydantic model_copy(update=)
**Files**: ui/apps/inference/state.py
**Impact**: minimal
**Test**: ui
```

### **Configuration Fix Example**
```markdown
## 2025-10-15 09:15 CONFIG - FP16 gradient scaling

**Issue**: Mixed precision training causing NaN losses
**Fix**: Added conservative gradient clipping to fp16_safe.yaml
**Files**: configs/train/fp16_safe.yaml
**Impact**: minimal
**Test**: unit
```

### **Dependency Fix Example**
```markdown
## 2025-10-14 16:45 DEP - PyTorch CUDA version mismatch

**Issue**: CUDA 12.1 runtime incompatible with PyTorch 2.1.0
**Fix**: Pinned torch==2.1.2 with CUDA 11.8 support
**Files**: pyproject.toml
**Impact**: major
**Test**: integration
```

## **Validation**

### **Entry Completeness**
- All required fields present (TYPE, Title, Issue, Fix, Files, Impact, Test)
- Timestamps in correct format (YYYY-MM-DD HH:MM)
- File paths are accurate and exist
- Impact and test values are from allowed options

### **Format Compliance**
- Proper Markdown formatting with consistent indentation
- Bold field labels (`**Field**:`)
- Chronological order (newest entries first)
- No template placeholders remaining

### **Content Accuracy**
- Problem description accurately reflects the issue
- Solution description correctly describes the fix
- File list includes all affected files
- Impact level correctly assessed

## **Troubleshooting**

### **Common Issues**
- **Script fails with permission error**: Ensure write access to docs/QUICK_FIXES.md
- **Invalid type/impact/test values**: Use only allowed values from the specification
- **Missing required fields**: All fields except IMPACT and TEST are mandatory
- **File not found**: Verify docs/QUICK_FIXES.md exists and is accessible

### **Recovery Procedures**
- **Corrupted entry**: Edit manually in docs/QUICK_FIXES.md
- **Wrong format**: Use the template and re-enter the information
- **Missing file**: Create docs/QUICK_FIXES.md with the header template
- **Script issues**: Fall back to manual entry method

## **Guidelines**

### **When to Log**
- Any code change that fixes an immediate issue
- Configuration adjustments that resolve runtime errors
- Dependency updates that fix compatibility problems
- Documentation corrections that fix misleading information

### **When NOT to Log**
- Planned feature implementations (use CHANGELOG.md)
- Refactoring without bug fixes
- Code style or formatting changes
- Test additions without fixes

### **Best Practices**
- **Keep it brief**: Maximum 5 lines per entry
- **Be specific**: Include exact error messages or symptoms
- **Reference commits**: Add commit hash if available
- **Update immediately**: Log fixes as they're applied
- **Review weekly**: Archive old entries (>30 days) to summary

## **Maintenance**

### **Weekly Review**
```bash
# Move entries older than 30 days to archive
# Create summary statistics
# Identify recurring issue patterns
```

### **Monthly Archive**
```bash
# Move entries to docs/_archive/quick_fixes/YYYY-MM.md
# Update summary statistics
# Clean up main log file
```

### **Tools**
```bash
# Quick fix counter
grep "^## " docs/QUICK_FIXES.md | wc -l

# Recent fixes by type
grep "^## " docs/QUICK_FIXES.md | grep "BUG" | head -5

# Impact analysis
grep "Impact:" docs/QUICK_FIXES.md | sort | uniq -c
```

## **Related Documents**
- [CHANGELOG.md](../../CHANGELOG.md) - Major feature releases and planned changes
- [Context Logging Protocol](./06_context_logging.md) - Detailed agent action logging
- [Debugging Workflow](./03_debugging_workflow.md) - Systematic debugging procedures
- [QUICK_FIXES.md](../../QUICK_FIXES.md) - The actual quick fixes log

---

**Last Updated**: 2025-10-15
**Version**: 1.0
**Authors**: AI Assistant
**Review Status**: Ready for use</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/ai_handbook/02_protocols/development/11_quick_fixes_protocol.md
