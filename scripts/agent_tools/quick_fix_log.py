#!/usr/bin/env python3
"""Quick fix logging utility for minimal bug fix documentation."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def log_quick_fix(fix_type: str, title: str, issue: str, fix: str, files: str, impact: str = "minimal", test: str = "none") -> None:
    """Log a quick fix to the QUICK_FIXES.md file."""

    # Format the entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"""## {timestamp} {fix_type.upper()} - {title}

**Issue**: {issue}
**Fix**: {fix}
**Files**: {files}
**Impact**: {impact}
**Test**: {test}
"""

    # Read existing content
    quick_fixes_file = Path("docs/QUICK_FIXES.md")
    if quick_fixes_file.exists():
        content = quick_fixes_file.read_text()
        # Find the separator line and insert before it
        lines = content.split("\n")
        sep_index = -1
        for i, line in enumerate(lines):
            if line.strip() == "---":
                sep_index = i
                break

        if sep_index >= 0:
            # Insert after the separator
            lines.insert(sep_index + 1, entry)
            new_content = "\n".join(lines)
        else:
            # Append to end
            new_content = content + "\n\n" + entry
    else:
        # Create new file
        new_content = f"""# Quick Fixes Log

This log tracks quick fixes, patches, and hotfixes applied to the codebase.
Format follows the [Quick Fixes Protocol](docs/ai_handbook/02_protocols/development/11_quick_fixes_protocol.md).

---

{entry}"""

    # Write back
    quick_fixes_file.write_text(new_content)
    print(f"✅ Logged quick fix: {title}")
    print(f"📝 Entry added to {quick_fixes_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Log quick fixes to QUICK_FIXES.md")
    parser.add_argument("type", choices=["bug", "compat", "config", "dep", "doc", "perf", "sec", "ui"], help="Type of fix")
    parser.add_argument("title", help="Brief title for the fix")
    parser.add_argument("--issue", required=True, help="One-line problem description")
    parser.add_argument("--fix", required=True, help="One-line solution description")
    parser.add_argument("--files", required=True, help="Comma-separated list of affected files")
    parser.add_argument("--impact", choices=["minimal", "major", "none"], default="minimal", help="Impact level (default: minimal)")
    parser.add_argument(
        "--test", choices=["unit", "ui", "manual", "integration", "none"], default="none", help="Testing performed (default: none)"
    )

    args = parser.parse_args()

    log_quick_fix(
        fix_type=args.type, title=args.title, issue=args.issue, fix=args.fix, files=args.files, impact=args.impact, test=args.test
    )


if __name__ == "__main__":
    main()
