#!/usr/bin/env python3
"""
Content Standardization Script

Automates content standardization for the AI Agent Handbook.
Validates documents against style guide requirements and applies standardization rules.
"""

import argparse
import os
import re
import sys
from pathlib import Path


class ContentStandardizer:
    """Standardizes documentation files according to style guide requirements."""

    def __init__(self, docs_dir: str, dry_run: bool = False):
        docs_path = Path(docs_dir)

        # Check if the path is a single file
        if docs_path.is_file():
            self.is_single_file = True
            self.target_file = docs_path
            # Set docs_dir to the parent directory containing the ai_handbook
            self.docs_dir = docs_path.parent
            # Find the base docs directory by looking for the ai_handbook directory
            current = self.docs_dir
            while current.name != "ai_handbook" and current.parent != current:
                current = current.parent
            if current.name == "ai_handbook":
                self.base_docs_dir = current
            else:
                # If we can't find ai_handbook, use the project root as best guess
                self.base_docs_dir = Path("docs/ai_handbook")
        else:
            self.is_single_file = False
            self.base_docs_dir = docs_path
            self.docs_dir = docs_path  # For backward compatibility
            self.target_file = None

        self.dry_run = dry_run
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.fixes_applied: list[str] = []
        self.total_files: int = 0

        # Required sections by document type
        self.required_sections = {
            "base": ["Overview", "Prerequisites", "Procedure", "Validation", "Troubleshooting", "Related Documents"],
            "development": ["Overview", "Prerequisites", "Procedure", "Validation", "Troubleshooting", "Related Documents"],
            "configuration": [
                "Overview",
                "Prerequisites",
                "Procedure",
                "Configuration Structure",
                "Validation",
                "Troubleshooting",
                "Related Documents",
            ],
            "governance": [
                "Overview",
                "Prerequisites",
                "Governance Rules",
                "Procedure",
                "Compliance Validation",
                "Enforcement",
                "Troubleshooting",
                "Related Documents",
            ],
            "components": [
                "Overview",
                "Prerequisites",
                "Component Architecture",
                "Procedure",
                "API Reference",
                "Validation",
                "Troubleshooting",
                "Related Documents",
            ],
            "references": [
                "Overview",
                "Key Concepts",
                "Detailed Information",
                "Examples",
                "Configuration Options",
                "Best Practices",
                "Troubleshooting",
                "Related References",
            ],
        }

        self.required_ai_cues = ["priority", "use_when"]

    def extract_sections(self, content: str) -> list[str]:
        """Extract section headers from markdown content."""
        # Find all level 2 headers (## Header)
        headers = re.findall(r"^##\s+(.+)$", content, re.MULTILINE)
        # Strip ** markers from section names
        cleaned_headers = []
        for h in headers:
            # Remove ** markers if present
            h = h.strip()
            h = re.sub(r"^\*\*(.+)\*$", r"\1", h)  # Remove trailing **
            h = re.sub(r"^\*\*(.+)\*\*$", r"\1", h)  # Remove both **
            cleaned_headers.append(h)
        return cleaned_headers

    def extract_ai_cues(self, content: str) -> dict[str, str]:
        """Extract AI cue comments from content."""
        cues = {}
        cue_pattern = r"<!-- ai_cue:(\w+)=([^>]+?) -->"
        matches = re.findall(cue_pattern, content)
        for key, value in matches:
            cues[key] = value.strip()
        return cues

    def validate_filename_header(self, content: str, file_path: Path) -> bool:
        """Validate the filename header format."""
        # Determine the correct path relative to the ai_handbook directory
        if hasattr(self, "base_docs_dir") and self.base_docs_dir.name == "ai_handbook":
            # Calculate relative path from ai_handbook directory
            ai_handbook_path = self.base_docs_dir
            relative_path = file_path.relative_to(
                ai_handbook_path.parent if ai_handbook_path.parent.name != "docs" else ai_handbook_path.parent.parent
            )
            expected_filename = str(relative_path).replace("\\", "/")
        else:
            # Fallback: try to calculate relative to docs/ai_handbook
            try:
                if "docs/ai_handbook" in str(file_path):
                    # Extract path after docs/ai_handbook
                    path_parts = str(file_path).split("docs/ai_handbook/")
                    if len(path_parts) > 1:
                        path_formatted = path_parts[1].replace("\\", "/")
                        expected_filename = f"docs/ai_handbook/{path_formatted}"
                    else:
                        expected_filename = f"docs/ai_handbook/{file_path.name}"
                else:
                    expected_filename = f"docs/ai_handbook/{file_path.relative_to(self.docs_dir.parent.parent) if self.docs_dir.parent.name == 'docs' else file_path.name}"
            except ValueError:
                # If relative path calculation fails, use a simpler approach
                expected_filename = f"docs/ai_handbook/{file_path.name}"

        filename_pattern = rf"# \*\*filename: {re.escape(expected_filename)}\*\*"
        if not re.search(filename_pattern, content, re.MULTILINE):
            self.errors.append(f"Incorrect or missing filename header in {file_path}")
            return False
        return True

    def validate_ai_cues(self, content: str, file_path: Path) -> bool:
        """Validate AI cue markers."""
        ai_cues = self.extract_ai_cues(content)
        has_errors = False

        for cue in self.required_ai_cues:
            if cue not in ai_cues:
                self.errors.append(f"Missing AI cue '{cue}' in {file_path}")
                has_errors = True
            elif not ai_cues[cue] or ai_cues[cue] == "{{" + cue + "}}":
                self.errors.append(f"Empty or placeholder AI cue '{cue}' in {file_path}")
                has_errors = True

        return not has_errors

    def validate_required_sections(self, content: str, file_path: Path, template_type: str) -> bool:
        """Validate required sections for the document type."""
        sections = self.extract_sections(content)
        if template_type in self.required_sections:
            required = self.required_sections[template_type]

            missing_sections = []
            for req_section in required:
                if req_section not in sections:
                    missing_sections.append(req_section)

            if missing_sections:
                self.errors.append(f"Missing required sections in {file_path}: {', '.join(missing_sections)}")
                return False
        return True

    def insert_missing_sections(self, content: str, file_path: Path, template_type: str) -> str:
        """Insert placeholder sections for missing required sections in correct order."""
        if template_type not in self.required_sections:
            return content

        sections = self.extract_sections(content)
        required = self.required_sections[template_type]

        # Find missing sections
        missing_sections = []
        for req_section in required:
            if req_section not in sections:
                missing_sections.append(req_section)

        if not missing_sections:
            return content

        # Find positions of existing required sections to determine insertion points
        content_lines = content.split("\n")
        existing_section_positions = {}

        for i, line in enumerate(content_lines):
            for section in required:
                if line.strip().startswith(f"## **{section}**"):
                    existing_section_positions[section] = i
                    break

        if existing_section_positions:
            # If some sections exist, insert missing sections in template order after the last existing section
            # Find the last existing section in the template order
            last_existing_pos = -1
            for section in required:
                if section in existing_section_positions and existing_section_positions[section] > last_existing_pos:
                    last_existing_pos = existing_section_positions[section]

            # Insert all missing sections after the last existing section
            insert_pos = last_existing_pos + 1
            for req_section in required:
                if req_section in missing_sections:
                    placeholder = f"## **{req_section}**\n\nTODO: Add content here.\n"
                    content_lines.insert(insert_pos, placeholder)
                    insert_pos += 1  # Next insertion point moves down
        else:
            # If no required sections exist, find a good insertion point (after the main content and before the first non-required section)
            # Look for a position after main headers and before other content
            insert_pos = len(content_lines)  # Default to end of file

            # Find the best insertion point - after headers and before other content
            for i, line in enumerate(content_lines):
                if line.strip().startswith("# **filename:") or line.strip().startswith("<!-- ai_cue:"):
                    continue  # Skip filename header and AI cues
                elif line.strip().startswith("# "):  # Main title
                    continue  # Skip main title
                else:
                    # Found first non-header content, insert before this
                    insert_pos = i
                    break

            # Insert missing sections in the correct template order
            for req_section in required:
                if req_section in missing_sections:
                    placeholder = f"## **{req_section}**\n\nTODO: Add content here.\n"
                    content_lines.insert(insert_pos, placeholder)
                    insert_pos += 1  # Next insertion point moves down

        return "\n".join(content_lines)

    def validate_placeholders(self, content: str, file_path: Path) -> bool:
        """Check for unresolved template placeholders."""
        placeholders = re.findall(r"\{\{[^}]+\}\}", content)
        if placeholders:
            unique_placeholders = set(placeholders)
            self.warnings.append(f"Unresolved template placeholders in {file_path}: {', '.join(unique_placeholders)}")
            return False
        return True

    def validate_headers_format(self, content: str, file_path: Path) -> list[str]:
        """Validate header formatting and return list of issues found."""
        issues = []

        # Check for headers that should be bold (level 2 headers)
        level2_headers = re.findall(r"^## (.+)$", content, re.MULTILINE)
        for header in level2_headers:
            if not (header.startswith("**") and header.endswith("**")):
                issues.append(f"Level 2 header '{header}' should be bold: ## **{header}**")

        # Check for headers that shouldn't be double-bold
        double_bold_headers = re.findall(r"^## \*\*\*\*(.+)\*\*\*\*$", content, re.MULTILINE)
        for header in double_bold_headers:
            issues.append(f"Level 2 header '{header}' has excessive bolding: ## **{header}**")

        if issues:
            self.warnings.extend([f"{file_path}: {issue}" for issue in issues])

        return issues

    def fix_headers_format(self, content: str) -> str:
        """Fix header formatting issues."""
        # Fix level 2 headers that should be bold
        content = re.sub(r"^## ([^*].*)$", r"## **\1**", content, flags=re.MULTILINE)

        # Fix double-bolded headers
        content = re.sub(r"^## \*\*\*\*(.+)\*\*\*\*$", r"## **\1**", content, flags=re.MULTILINE)

        return content

    def fix_filename_header(self, content: str, file_path: Path) -> str:
        """Fix the filename header."""
        # Calculate the relative path from the docs directory
        relative_path = file_path.relative_to(self.docs_dir)
        expected_filename = str(relative_path)
        filename_header_pattern = r"# \*\*filename: .+\*\*"

        if re.search(filename_header_pattern, content):
            # Replace existing filename header
            content = re.sub(filename_header_pattern, f"# **filename: {expected_filename}**", content, count=1)
        else:
            # Add filename header at the beginning
            content = f"# **filename: {expected_filename}**\n{content}"

        return content

    def fix_ai_cues(self, content: str) -> str:
        """Add missing AI cues if they don't exist."""
        ai_cues = self.extract_ai_cues(content)
        missing_cues = [cue for cue in self.required_ai_cues if cue not in ai_cues]

        if missing_cues:
            # Find position after the filename header
            filename_pos = content.find("\n", content.find("# **filename:"))
            if filename_pos == -1:
                filename_pos = 0

            # Add missing AI cues
            for cue in missing_cues:
                cue_str = f"<!-- ai_cue:{cue}=TODO -->\n"
                content = content[:filename_pos] + cue_str + content[filename_pos:]

        return content

    def determine_template_type(self, file_path: Path) -> str:
        """Determine which template type a file should follow based on its path."""
        path_str = str(file_path)

        if "02_protocols/development" in path_str:
            return "development"
        elif "02_protocols/configuration" in path_str:
            return "configuration"
        elif "02_protocols/governance" in path_str:
            return "governance"
        elif "02_protocols/components" in path_str:
            return "components"
        elif "03_references" in path_str:
            return "references"
        else:
            return "base"

    def standardize_file(self, file_path: Path) -> bool:
        """Standardize a single documentation file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                original_content = f.read()
        except Exception as e:
            self.errors.append(f"Cannot read file {file_path}: {e}")
            return False

        content = original_content
        template_type = self.determine_template_type(file_path)

        # Validate filename header
        self.validate_filename_header(content, file_path)

        # Validate AI cues
        self.validate_ai_cues(content, file_path)

        # Check for unresolved placeholders
        self.validate_placeholders(content, file_path)

        # Validate header formatting
        header_issues = self.validate_headers_format(content, file_path)

        # Validate required sections against original content (for reporting in dry-run)
        self.validate_required_sections(content, file_path, template_type)

        # Apply fixes if not in dry-run mode
        if not self.dry_run:
            # Fix header formatting
            if header_issues:
                content = self.fix_headers_format(content)
                self.fixes_applied.append(f"Fixed header formatting in {file_path}")

            # Fix filename header
            fixed_content = self.fix_filename_header(content, file_path)
            if fixed_content != content:
                content = fixed_content
                self.fixes_applied.append(f"Fixed filename header in {file_path}")

            # Fix AI cues
            fixed_content = self.fix_ai_cues(content)
            if fixed_content != content:
                content = fixed_content
                self.fixes_applied.append(f"Fixed AI cues in {file_path}")

            # Insert missing required sections as placeholders
            sections_before = set(self.extract_sections(content))
            fixed_content = self.insert_missing_sections(content, file_path, template_type)
            if fixed_content != content:
                content = fixed_content
                sections_after = set(self.extract_sections(content))
                added_sections = sections_after - sections_before
                if added_sections:
                    self.fixes_applied.append(f"Added placeholder sections in {file_path}: {', '.join(added_sections)}")

            # Re-validate required sections against final content after all fixes
            # Clear previous errors for required sections since we're re-validating
            self.errors = [e for e in self.errors if "Missing required sections" not in e]
            self.validate_required_sections(content, file_path, template_type)

            # Write back the standardized content if it changed
            if content != original_content:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"Standardized {file_path}")
                except Exception as e:
                    self.errors.append(f"Cannot write to file {file_path}: {e}")
                    return False

        return True

        # Apply fixes if not in dry-run mode
        if not self.dry_run:
            # Fix header formatting
            if header_issues:
                content = self.fix_headers_format(content)
                self.fixes_applied.append(f"Fixed header formatting in {file_path}")

            # Fix filename header
            fixed_content = self.fix_filename_header(content, file_path)
            if fixed_content != content:
                content = fixed_content
                self.fixes_applied.append(f"Fixed filename header in {file_path}")

            # Fix AI cues
            fixed_content = self.fix_ai_cues(content)
            if fixed_content != content:
                content = fixed_content
                self.fixes_applied.append(f"Fixed AI cues in {file_path}")

            # Insert missing required sections as placeholders
            sections_before = set(self.extract_sections(content))
            fixed_content = self.insert_missing_sections(content, file_path, template_type)
            if fixed_content != content:
                content = fixed_content
                sections_after = set(self.extract_sections(content))
                added_sections = sections_after - sections_before
                if added_sections:
                    self.fixes_applied.append(f"Added placeholder sections in {file_path}: {', '.join(added_sections)}")

            # Write back the standardized content if it changed
            if content != original_content:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"Standardized {file_path}")
                except Exception as e:
                    self.errors.append(f"Cannot write to file {file_path}: {e}")
                    return False

        return True

    def standardize_all_files(self) -> bool:
        """Standardize all documentation files."""
        # Find all markdown files in docs directory (excluding templates)
        doc_files: list[Path] = []

        if self.docs_dir.is_file():
            # Single file provided
            if self.docs_dir.suffix == ".md":
                doc_files = [self.docs_dir]
            else:
                self.warnings.append("Provided file is not a markdown file")
                return True
        else:
            # Directory provided
            for pattern in ["**/*.md"]:
                doc_files.extend(self.docs_dir.rglob(pattern))

        # Exclude template files themselves and deprecated files
        doc_files = [f for f in doc_files if "_templates" not in str(f) and "deprecated" not in str(f)]

        if not doc_files:
            self.warnings.append("No documentation files found to standardize")
            return True

        self.total_files = len(doc_files)
        print(f"Standardizing {len(doc_files)} documentation files")
        print(f"Dry run mode: {'ON' if self.dry_run else 'OFF'}")

        for file_path in doc_files:
            print(f"Processing {file_path}")
            self.standardize_file(file_path)

        # Report results
        success = len(self.errors) == 0

        return success

    def generate_standardization_report(self) -> dict:
        """Generate a standardization report."""
        return {
            "total_files": self.total_files,
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "total_fixes": len(self.fixes_applied),
            "errors": self.errors.copy(),
            "warnings": self.warnings.copy(),
            "fixes_applied": self.fixes_applied.copy(),
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Standardize AI Agent Handbook content")
    parser.add_argument("docs_dir", help="Documentation directory to standardize (or single file)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    parser.add_argument("--report-only", action="store_true", help="Only generate report without making changes (implies dry-run)")

    args = parser.parse_args()

    if not os.path.exists(args.docs_dir):
        print(f"Error: Path '{args.docs_dir}' does not exist")
        sys.exit(1)

    # If report-only is specified, enable dry-run mode
    dry_run = args.dry_run or args.report_only

    # Determine if the provided path is a file or directory
    path = Path(args.docs_dir)
    if path.is_file():
        # If it's a file, we need to determine the appropriate docs directory.
        # The docs_dir parameter in ContentStandardizer is used to calculate relative paths for filename headers.
        # For filename header like `# **filename: docs/ai_handbook/...**`, we need to make sure
        # file_path.relative_to(docs_dir) produces the correct relative path.
        # If our file is docs/ai_handbook/03_references/README.md, the correct header should be
        # docs/ai_handbook/03_references/README.md, so docs_dir should be the project root (parent of docs/)
        file_path = path

        # Find the project root by looking for the 'docs' directory in the hierarchy
        docs_parent_dir = None
        for parent in file_path.parents:
            if parent.name == "docs":
                # The docs_dir should be the parent of the 'docs' directory
                docs_parent_dir = parent.parent
                break

        if docs_parent_dir is None:
            print(f"Error: Could not find 'docs' directory in path hierarchy of {file_path}")
            sys.exit(1)

        standardizer = ContentStandardizer(docs_parent_dir, dry_run)
        # Process only this specific file
        success = standardizer.standardize_file(file_path)
        report = standardizer.generate_standardization_report()
    else:
        # If it's a directory, process all files in the directory
        standardizer = ContentStandardizer(args.docs_dir, dry_run)
        success = standardizer.standardize_all_files()
        report = standardizer.generate_standardization_report()

    # Print summary
    print("\n" + "=" * 50)
    print("STANDARDIZATION REPORT")
    print("=" * 50)
    print(f"Files processed: {report['total_files']}")
    print(f"Errors found: {report['total_errors']}")
    print(f"Warnings issued: {report['total_warnings']}")
    print(f"Fixes applied: {report['total_fixes']}")

    if report["errors"]:
        print(f"\n❌ Found {len(report['errors'])} errors:")
        for error in report["errors"]:
            print(f"  - {error}")

    if report["warnings"]:
        print(f"\n⚠️ Warnings ({len(report['warnings'])}):")
        for warning in report["warnings"]:
            print(f"  - {warning}")

    if report["fixes_applied"]:
        print(f"\n🔧 Fixes applied ({len(report['fixes_applied'])}):")
        for fix in report["fixes_applied"]:
            print(f"  - {fix}")

    if success and report["total_errors"] == 0:
        print("\n✅ All files meet standardization requirements!")
    else:
        print("\n❌ Some issues were found that require attention.")

    # Exit with error code if there were errors
    sys.exit(0 if success and report["total_errors"] == 0 else 1)


if __name__ == "__main__":
    main()
