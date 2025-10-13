#!/usr/bin/env python3
"""
Template Validation Script

Validates documentation files against standardized templates.
Ensures new documents follow proper structure and formatting.
"""

import os
import re
import sys
from pathlib import Path


class TemplateValidator:
    """Validates documentation files against templates."""

    def __init__(self, templates_dir: str, docs_dir: str):
        self.templates_dir = Path(templates_dir)
        self.docs_dir = Path(docs_dir)
        self.errors: list[str] = []
        self.warnings: list[str] = []

        # Template requirements
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

    def load_template(self, template_name: str) -> str | None:
        """Load a template file."""
        template_path = self.templates_dir / f"{template_name}.md"
        if template_path.exists():
            with open(template_path, encoding="utf-8") as f:
                return f.read()
        return None

    def extract_sections(self, content: str) -> list[str]:
        """Extract section headers from markdown content."""
        # Find all level 2 headers (## Header)
        headers = re.findall(r"^##\s+(.+)$", content, re.MULTILINE)
        # Strip ** markers from section names
        cleaned_headers = []
        for h in headers:
            # Remove ** markers if present
            h = h.strip()
            h = re.sub(r"^\*\*(.+)\*\*$", r"\1", h)
            cleaned_headers.append(h)
        return cleaned_headers

    def extract_ai_cues(self, content: str) -> dict[str, str]:
        """Extract AI cue comments from content."""
        cues = {}
        cue_pattern = r"<!-- ai_cue:(\w+)=(.+?) -->"
        matches = re.findall(cue_pattern, content)
        for key, value in matches:
            cues[key] = value.strip()
        return cues

    def validate_file_structure(self, file_path: Path, template_type: str) -> None:
        """Validate a file against its template structure."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"Cannot read file {file_path}: {e}")
            return

        # Check AI cues
        ai_cues = self.extract_ai_cues(content)
        for cue in self.required_ai_cues:
            if cue not in ai_cues:
                self.errors.append(f"Missing AI cue '{cue}' in {file_path}")
            elif not ai_cues[cue] or ai_cues[cue] == "{{" + cue + "}}":
                self.errors.append(f"Empty or placeholder AI cue '{cue}' in {file_path}")

        # Check required sections
        if template_type in self.required_sections:
            sections = self.extract_sections(content)
            required = self.required_sections[template_type]

            for req_section in required:
                if req_section not in sections:
                    self.errors.append(f"Missing required section '{req_section}' in {file_path}")

        # Check for template placeholders
        placeholders = re.findall(r"\{\{[^}]+\}\}", content)
        if placeholders:
            unique_placeholders = set(placeholders)
            self.warnings.append(f"Unresolved template placeholders in {file_path}: {', '.join(unique_placeholders)}")

        # Check filename header
        expected_filename = f"docs/ai_handbook/{file_path.relative_to(self.docs_dir)}"
        filename_pattern = rf"# \*\*filename: {re.escape(expected_filename)}\*\*"
        if not re.search(filename_pattern, content, re.MULTILINE):
            self.errors.append(f"Incorrect or missing filename header in {file_path}")

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

    def validate_all_files(self) -> bool:
        """Validate all documentation files against templates."""
        # Find all markdown files in docs directory (excluding templates)
        doc_files: list[Path] = []
        for pattern in ["**/*.md"]:
            doc_files.extend(self.docs_dir.rglob(pattern))

        # Exclude template files themselves
        doc_files = [f for f in doc_files if "_templates" not in str(f)]

        if not doc_files:
            self.warnings.append("No documentation files found to validate")
            return True

        print(f"Validating {len(doc_files)} documentation files against templates")

        for file_path in doc_files:
            template_type = self.determine_template_type(file_path)
            print(f"Validating {file_path} against {template_type} template")
            self.validate_file_structure(file_path, template_type)

        # Report results
        if self.errors:
            print(f"\n❌ Found {len(self.errors)} template validation errors:")
            for error in self.errors:
                print(f"  - {error}")
            return False
        else:
            print("\n✅ All files pass template validation!")
            return True

    def generate_template_report(self) -> dict:
        """Generate a validation report."""
        return {
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "errors": self.errors.copy(),
            "warnings": self.warnings.copy(),
        }


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python validate_templates.py <templates_dir> <docs_dir>")
        sys.exit(1)

    templates_dir = sys.argv[1]
    docs_dir = sys.argv[2]

    if not os.path.exists(templates_dir):
        print(f"Error: Templates directory '{templates_dir}' does not exist")
        sys.exit(1)

    if not os.path.exists(docs_dir):
        print(f"Error: Documentation directory '{docs_dir}' does not exist")
        sys.exit(1)

    validator = TemplateValidator(templates_dir, docs_dir)
    success = validator.validate_all_files()

    report = validator.generate_template_report()

    if report["warnings"]:
        print("\n⚠️ Warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")

    print("\nValidation Summary:")
    print(f"  Errors: {report['total_errors']}")
    print(f"  Warnings: {report['total_warnings']}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
