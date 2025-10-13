#!/usr/bin/env python3
"""
Documentation Link Validator

Validates internal and external links in documentation files.
Checks for broken internal references and unreachable external URLs.
"""

import os
import re
import sys
from pathlib import Path

import requests


class LinkValidator:
    """Validates links in documentation files."""

    def __init__(self, docs_root: str):
        self.docs_root = Path(docs_root)
        self.errors: list[str] = []
        self.warnings: list[str] = []

        # Common file extensions for documentation
        self.doc_extensions = {".md", ".markdown", ".txt", ".rst"}

        # Files to skip
        self.skip_files = {"README.md", "CONTRIBUTING.md", "CHANGELOG.md"}

    def find_doc_files(self) -> list[Path]:
        """Find all documentation files in the docs directory."""
        doc_files: list[Path] = []
        for ext in self.doc_extensions:
            doc_files.extend(self.docs_root.rglob(f"*{ext}"))
        return [f for f in doc_files if f.name not in self.skip_files]

    def extract_links(self, file_path: Path) -> list[tuple[str, int]]:
        """Extract all links from a markdown file."""
        links = []
        try:
            with open(file_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    # Find markdown links: [text](url)
                    md_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", line)
                    for text, url in md_links:
                        links.append((url.strip(), line_num))

                    # Find reference-style links: [text][ref] or [ref]: url
                    ref_links = re.findall(r"\[([^\]]+)\]\[([^\]]+)\]", line)
                    for text, ref in ref_links:
                        links.append((f"#{ref}", line_num))

        except Exception as e:
            self.errors.append(f"Error reading {file_path}: {e}")

        return links

    def validate_internal_link(self, url: str, file_path: Path) -> bool:
        """Validate internal links (relative paths and anchors)."""
        if url.startswith(("http://", "https://", "mailto:", "tel:")):
            return True  # External links handled separately

        if url.startswith("#"):
            # Anchor link - check if it exists in the same file
            return self.check_anchor_exists(file_path, url[1:])

        # Relative path
        try:
            target_path = (file_path.parent / url).resolve()
            if target_path.exists():
                return True
            else:
                # Try with .md extension if not present
                if not target_path.suffix and "." not in target_path.name:
                    target_path = target_path.with_suffix(".md")
                    return target_path.exists()
        except Exception:
            pass

        return False

    def check_anchor_exists(self, file_path: Path, anchor: str) -> bool:
        """Check if an anchor exists in a file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                # Look for headers that would generate this anchor
                # GitHub-style anchor generation: lowercase, spaces to hyphens
                anchor_pattern = re.compile(rf"^#+\s+{re.escape(anchor.replace('-', ' '))}", re.MULTILINE | re.IGNORECASE)
                return bool(anchor_pattern.search(content))
        except Exception:
            return False

    def validate_external_link(self, url: str) -> bool:
        """Validate external links by making HTTP requests."""
        try:
            # Only check HTTP/HTTPS links
            if not url.startswith(("http://", "https://")):
                return True

            response = requests.head(url, timeout=10, allow_redirects=True)
            return response.status_code < 400
        except Exception:
            return False

    def validate_file(self, file_path: Path) -> None:
        """Validate all links in a single file."""
        links = self.extract_links(file_path)

        for url, line_num in links:
            if not self.validate_internal_link(url, file_path):
                if url.startswith(("http://", "https://")):
                    if not self.validate_external_link(url):
                        self.errors.append(f"Broken external link in {file_path}:{line_num}: {url}")
                else:
                    self.errors.append(f"Broken internal link in {file_path}:{line_num}: {url}")

    def validate_all(self) -> bool:
        """Validate all documentation files."""
        doc_files = self.find_doc_files()

        if not doc_files:
            self.warnings.append("No documentation files found")
            return True

        print(f"Found {len(doc_files)} documentation files to validate")

        for file_path in doc_files:
            print(f"Validating {file_path}")
            self.validate_file(file_path)

        # Report results
        if self.errors:
            print(f"\n❌ Found {len(self.errors)} link errors:")
            for error in self.errors:
                print(f"  - {error}")
            return False
        else:
            print("\n✅ All links are valid!")
            return True


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python validate_links.py <docs_root>")
        sys.exit(1)

    docs_root = sys.argv[1]

    if not os.path.exists(docs_root):
        print(f"Error: Documentation root '{docs_root}' does not exist")
        sys.exit(1)

    validator = LinkValidator(docs_root)
    success = validator.validate_all()

    if validator.warnings:
        print("\n⚠️ Warnings:")
        for warning in validator.warnings:
            print(f"  - {warning}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
