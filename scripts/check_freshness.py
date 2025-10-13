#!/usr/bin/env python3
"""
Documentation Freshness Checker

Checks how recently documentation files have been updated and flags files
that haven't been updated within specified time thresholds.
"""

import os
import sys
from datetime import datetime
from pathlib import Path


class FreshnessChecker:
    """Checks documentation freshness based on modification dates."""

    def __init__(self, docs_root: str, max_age_days: int = 30):
        self.docs_root = Path(docs_root)
        self.max_age_days = max_age_days
        self.now = datetime.now()
        self.stale_files: list[tuple[Path, int]] = []
        self.fresh_files: list[Path] = []

        # Common file extensions for documentation
        self.doc_extensions = {".md", ".markdown", ".txt", ".rst"}

        # Files to skip freshness checks
        self.skip_files = {"README.md", "CONTRIBUTING.md", "CHANGELOG.md"}

    def find_doc_files(self) -> list[Path]:
        """Find all documentation files in the docs directory."""
        doc_files: list[Path] = []
        for ext in self.doc_extensions:
            doc_files.extend(self.docs_root.rglob(f"*{ext}"))
        return [f for f in doc_files if f.name not in self.skip_files]

    def get_file_age_days(self, file_path: Path) -> int:
        """Get the age of a file in days since last modification."""
        try:
            mtime = file_path.stat().st_mtime
            modified_date = datetime.fromtimestamp(mtime)
            age = self.now - modified_date
            return age.days
        except Exception:
            return -1  # Error getting age

    def check_file_freshness(self, file_path: Path) -> None:
        """Check if a file is fresh or stale."""
        age_days = self.get_file_age_days(file_path)

        if age_days == -1:
            # Error getting file age
            self.stale_files.append((file_path, -1))
        elif age_days > self.max_age_days:
            self.stale_files.append((file_path, age_days))
        else:
            self.fresh_files.append(file_path)

    def check_all_freshness(self) -> bool:
        """Check freshness of all documentation files."""
        doc_files = self.find_doc_files()

        if not doc_files:
            print("No documentation files found")
            return True

        print(f"Checking freshness of {len(doc_files)} documentation files")
        print(f"Maximum allowed age: {self.max_age_days} days")

        for file_path in doc_files:
            self.check_file_freshness(file_path)

        # Report results
        if self.stale_files:
            print(f"\n⚠️ Found {len(self.stale_files)} stale documentation files:")
            for file_path, age in self.stale_files:
                if age == -1:
                    print(f"  - {file_path}: Error getting modification date")
                else:
                    print(f"  - {file_path}: {age} days old (max: {self.max_age_days})")
            return False
        else:
            print(f"\n✅ All {len(self.fresh_files)} documentation files are fresh!")
            return True

    def generate_report(self) -> dict:
        """Generate a freshness report."""
        return {
            "total_files": len(self.fresh_files) + len(self.stale_files),
            "fresh_files": len(self.fresh_files),
            "stale_files": len(self.stale_files),
            "max_age_days": self.max_age_days,
            "stale_file_details": [{"path": str(fp), "age_days": age} for fp, age in self.stale_files],
        }


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python check_freshness.py <docs_root> [max_age_days]")
        sys.exit(1)

    docs_root = sys.argv[1]
    max_age_days = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    if not os.path.exists(docs_root):
        print(f"Error: Documentation root '{docs_root}' does not exist")
        sys.exit(1)

    checker = FreshnessChecker(docs_root, max_age_days)
    success = checker.check_all_freshness()

    # Generate and print summary
    report = checker.generate_report()
    print("\nFreshness Summary:")
    print(f"  Total files: {report['total_files']}")
    print(f"  Fresh files: {report['fresh_files']}")
    print(f"  Stale files: {report['stale_files']}")
    print(f"  Max age: {report['max_age_days']} days")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
