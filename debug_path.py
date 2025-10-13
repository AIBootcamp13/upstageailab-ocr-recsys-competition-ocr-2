#!/usr/bin/env python3
"""Debug script to check path handling."""

from pathlib import Path

# Test the path handling that's happening in the standardizer
docs_dir = Path("docs/ai_handbook")
file_path = Path("docs/ai_handbook/03_references/README.md")

print(f"docs_dir: {docs_dir}")
print(f"file_path: {file_path}")
print(f"file_path.relative_to(docs_dir): {file_path.relative_to(docs_dir)}")

# This is what the standardizer does:
expected_filename = f"docs/ai_handbook/{file_path.relative_to(docs_dir)}"
print(f"expected_filename: {expected_filename}")
