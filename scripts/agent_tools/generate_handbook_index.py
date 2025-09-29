#!/usr/bin/env python3
"""Generate a lightweight JSON index from docs/ai_handbook/index.md."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

LINK_PATTERN = re.compile(r"\[(?P<title>[^\]]+)\]\((?P<path>[^)]+)\)")
HEADING_PATTERN = re.compile(r"^(?P<level>#{2,4})\s+(?P<title>.+?)\s*$")
VERSION_PATTERN = re.compile(r"^Version:\s*(?P<version>.+?)\s*$", re.IGNORECASE)


def parse_links(markdown_lines: Iterable[str]) -> dict[str, list[dict[str, str]]]:
    """Extract links grouped by their most recent heading."""
    current_section = "Ungrouped"
    sections: dict[str, list[dict[str, str]]] = {current_section: []}

    for line in markdown_lines:
        if match := HEADING_PATTERN.match(line):
            level = len(match.group("level"))
            title = match.group("title").strip().strip("*")
            # Only treat level 3/4 headings as sections to avoid top-level headers
            if level >= 3:
                current_section = title
                sections.setdefault(current_section, [])
            continue

        for match in LINK_PATTERN.finditer(line):
            entry = {"title": match.group("title").strip(), "path": match.group("path").strip()}
            sections.setdefault(current_section, []).append(entry)

    # Drop empty section placeholders
    return {section: entries for section, entries in sections.items() if entries}


def extract_version(markdown_lines: Iterable[str]) -> str | None:
    for line in markdown_lines:
        if match := VERSION_PATTERN.match(line):
            return match.group("version").strip()
    return None


def build_index(source: Path) -> dict:
    lines = source.read_text(encoding="utf-8").splitlines()
    sections = parse_links(lines)
    version = extract_version(lines)
    entries = [
        {
            "title": item["title"],
            "path": item["path"],
            "section": section,
        }
        for section, links in sections.items()
        for item in links
    ]

    return {
        "source": str(source),
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSON index for the AI handbook.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("docs/ai_handbook/index.md"),
        help="Path to the index markdown file (default: docs/ai_handbook/index.md).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/ai_handbook/index.json"),
        help="Destination for the generated JSON (default: docs/ai_handbook/index.json).",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON to stdout instead of writing to the output file.",
    )
    args = parser.parse_args()

    index_data = build_index(args.source)

    if args.stdout:
        print(json.dumps(index_data, indent=2, ensure_ascii=False))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(index_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON index written to {args.output}")


if __name__ == "__main__":
    main()
