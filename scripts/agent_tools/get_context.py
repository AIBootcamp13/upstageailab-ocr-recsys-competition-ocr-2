from __future__ import annotations

"""Lookup utility for documentation context bundles.

Usage examples:
    uv run python scripts/agent_tools/get_context.py --bundle streamlit-maintenance
    uv run python scripts/agent_tools/get_context.py --list-bundles
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DOC_INDEX_PATH = Path("docs/ai_handbook/index.json")


def load_index() -> dict[str, Any]:
    if not DOC_INDEX_PATH.exists():
        msg = f"Handbook index not found: {DOC_INDEX_PATH}"
        raise SystemExit(msg)

    with DOC_INDEX_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def print_bundle(index: dict[str, Any], bundle_name: str) -> None:
    bundles: dict[str, Any] = index.get("bundles", {})
    entries: list[dict[str, Any]] = index.get("entries", [])

    if bundle_name not in bundles:
        available = ", ".join(sorted(bundles)) or "<none>"
        raise SystemExit(f"Unknown bundle '{bundle_name}'. Available: {available}")

    bundle = bundles[bundle_name]
    print(f"Bundle: {bundle.get('title', bundle_name)}")
    if description := bundle.get("description"):
        print(f"Description: {description}")
    print()

    entry_map = {entry.get("id"): entry for entry in entries}

    for entry_id in bundle.get("entries", []):
        entry = entry_map.get(entry_id)
        if not entry:
            print(f"- {entry_id} (missing from entries)")
            continue

        title = entry.get("title", entry_id)
        path = entry.get("path")
        priority = entry.get("priority", "unknown")
        summary = entry.get("summary", "")
        print(f"- {title}")
        if path:
            print(f"    path: {path}")
        print(f"    priority: {priority}")
        if summary:
            print(f"    summary: {summary}")
        tags = entry.get("tags") or []
        if tags:
            print(f"    tags: {', '.join(tags)}")
        print()


def list_bundles(index: dict[str, Any]) -> None:
    bundles: dict[str, Any] = index.get("bundles", {})
    if not bundles:
        print("No bundles defined.")
        return

    for name, meta in sorted(bundles.items()):
        title = meta.get("title", name)
        description = meta.get("description", "")
        print(f"- {name}: {title}")
        if description:
            print(f"    {description}")


def lookup_entry(index: dict[str, Any], entry_id: str) -> None:
    entries: list[dict[str, Any]] = index.get("entries", [])
    for entry in entries:
        if entry.get("id") == entry_id:
            print(json.dumps(entry, indent=2))
            return
    raise SystemExit(f"Entry '{entry_id}' not found.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print documentation bundles or entries for agents.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bundle", help="Bundle identifier to print.")
    group.add_argument("--entry", help="Entry identifier to print details for.")
    group.add_argument("--list-bundles", action="store_true", help="List available bundles.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    index = load_index()

    if args.list_bundles:
        list_bundles(index)
    elif args.entry:
        lookup_entry(index, args.entry)
    elif args.bundle:
        print_bundle(index, args.bundle)


if __name__ == "__main__":
    main()
