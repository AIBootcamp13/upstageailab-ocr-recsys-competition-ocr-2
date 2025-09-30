from __future__ import annotations

"""Utility for stripping or restoring AI documentation markers in source files."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = ROOT / "tmp" / "ai_docs_markers.json"
MARKER_START = "# AI_DOCS["
MARKER_END = "# ]"


def iter_python_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*.py") if ".venv" not in path.parts and "__pycache__" not in path.parts]


def strip_markers(path: Path) -> tuple[str, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    new_lines: list[str] = []
    captured_blocks: list[str] = []
    capturing = False
    buffer: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(MARKER_START):
            capturing = True
            buffer = [line]
            continue
        if capturing:
            buffer.append(line)
            if stripped == MARKER_END:
                captured_blocks.append("\n".join(buffer))
                capturing = False
                buffer = []
            continue
        new_lines.append(line)

    if capturing and buffer:
        captured_blocks.append("\n".join(buffer))

    return "\n".join(new_lines) + "\n", captured_blocks


def restore_markers(path: Path, markers: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text += "\n"
    restored = text + "\n".join(markers)
    path.write_text(restored.rstrip("\n") + "\n", encoding="utf-8")


def ensure_snapshot_dir() -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_snapshot() -> dict[str, Any]:
    if not SNAPSHOT_PATH.exists():
        raise SystemExit(f"Snapshot file not found: {SNAPSHOT_PATH}")
    with SNAPSHOT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_snapshot(data: dict[str, Any]) -> None:
    ensure_snapshot_dir()
    with SNAPSHOT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage AI documentation markers in source files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true", help="Remove markers and save snapshot.")
    group.add_argument("--dry-run", action="store_true", help="Preview files containing markers.")
    group.add_argument("--restore", action="store_true", help="Restore markers from the latest snapshot.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root: Path = args.root

    if args.restore:
        snapshot = load_snapshot()
        for file_str, markers in snapshot.get("files", {}).items():
            path = root / file_str
            restore_markers(path, markers)
            print(f"restored {len(markers)} marker blocks in {file_str}")
        return

    files = iter_python_files(root)
    results: dict[str, list[str]] = {}

    for path in files:
        new_text, blocks = strip_markers(path)
        if not blocks:
            continue
        rel_path = path.relative_to(root)
        results[str(rel_path)] = blocks
        if args.apply:
            path.write_text(new_text, encoding="utf-8")
            print(f"stripped {len(blocks)} marker blocks from {rel_path}")
        elif args.dry_run:
            print(f"{rel_path}: {len(blocks)} block(s)")

    if args.apply and results:
        save_snapshot({"files": results})
    elif args.apply:
        print("No markers found to strip.")


if __name__ == "__main__":
    main()
