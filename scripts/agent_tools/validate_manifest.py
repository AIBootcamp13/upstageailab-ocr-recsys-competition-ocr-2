from __future__ import annotations

"""Validate the AI handbook manifest for structural and referential integrity."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "docs/ai_handbook/index.json"


class ManifestValidationError(Exception):
    """Raised when the manifest cannot be parsed."""


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ManifestValidationError(f"Manifest not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - handled as fatal
        raise ManifestValidationError(f"Failed to parse manifest: {exc}") from exc


def validate_manifest(data: dict[str, Any], manifest_path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    manifest_dir = manifest_path.parent.resolve()
    entries_obj = data.get("entries")
    if not isinstance(entries_obj, list):
        errors.append("'entries' must be a list")
        return errors, warnings

    entry_ids: set[str] = set()
    entry_to_bundles: dict[str, list[str]] = {}

    for index, entry in enumerate(entries_obj):
        context = f"entries[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{context} must be an object")
            continue

        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            errors.append(f"{context} missing valid 'id'")
            continue

        if entry_id in entry_ids:
            errors.append(f"Duplicate entry id: {entry_id}")
        else:
            entry_ids.add(entry_id)

        for field in ("title", "path", "section"):
            if field not in entry or not isinstance(entry[field], str) or not entry[field].strip():
                errors.append(f"{context} ({entry_id}) missing field '{field}'")

        path_value = entry.get("path")
        if isinstance(path_value, str) and path_value.strip():
            target_path = (manifest_dir / Path(path_value)).resolve()
            try:
                target_path.relative_to(manifest_dir)
            except ValueError:
                errors.append(f"Entry '{entry_id}' path escapes docs root: {path_value}")
            else:
                if not target_path.exists():
                    errors.append(f"Entry '{entry_id}' path not found: {path_value}")
        else:
            errors.append(f"Entry '{entry_id}' has invalid 'path'")

        bundles_field = entry.get("bundles", [])
        if bundles_field is None:
            bundles_field = []
        if not isinstance(bundles_field, list):
            errors.append(f"Entry '{entry_id}' bundles must be a list")
            continue

        bundle_names: list[str] = []
        for bundle_name in bundles_field:
            if not isinstance(bundle_name, str) or not bundle_name.strip():
                errors.append(f"Entry '{entry_id}' has invalid bundle reference: {bundle_name!r}")
                continue
            bundle_names.append(bundle_name)
        entry_to_bundles[entry_id] = bundle_names

    bundles_obj = data.get("bundles", {})
    if not isinstance(bundles_obj, dict):
        errors.append("'bundles' must be an object")
        bundles_obj = {}

    bundle_names_set = set(bundles_obj)
    for entry_id, bundles in entry_to_bundles.items():
        for bundle_name in bundles:
            if bundle_name not in bundle_names_set:
                errors.append(f"Entry '{entry_id}' references unknown bundle '{bundle_name}'")

    bundle_entry_refs: dict[str, set[str]] = {name: set() for name in bundle_names_set}

    for bundle_name, bundle in bundles_obj.items():
        if not isinstance(bundle, dict):
            errors.append(f"Bundle '{bundle_name}' must be an object")
            continue

        entries_value = bundle.get("entries", [])
        if not isinstance(entries_value, list):
            errors.append(f"Bundle '{bundle_name}' entries must be a list")
            continue

        seen_entries: set[str] = set()
        for ref in entries_value:
            if not isinstance(ref, str):
                errors.append(f"Bundle '{bundle_name}' contains invalid entry reference: {ref!r}")
                continue
            if ref in seen_entries:
                warnings.append(f"Bundle '{bundle_name}' references entry '{ref}' more than once")
            seen_entries.add(ref)
            if ref not in entry_ids:
                errors.append(f"Bundle '{bundle_name}' references missing entry '{ref}'")
            else:
                bundle_entry_refs[bundle_name].add(ref)

    commands_obj = data.get("commands", [])
    if commands_obj is None:
        commands_obj = []
    if not isinstance(commands_obj, list):
        errors.append("'commands' must be a list")
    else:
        command_names: set[str] = set()
        for index, command in enumerate(commands_obj):
            context = f"commands[{index}]"
            if not isinstance(command, dict):
                errors.append(f"{context} must be an object")
                continue

            name = command.get("name")
            cmd = command.get("command")
            bundle_name = command.get("bundle")

            if not isinstance(name, str) or not name.strip():
                errors.append(f"{context} missing valid 'name'")
            elif name in command_names:
                warnings.append(f"Duplicate command name detected: {name}")
            else:
                command_names.add(name)

            if not isinstance(cmd, str) or not cmd.strip():
                errors.append(f"{context} missing valid 'command'")

            if bundle_name and bundle_name not in bundle_names_set:
                errors.append(f"{context} references unknown bundle '{bundle_name}'")

    bundled_entries = {ref for refs in bundle_entry_refs.values() for ref in refs}
    unbundled = sorted(entry_ids - bundled_entries)
    if unbundled:
        warnings.append("Entries not referenced in any bundle: " + ", ".join(unbundled))

    return errors, warnings


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the AI handbook manifest.")
    parser.add_argument(
        "manifest",
        nargs="?",
        default=DEFAULT_MANIFEST,
        type=Path,
        help="Path to the manifest JSON file.",
    )
    parser.add_argument(
        "--allow-unbundled",
        action="store_true",
        help="Downgrade unbundled entry warnings to informational messages.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest_path: Path = args.manifest

    try:
        data = load_manifest(manifest_path)
    except ManifestValidationError as exc:
        print(str(exc))
        raise SystemExit(1) from exc

    errors, warnings = validate_manifest(data, manifest_path)

    if args.allow_unbundled:
        warnings = [w for w in warnings if not w.startswith("Entries not referenced")]

    if errors:
        print("Manifest check: FAIL")
        for message in errors:
            print(f"ERROR: {message}")
        for message in warnings:
            print(f"WARNING: {message}")
        raise SystemExit(1)

    print("Manifest check: PASS")
    for message in warnings:
        print(f"WARNING: {message}")


if __name__ == "__main__":
    main()
