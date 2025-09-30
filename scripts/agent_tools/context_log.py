from __future__ import annotations

"""Utilities and CLI helpers for managing agent context logs."""

import argparse
import json
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TIMESTAMP_FMT = "%Y-%m-%d_%H-%M-%S"


def get_log_root() -> Path:
    env_dir = os.environ.get("AGENT_CONTEXT_LOG_DIR")
    return Path(env_dir) if env_dir else Path("logs/agent_runs")


def _ensure_log_dir() -> Path:
    root = get_log_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sanitize_label(label: str | None) -> str:
    if not label:
        return ""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", label.strip().lower()).strip("-")
    return f"_{slug}" if slug else ""


def create_log_file(label: str | None = None) -> Path:
    root = _ensure_log_dir()
    if label is None:
        label = os.environ.get("AGENT_CONTEXT_LOG_LABEL")
    timestamp = datetime.now(timezone.utc).strftime(TIMESTAMP_FMT)
    suffix = _sanitize_label(label)
    path = root / f"{timestamp}{suffix}.jsonl"
    path.touch(exist_ok=True)
    return path


def log_agent_action(
    log_path: Path,
    action: str,
    *,
    thought: str | None = None,
    outcome: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    output_snippet: str | None = None,
    timestamp: datetime | None = None,
) -> None:
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    event_time = timestamp or datetime.now(timezone.utc)
    entry = {
        "timestamp": event_time.isoformat().replace("+00:00", "Z"),
        "action": action,
        "thought": thought,
        "outcome": outcome,
        "parameters": parameters or {},
        "output_snippet": output_snippet,
    }

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def summarize_log(log_path: Path) -> None:
    from scripts.agent_tools.summarize_run import summarize_log_file  # Late import to avoid API dependency when not needed

    summarize_log_file(log_path)


def parse_parameters(raw: str | None) -> Mapping[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - user error path
        raise argparse.ArgumentTypeError(f"Could not parse parameters JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):  # pragma: no cover - user error path
        raise argparse.ArgumentTypeError("Parameters must decode to a JSON object")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage agent context logs")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Create a new context log and output its path")
    start.add_argument("--label", help="Optional short label for the log filename")

    append = sub.add_parser("log", help="Append an action entry to an existing log")
    append.add_argument("--log-file", type=Path, required=True, help="Path to the .jsonl log file")
    append.add_argument("--action", required=True, help="Name of the action performed")
    append.add_argument("--thought", help="Reasoning associated with the action")
    append.add_argument("--outcome", help="Outcome status (success, failure, skipped, etc.)")
    append.add_argument("--parameters", help="JSON object of parameters for the action")
    append.add_argument("--output", help="Short snippet of output or result")

    summarize = sub.add_parser("summarize", help="Generate a Markdown summary for a log")
    summarize.add_argument("--log-file", type=Path, required=True, help="Path to the .jsonl log file")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "start":
        log_path = create_log_file(args.label)
        print(log_path)
        return

    if args.command == "log":
        params = parse_parameters(args.parameters)
        log_agent_action(
            args.log_file,
            action=args.action,
            thought=args.thought,
            outcome=args.outcome,
            parameters=params,
            output_snippet=args.output,
        )
        print(f"Logged action to {args.log_file}")
        return

    if args.command == "summarize":
        summarize_log(args.log_file)
        return


if __name__ == "__main__":
    main()
