from __future__ import annotations

import shlex
from contextlib import suppress


def format_command_output(output: str) -> str:
    """Format command output with lightweight glyph cues."""
    if not output.strip():
        return output

    lines = output.split("\n")
    formatted_lines: list[str] = []

    for line in lines:
        lower = line.lower()
        if "epoch" in lower and "loss" in lower:
            formatted_lines.append(f"📊 {line}")
        elif any(keyword in lower for keyword in ["error", "exception", "failed", "traceback"]):
            formatted_lines.append(f"❌ {line}")
        elif "warning" in lower:
            formatted_lines.append(f"⚠️ {line}")
        elif any(keyword in lower for keyword in ["success", "completed", "done"]):
            formatted_lines.append(f"✅ {line}")
        elif line.strip().startswith("[") and "]" in line:
            formatted_lines.append(f"🔄 {line}")
        else:
            formatted_lines.append(line)

    return "\n".join(formatted_lines)


def pretty_format_command(cmd: str) -> str:
    """Pretty-format a long uv/python command into Bash-friendly multi-line string."""
    if not cmd:
        return ""
    fallback_note = None
    try:
        parts = shlex.split(cmd)
    except Exception as exc:  # noqa: BLE001 - display fallback note to user
        parts = cmd.split()
        fallback_note = f"# Note: best-effort formatting due to quoting error: {exc}"

    first_break_idx = min(4, len(parts))
    with suppress(ValueError):
        cfg_idx = parts.index("--config-path")
        first_break_idx = max(first_break_idx, cfg_idx + 2)

    head = parts[:first_break_idx]
    tail = parts[first_break_idx:]

    lines: list[str] = []
    if head:
        head_line = " ".join(head)
        lines.append(head_line + (" " + "\\" if tail else ""))
    if tail:
        for i, tok in enumerate(tail):
            suffix = "" if i == len(tail) - 1 else " " + "\\"
            lines.append(f"  {tok}{suffix}")
    if fallback_note:
        lines.insert(0, fallback_note)
    return "\n".join(lines)
