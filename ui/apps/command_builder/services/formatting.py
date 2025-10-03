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
    """Pretty-format a long uv/python command into Bash-friendly multi-line string.

    Preserves shell quoting for arguments that need it (e.g., values with special chars).
    """
    if not cmd:
        return ""
    fallback_note = None

    # Parse the command while tracking which parts were originally quoted
    try:
        parts = shlex.split(cmd)
    except Exception as exc:  # noqa: BLE001 - display fallback note to user
        parts = cmd.split()
        fallback_note = f"# Note: best-effort formatting due to quoting error: {exc}"

    # Re-quote parts that need shell quoting
    # After shlex.split, we lose the original quotes, so we need to re-quote
    # args that contain special characters
    quoted_parts = []
    for part in parts:
        # Check if this part needs shell quoting
        # These are Hydra overrides with special chars in values
        if "=" in part:
            key, value = part.split("=", 1)
            # If value contains special chars, needs to be quoted for shell
            special_chars = ["=", " ", "\t", "'", '"', ",", ":", "{", "}", "[", "]"]
            if any(ch in value for ch in special_chars):
                # Wrap in single quotes for shell (preserving any double quotes in value)
                quoted_parts.append(f"'{part}'")
            else:
                quoted_parts.append(part)
        else:
            quoted_parts.append(part)

    first_break_idx = min(4, len(quoted_parts))
    with suppress(ValueError):
        cfg_idx = quoted_parts.index("--config-path")
        first_break_idx = max(first_break_idx, cfg_idx + 2)

    head = quoted_parts[:first_break_idx]
    tail = quoted_parts[first_break_idx:]

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
