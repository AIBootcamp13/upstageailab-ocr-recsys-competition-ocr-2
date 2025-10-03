"""
Hydra/Shell Quoting Utilities

Utilities for properly quoting Hydra override values for both Hydra and shell compatibility.
"""


def quote_override(ov: str) -> str:
    """Quote override values for both Hydra and shell compatibility.

    Hydra needs special characters in values to be quoted with double quotes.
    The shell needs the entire override wrapped in single quotes to preserve those double quotes.

    Following Hydra best practices:
    - For Hydra: key="value" (double quotes around value with special chars)
    - For shell: 'key="value"' (single quotes around entire override)
    """
    # Check if already properly shell-quoted
    if ov.startswith("'") and ov.endswith("'"):
        return ov

    if "=" not in ov:
        return ov

    key, value = ov.split("=", 1)

    # Check if value already has Hydra quotes (double quotes)
    value_is_quoted = value.startswith('"') and value.endswith('"')

    # Characters that need Hydra quoting
    special_chars = ["=", " ", "\t", "'", ",", ":", "{", "}", "[", "]"]
    needs_hydra_quotes = any(ch in value for ch in special_chars)

    # Apply Hydra quoting if needed
    if needs_hydra_quotes and not value_is_quoted:
        # Escape any double quotes in the value
        escaped_value = value.replace('"', '\\"')
        hydra_override = f'{key}="{escaped_value}"'
    else:
        hydra_override = ov

    # Wrap in single quotes for shell if it contains special chars or is already Hydra-quoted
    if needs_hydra_quotes or value_is_quoted:
        return f"'{hydra_override}'"

    return hydra_override


def is_special_char(value: str) -> bool:
    """Check if value contains special characters that need quoting."""
    special_chars = ["=", " ", "\t", "'", ",", ":", "{", "}", "[", "]"]
    return any(ch in value for ch in special_chars)
