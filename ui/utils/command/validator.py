"""
Command Validator

Validates CLI commands before execution to ensure they are properly formatted.
"""

from pathlib import Path


class CommandValidator:
    """Validate CLI commands before execution."""

    def validate_command(self, command: str) -> tuple[bool, str]:
        """Validate that a command can be executed.

        Args:
            command: Command string to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            parts = command.split()
            if not parts:
                return False, "Empty command"

            # Check for 'uv run python' structure
            if parts[:3] == ["uv", "run", "python"]:
                script_path = Path(parts[3])
                if not script_path.exists():
                    return False, f"Script not found: {script_path}"
                return True, ""

            return False, "Command must start with 'uv run python'"

        except Exception as e:
            return False, f"Command validation error: {e}"
