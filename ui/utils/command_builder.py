"""
Command Builder for Streamlit UI (Backward Compatibility Wrapper)

This module provides backward compatibility for the refactored command builder.
The functionality has been moved to the ui.utils.command package for better modularity.
"""

from ui.utils.command import CommandBuilder as _CommandBuilder
from ui.utils.command.executor import CommandExecutor as _CommandExecutor
from ui.utils.command.validator import CommandValidator as _CommandValidator

# Re-export the main classes for backward compatibility
CommandBuilder = _CommandBuilder
CommandExecutor = _CommandExecutor
CommandValidator = _CommandValidator

# For backward compatibility, preserve the original interface
__all__ = ["CommandBuilder", "CommandExecutor", "CommandValidator"]

# Deprecation notice
import warnings

warnings.warn(
    "The command_builder module has been refactored into ui.utils.command for better modularity. "
    "Please update your imports to use 'from ui.utils.command import CommandBuilder' instead. "
    "This backward compatibility wrapper will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
