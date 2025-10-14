"""Thin entrypoint delegating to the modular command builder app."""

import warnings

# Suppress known wandb Pydantic compatibility warnings
# This is a known issue where wandb uses incorrect Field() syntax in Annotated types
# The warnings come from Pydantic when processing wandb's type annotations
warnings.filterwarnings("ignore", message=r"The '(repr|frozen)' attribute.*Field.*function.*no effect", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*(repr|frozen).*Field.*function.*no effect", category=UserWarning)

# Also suppress by category for more reliable filtering
try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning

    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except ImportError:
    pass  # In case the warning class is not available in future pydantic versions

from ui.apps.command_builder import main as app_main


def main() -> None:
    app_main()


if __name__ == "__main__":
    main()
