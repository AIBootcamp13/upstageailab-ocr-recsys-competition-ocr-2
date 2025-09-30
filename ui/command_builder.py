"""Thin entrypoint delegating to the modular command builder app."""

from ui.apps.command_builder import main as app_main


def main() -> None:
    app_main()


if __name__ == "__main__":
    main()
