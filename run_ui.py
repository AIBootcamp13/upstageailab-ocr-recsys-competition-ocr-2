#!/usr/bin/env python3
"""
Streamlit UI Runner

This script provides convenient commands to run the Streamlit UI applications.
"""

import subprocess
import sys
from pathlib import Path


def run_command_builder():
    """Run the command builder UI."""
    ui_path = Path(__file__).parent / "ui" / "command_builder.py"
    cmd = ["uv", "run", "streamlit", "run", str(ui_path)]
    subprocess.run(cmd)


def run_evaluation_viewer():
    """Run the evaluation results viewer UI."""
    ui_path = Path(__file__).parent / "ui" / "evaluation_viewer.py"
    cmd = ["uv", "run", "streamlit", "run", str(ui_path)]
    subprocess.run(cmd)


def run_inference_ui():
    """Run the real-time inference UI."""
    ui_path = Path(__file__).parent / "ui" / "inference_ui.py"
    cmd = ["uv", "run", "streamlit", "run", str(ui_path)]
    subprocess.run(cmd)


def run_resource_monitor():
    """Run the resource monitor UI."""
    ui_path = Path(__file__).parent / "ui" / "resource_monitor.py"
    cmd = ["uv", "run", "streamlit", "run", str(ui_path)]
    subprocess.run(cmd)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_ui.py <command>")
        print("Commands:")
        print("  command_builder  - Run the CLI command builder UI")
        print("  evaluation_viewer - Run the evaluation results viewer UI")
        print("  inference        - Run the real-time inference UI")
        print("  resource_monitor - Run the system resource monitor UI")
        sys.exit(1)

    command = sys.argv[1]

    if command == "command_builder":
        run_command_builder()
    elif command == "evaluation_viewer":
        run_evaluation_viewer()
    elif command == "inference":
        run_inference_ui()
    elif command == "resource_monitor":
        run_resource_monitor()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: command_builder, evaluation_viewer, inference, resource_monitor")
        sys.exit(1)
