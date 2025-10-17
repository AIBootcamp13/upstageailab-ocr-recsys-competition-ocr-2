#!/usr/bin/env python3
"""
Streamlit UI Runner

This script provides convenient commands to run the Streamlit UI applications.
"""

import subprocess
import sys

from ocr.utils.path_utils import get_path_resolver, setup_project_paths

setup_project_paths()


def run_command_builder():
    """Run the command builder UI."""
    ui_path = get_path_resolver().config.project_root / "ui" / "command_builder.py"
    cmd = ["uv", "run", "streamlit", "run", str(ui_path)]
    subprocess.run(cmd)


def run_evaluation_viewer():
    """Run the evaluation results viewer UI."""
    ui_path = get_path_resolver().config.project_root / "ui" / "evaluation_viewer.py"
    cmd = ["uv", "run", "streamlit", "run", str(ui_path)]
    subprocess.run(cmd)


def run_inference_ui():
    """Run the real-time inference UI."""
    ui_path = get_path_resolver().config.project_root / "ui" / "inference_ui.py"
    cmd = ["uv", "run", "streamlit", "run", str(ui_path)]
    subprocess.run(cmd)


def run_preprocessing_viewer():
    """Run the preprocessing viewer UI."""
    ui_path = get_path_resolver().config.project_root / "ui" / "preprocessing_viewer_app.py"
    cmd = ["uv", "run", "streamlit", "run", str(ui_path)]
    subprocess.run(cmd)


def run_resource_monitor():
    """Run the resource monitor UI."""
    ui_path = get_path_resolver().config.project_root / "ui" / "resource_monitor.py"
    cmd = ["uv", "run", "streamlit", "run", str(ui_path)]
    subprocess.run(cmd)
    if len(sys.argv) < 2:
        print("Usage: python run_ui.py <command>")
        print("Commands:")
        print("  command_builder  - Run the CLI command builder UI")
        print("  evaluation_viewer - Run the evaluation results viewer UI")
        print("  inference        - Run the real-time inference UI")
        print("  preprocessing_viewer - Run the preprocessing pipeline viewer UI")
        print("  resource_monitor - Run the system resource monitor UI")
        sys.exit(1)

    command = sys.argv[1]

    if command == "command_builder":
        run_command_builder()
    elif command == "evaluation_viewer":
        run_evaluation_viewer()
    elif command == "inference":
        run_inference_ui()
    elif command == "preprocessing_viewer":
        run_preprocessing_viewer()
    elif command == "resource_monitor":
        run_resource_monitor()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: command_builder, evaluation_viewer, inference, preprocessing_viewer, resource_monitor")
        sys.exit(1)
