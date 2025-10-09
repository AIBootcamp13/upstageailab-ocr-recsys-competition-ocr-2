#!/usr/bin/env python3
"""
Seroost indexing setup script for the OCR project.

This script configures Seroost to index source code, documentation,
and configuration files while excluding build artifacts, dependencies,
and large data files.
"""

import json
import subprocess
from pathlib import Path


def setup_seroost_index():
    """
    Sets up the Seroost index with the project-specific configuration.
    """
    project_root = Path(__file__).parent.parent.parent.resolve()  # Go up to project root
    config_path = project_root / "configs" / "tools" / "seroost_config.json"

    # Read the configuration
    with open(config_path) as f:
        json.load(f)

    print(f"Setting up Seroost index for project at: {project_root}")
    print(f"Configuration file: {config_path}")

    # Try to run seroost binary
    try:
        seroost_binary = project_root / ".." / "workspace" / "seroost" / "target" / "release" / "seroost"

        if not seroost_binary.exists():
            raise FileNotFoundError(f"Seroost binary not found at {seroost_binary}")

        print("Starting indexing process...")
        result = subprocess.run([str(seroost_binary), "--index-path", str(project_root), "index"], capture_output=True, text=True)

        if result.returncode == 0:
            print("Indexing completed successfully!")
            print(result.stdout)
        else:
            print(f"Indexing failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure Seroost is built and available at the expected location.")
        print("Build instructions: cd ../workspace/seroost && cargo build --release")
    except Exception as e:
        print(f"Error occurred during indexing: {e}")
        print("Please check your Seroost installation and configuration.")


if __name__ == "__main__":
    setup_seroost_index()
