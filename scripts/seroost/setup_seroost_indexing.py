#!/usr/bin/env python3
"""
Seroost indexing setup script for the OCR project.

This script configures Seroost to index source code, documentation,
and configuration files while excluding build artifacts, dependencies,
and large data files.
"""

import json
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

    # Try to import and use seroost functions
    try:
        from seroost import seroost_index, seroost_set_index

        seroost_set_index(str(project_root))

        print("Starting indexing process...")
        seroost_index()
        print("Indexing completed successfully!")

    except ImportError:
        print("Warning: Seroost module not found. To use this indexing configuration:")
        print("1. Install the seroost package: pip install seroost")
        print("2. Run this script again: python setup_seroost_indexing.py")
        print("\nAlternatively, you can use the configuration file directly:")
        print(f"   Configuration file: {config_path}")
        print("   Project directory to index: ", project_root)
    except Exception as e:
        print(f"Error occurred during indexing: {e}")
        print("Please check your Seroost installation and configuration.")


if __name__ == "__main__":
    setup_seroost_index()
