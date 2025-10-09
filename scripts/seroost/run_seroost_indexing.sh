#!/bin/bash

# Seroost indexing script for the OCR project
# This script sets up and runs the Seroost indexing with the project-specific configuration

set -e  # Exit immediately if a command exits with a non-zero status

echo "Setting up Seroost indexing for the OCR project..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/configs/tools/seroost_config.json"

echo "Project root: $PROJECT_ROOT"
echo "Config file: $CONFIG_FILE"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at $CONFIG_FILE"
    exit 1
fi

echo "Running setup_seroost_indexing.py..."
PYTHONPATH="$PYTHONPATH:$PROJECT_ROOT/../workspace/seroost" python "$PROJECT_ROOT/scripts/seroost/setup_seroost_indexing.py"

echo "Seroost indexing setup completed successfully!"
echo "You can now use Seroost to search through the indexed codebase."
