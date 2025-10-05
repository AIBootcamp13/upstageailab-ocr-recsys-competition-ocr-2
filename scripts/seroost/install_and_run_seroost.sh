#!/bin/bash

# Script to install Seroost and run indexing

set -e  # Exit immediately if a command exits with a non-zero status

echo "Installing Seroost and setting up indexing..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed or not in PATH"
    exit 1
fi

# Install seroost
echo "Installing seroost package..."
pip install seroost

# Run the setup script
echo "Running seroost indexing setup..."
python setup_seroost_indexing.py

echo "Seroost installation and indexing setup completed!"
