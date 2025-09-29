#!/bin/bash
# Environment Setup Script for DBNet OCR Project
# This script ensures the correct UV environment is set up and configured

set -e

echo "🔧 Setting up DBNet OCR Project Environment"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must be run from project root (where pyproject.toml is located)"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d. -f1-2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Error: Python $REQUIRED_VERSION+ required, found $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python version: $PYTHON_VERSION"

# Install/sync dependencies with UV
echo "📦 Installing dependencies with UV..."
uv sync --group dev

# Verify virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment not created"
    exit 1
fi

echo "✅ Virtual environment created at .venv/"

# Test basic imports
echo "🧪 Testing basic imports..."
uv run python -c "
import torch
import lightning
import hydra
print('✅ Core dependencies imported successfully')
print(f'   PyTorch: {torch.__version__}')
print(f'   Lightning: {lightning.__version__}')
"

# Test pytest
echo "🧪 Testing pytest..."
uv run python -m pytest --version > /dev/null
echo "✅ pytest working"

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "Next steps:"
echo "  • Open in VS Code (interpreter should auto-detect)"
echo "  • Run 'uv run pytest tests/' to verify tests"
echo "  • Use 'uv run' prefix for all Python commands"
echo ""
echo "VS Code should automatically use: ./venv/bin/python"
