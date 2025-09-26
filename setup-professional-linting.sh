#!/usr/bin/env bash
# setup-professional-linting.sh - One-command setup for professional Python linting
# Sets up Ruff, pre-commit hooks, VS Code config, and CI pipeline

set -e

echo "🚀 Setting up professional Python linting with Ruff..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if uv is available
if ! command -v uv &> /dev/null; then
    print_warning "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Install Ruff
print_status "Installing Ruff..."
uv add --dev ruff pre-commit

# Create Ruff config
print_status "Creating Ruff configuration..."
cat > pyproject.toml.ruff << 'EOF'
[tool.ruff]
line-length = 140
target-version = "py310"

[tool.ruff.lint]
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
]
ignore = [
    "E203", # whitespace before ':'
    "E501", # line too long, handled by black
    "E402", # module level import not at top of file
    "E722", # bare except
    "B008", # do not perform function calls in argument defaults
    "C901", # too complex
    "E741", # ambiguous variable name
    "B007", # unused loop variable
    "B904", # raise without from
    "F821", # undefined name
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]
"test_*.py" = ["B011"]
"tests/**" = ["E402", "E741", "F821"]
"ui/**" = ["E402"]
"scripts/**" = ["E402"]
"docker/**" = ["E402"]

[tool.ruff.lint.isort]
known-first-party = ["your_package_name"]  # Change this!

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"

[tool.mypy]
ignore_missing_imports = true
show_error_codes = false
# Development-friendly settings
warn_unused_ignores = false
warn_redundant_casts = false
warn_unused_configs = false
warn_unreachable = false
# Allow untyped calls and definitions for development
disallow_untyped_calls = false
disallow_untyped_defs = false
disallow_incomplete_defs = false
check_untyped_defs = false
EOF

# Update pyproject.toml (you'll need to merge this manually)
print_warning "Please merge the Ruff config from pyproject.toml.ruff into your pyproject.toml"
print_warning "Don't forget to update 'known-first-party' in [tool.ruff.lint.isort]"

# Create pre-commit config
print_status "Setting up pre-commit hooks..."
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.13.2
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.18.2
    hooks:
      - id: mypy
        args: [--ignore-missing-imports, --no-error-summary]
        additional_dependencies: [types-Pillow, types-tqdm, types-PyYAML]
        files: ^src/  # Change this to your source directory!
EOF

# Install pre-commit hooks
uv run pre-commit install
uv run pre-commit autoupdate

# Update Makefile
print_status "Updating Makefile..."
cat >> Makefile << 'EOF'

# Ruff-based quality checks (much faster!)
lint:
	uv run ruff check .

format:
	uv run ruff format .

quality-check: lint
	uv run mypy src/  # Change src/ to your source directory!
	uv run ruff check .
	uv run ruff format --check .

quality-fix:
	uv run ruff check . --fix --unsafe-fixes
	uv run ruff format .
EOF

# Create VS Code settings
print_status "Setting up VS Code configuration..."
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
  "python.linting.enabled": true,
  "python.linting.mypyEnabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": "explicit"
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    }
  }
}
EOF

# Create GitHub Actions workflow
print_status "Setting up CI pipeline..."
mkdir -p .github/workflows
cat > .github/workflows/ci.yml << 'EOF'
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Install uv
      uses: astral-sh/setup-uv@v3
      with:
        version: "0.4.15"

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: uv sync --extra dev

    - name: Run quality checks
      run: make quality-check

    - name: Run tests
      run: uv run pytest tests/ -v --tb=short
EOF

print_success "🎉 Professional linting setup complete!"
echo ""
echo "Next steps:"
echo "1. Merge Ruff config from pyproject.toml.ruff into pyproject.toml"
echo "2. Update source directory paths in configs (currently set to 'src/')"
echo "3. Update package name in Ruff config"
echo "4. Install VS Code extensions: Python, Ruff"
echo "5. Run 'make quality-check' to verify everything works"
echo ""
echo "Your linting is now 10-100x faster and fully automated! 🚀"
