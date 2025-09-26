#!/usr/bin/env bash
# code-quality.sh - Automated code quality maintenance script

set -e

echo "🤖 Starting automated code quality maintenance..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a git repository. Please run this script from the project root."
    exit 1
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    print_warning "You have uncommitted changes. Please commit or stash them first."
    print_status "Current status:"
    git status --short
    exit 1
fi

print_status "Installing/updating pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    pre-commit autoupdate
else
    print_warning "pre-commit not found. Installing..."
    pip install pre-commit
    pre-commit install
    pre-commit autoupdate
fi

print_status "Running pre-commit on all files..."
pre-commit run --all-files

# Check if there are changes
if [ -n "$(git status --porcelain)" ]; then
    print_success "Code quality issues found and fixed!"
    print_status "Changes made:"
    git diff --name-only

    # Ask user if they want to commit
    echo
    read -p "Do you want to commit these changes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        git commit -m "🤖 Auto-fix: code formatting and linting

- Applied Black code formatting
- Sorted imports with isort
- Fixed flake8 violations
- Removed unused imports and variables
- Applied automated code quality fixes"

        print_success "Changes committed successfully!"
        print_status "Ready to push: git push"
    else
        print_status "Changes staged but not committed. Review with: git diff"
    fi
else
    print_success "No code quality issues found! 🎉"
fi

print_status "Code quality maintenance complete!"
