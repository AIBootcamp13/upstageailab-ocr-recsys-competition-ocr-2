#!/usr/bin/env bash
# CI/CD Integration for Automated Diagram Updates
# This script runs as part of the CI pipeline to update diagrams when code changes

set -e

echo "🔍 Checking for diagram updates..."

# Run the diagram generation script
python scripts/generate_diagrams.py --check-changes > /tmp/diagram_changes.txt

# Check if any diagrams need updates
if grep -q "YES" /tmp/diagram_changes.txt; then
    echo "📊 Diagrams need updates. Generating new versions..."

    # Update all diagrams that changed
    python scripts/generate_diagrams.py --update

    # Validate the updated diagrams
    python scripts/generate_diagrams.py --validate

    echo "✅ Diagrams updated and validated successfully"

    # Check if we're in a CI environment with git
    if [ -n "$CI" ] && [ -d ".git" ]; then
        echo "🤖 Committing diagram updates..."

        # Add the updated diagrams
        git add docs/ai_handbook/03_references/architecture/diagrams/
        git add docs/ai_handbook/03_references/architecture/diagrams/_generated/

        # Check if there are changes to commit
        if git diff --cached --quiet; then
            echo "ℹ️ No diagram changes to commit"
        else
            # Create commit with diagram changes
            git commit -m "🤖 Auto-update Mermaid diagrams

Generated from codebase changes:
$(cat /tmp/diagram_changes.txt | grep "YES" | sed 's/: YES/- /')

Checksum: $(date +%s)"
            echo "📝 Committed diagram updates"
        fi
    else
        echo "ℹ️ Not in CI environment, skipping git commit"
    fi
else
    echo "✅ All diagrams are up to date"
fi

# Clean up
rm -f /tmp/diagram_changes.txt
