#!/bin/bash
# Diagram Management Script
# Quick commands for diagram operations during development

set -e

case "$1" in
    "update")
        echo "🔄 Updating all diagrams..."
        python scripts/generate_diagrams.py --update
        ;;
    "check")
        echo "🔍 Checking for diagram updates..."
        python scripts/generate_diagrams.py --check-changes
        ;;
    "validate")
        echo "✅ Validating diagram syntax..."
        python scripts/generate_diagrams.py --validate
        ;;
    "force-update")
        echo "🔄 Force updating all diagrams..."
        python scripts/generate_diagrams.py --update --force
        ;;
    *)
        echo "Usage: $0 {update|check|validate|force-update}"
        echo ""
        echo "Commands:"
        echo "  update      - Update diagrams that have changed"
        echo "  check       - Check which diagrams need updates"
        echo "  validate    - Validate diagram syntax"
        echo "  force-update- Force update all diagrams"
        exit 1
        ;;
esac
