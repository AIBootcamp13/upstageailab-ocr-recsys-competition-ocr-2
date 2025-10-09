#!/bin/bash
# Qwen version manager script
# Usage: ./qwen-version.sh [stable|nightly|local]

VERSION=${1:-stable}

case $VERSION in
    "stable")
        echo "Using stable version (0.0.14)"
        qwen "$@"
        ;;
    "nightly")
        echo "Using nightly version (0.0.15-nightly.3)"
        npx @qwen-code/qwen-code@nightly "${@:2}"
        ;;
    "local")
        echo "Using local development version"
        /home/vscode/workspace/qwen-code/packages/cli/dist/index.js "${@:2}"
        ;;
    *)
        echo "Usage: $0 [stable|nightly|local] [qwen arguments]"
        exit 1
        ;;
esac
