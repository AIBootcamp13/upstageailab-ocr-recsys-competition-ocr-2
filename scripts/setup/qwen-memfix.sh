#!/bin/bash
# Qwen wrapper script with increased memory limits
# This prevents heap exhaustion during intensive code generation tasks

# Set heap size to 8GB (8192 MB) - adjust as needed
HEAP_SIZE=16384

# Get the qwen binary path
QWEN_BIN="/home/vscode/.nvm/versions/node/v22.20.0/bin/qwen"

# Run qwen with increased heap size
exec node --max-old-space-size=$HEAP_SIZE "$QWEN_BIN" "$@"
