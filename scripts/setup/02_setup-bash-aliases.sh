#!/bin/bash
# Script to add OCR project make wrapper functions and aliases to ~/.bashrc
# This script safely appends the functions if they don't already exist

set -e

BASHRC_FILE="$HOME/.bashrc"
BACKUP_FILE="$HOME/.bashrc.backup.$(date +%Y%m%d_%H%M%S)"

# Function to check if content already exists in .bashrc
content_exists() {
    local content="$1"
    grep -F "$content" "$BASHRC_FILE" >/dev/null 2>&1
}

# Backup existing .bashrc
echo "Creating backup of existing .bashrc: $BACKUP_FILE"
cp "$BASHRC_FILE" "$BACKUP_FILE"

# Check and add omake function
if ! content_exists "function omake()"; then
    echo "Adding omake function to .bashrc..."
    cat >> "$BASHRC_FILE" << 'EOF'

# Generic make wrapper
function omake() {
  (cd /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2 && make "$@")
}
EOF
else
    echo "omake function already exists in .bashrc"
fi

# Check and add aliases
if ! content_exists "alias mcb='omake serve-ui'"; then
    echo "Adding UI aliases to .bashrc..."
    cat >> "$BASHRC_FILE" << 'EOF'

alias mcb='omake serve-ui'
alias mev='omake serve-evaluation-ui'
alias minf='omake serve-inference-ui'
EOF
else
    echo "UI aliases already exist in .bashrc"
fi

# Check and add UI functions
if ! content_exists "function ui-train()"; then
    echo "Adding UI functions to .bashrc..."
    cat >> "$BASHRC_FILE" << 'EOF'

function ui-train() { omake serve-ui PORT="${1:-8502}"; }
function ui-eval()  { omake serve-evaluation-ui PORT="${1:-8503}"; }
function ui-infer() { omake serve-inference-ui PORT="${1:-8504}"; }
EOF
else
    echo "UI functions already exist in .bashrc"
fi

echo "Setup complete!"
echo "Backup saved to: $BACKUP_FILE"
echo "Please run 'source ~/.bashrc' or restart your terminal to apply changes."
