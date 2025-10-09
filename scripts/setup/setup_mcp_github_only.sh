#!/bin/bash
# MCP Setup Helper Script - GitHub MCP Only
# Simplified setup for GitHub MCP server only

set -e

echo "🤖 MCP Setup Helper - GitHub Only"
echo "=================================="

# Check if we're in the right directory
if [ ! -f ".vscode/mcp.shareable.json" ]; then
    echo "❌ Error: .vscode/mcp.shareable.json not found"
    echo "   Please run this script from the workspace root"
    exit 1
fi

# Backup existing config if it exists
if [ -f ".vscode/mcp.json" ]; then
    echo "📋 Backing up existing MCP configuration..."
    cp .vscode/mcp.json ".vscode/mcp.json.backup.$(date +%Y%m%d_%H%M%S)"
fi

# Copy simplified config
echo "📋 Installing simplified MCP configuration (GitHub only)..."
cp .vscode/mcp.shareable.json .vscode/mcp.json

echo ""
echo "🎉 MCP setup complete!"
echo ""
echo "📋 Configuration:"
echo "   • GitHub MCP Server: ✅ Enabled"
echo "   • Other servers: ❌ Disabled (simplified setup)"
echo ""
echo "🚀 Next steps:"
echo "   1. Restart VS Code"
echo "   2. Check MCP extension status"
echo ""
echo "📖 For sharing with other agents, see MCP_AGENT_SHARING.md"
