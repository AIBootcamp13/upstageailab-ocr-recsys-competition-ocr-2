#!/bin/bash
# MCP Agent Sharing Setup Script
# Share MCP servers between VS Code and other AI agents

set -e

echo "🤖 MCP Agent Sharing Setup"
echo "=========================="

# Check if shareable config exists
if [ ! -f ".vscode/mcp.shareable.json" ]; then
    echo "❌ Error: .vscode/mcp.shareable.json not found"
    echo "   Run this script from the workspace root"
    exit 1
fi

# Create shared environment file
if [ ! -f ~/.mcp-env ]; then
    echo "📝 Creating shared MCP environment file..."
    cat > ~/.mcp-env << 'EOF'
# MCP API Keys - Set your actual keys here
export MCP_TAVILY_API_KEY="your_tavily_api_key_here"
export MCP_UPSTAGE_API_KEY="your_upstage_api_key_here"
export MCP_PERPLEXITY_API_KEY="your_perplexity_api_key_here"

# Optional: Claude CLI path
# export CLAUDE_PATH="/path/to/claude"
EOF
    echo "   ✅ Created ~/.mcp-env - please edit with your API keys"
else
    echo "   ℹ️  ~/.mcp-env already exists"
fi

# Setup Claude Desktop (if installed)
if [ -d ~/.config/Claude ] || [ -d ~/Library/Application\ Support/Claude ]; then
    echo "🤖 Setting up Claude Desktop MCP config..."

    # Determine Claude config location
    if [ -d ~/.config/Claude ]; then
        CLAUDE_CONFIG_DIR=~/.config/Claude
    else
        CLAUDE_CONFIG_DIR=~/Library/Application\ Support/Claude
    fi

    # Create Claude config
    mkdir -p "$CLAUDE_CONFIG_DIR"
    cp .vscode/mcp.shareable.json "$CLAUDE_CONFIG_DIR/claude_desktop_config.json"

    echo "   ✅ Claude Desktop config created at: $CLAUDE_CONFIG_DIR/claude_desktop_config.json"
else
    echo "   ⚠️  Claude Desktop not detected - install it and re-run this script"
fi

# Setup Cursor (if installed)
if [ -d ~/.config/Cursor ] || [ -d ~/Library/Application\ Support/Cursor ]; then
    echo "🖱️  Setting up Cursor MCP config..."

    # Determine Cursor config location
    if [ -d ~/.config/Cursor ]; then
        CURSOR_CONFIG_DIR=~/.config/Cursor/User
    else
        CURSOR_CONFIG_DIR=~/Library/Application\ Support/Cursor/User
    fi

    # Create Cursor config
    mkdir -p "$CURSOR_CONFIG_DIR"
    cp .vscode/mcp.shareable.json "$CURSOR_CONFIG_DIR/mcp.json"

    echo "   ✅ Cursor config created at: $CURSOR_CONFIG_DIR/mcp.json"
else
    echo "   ⚠️  Cursor not detected - install it and re-run this script"
fi

# Add environment sourcing to shell
SHELL_RC=""
if [ -f ~/.bashrc ]; then
    SHELL_RC=~/.bashrc
elif [ -f ~/.zshrc ]; then
    SHELL_RC=~/.zshrc
fi

if [ -n "$SHELL_RC" ]; then
    if ! grep -q "source ~/.mcp-env" "$SHELL_RC"; then
        echo "📝 Adding MCP environment to $SHELL_RC..."
        echo "" >> "$SHELL_RC"
        echo "# MCP Environment Variables" >> "$SHELL_RC"
        echo "source ~/.mcp-env" >> "$SHELL_RC"
        echo "   ✅ Environment sourcing added to $SHELL_RC"
    else
        echo "   ℹ️  MCP environment already sourced in $SHELL_RC"
    fi
fi

echo ""
echo "🎉 MCP Agent Sharing Setup Complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Edit ~/.mcp-env with your actual API keys"
echo "   2. Restart your AI agents (VS Code, Claude Desktop, Cursor)"
echo "   3. Source environment: source ~/.mcp-env"
echo ""
echo "🔍 Verify setup:"
echo "   • VS Code: Check MCP extension status"
echo "   • Claude Desktop: Check MCP server connections"
echo "   • Cursor: Check MCP server connections"
echo ""
echo "📖 For detailed help, see MCP_AGENT_SHARING.md"</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/setup_mcp_agents.sh
