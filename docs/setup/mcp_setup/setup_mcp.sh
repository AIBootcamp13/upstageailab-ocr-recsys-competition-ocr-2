#!/bin/bash
# MCP Setup Helper Script
# Quickly set up MCP servers from the shareable template

set -e

echo "🤖 MCP Server Setup Helper"
echo "=========================="

# Check if we're in the right directory
if [ ! -f "mcp.shareable.json" ]; then
    echo "❌ Error: mcp.shareable.json not found in current directory"
    echo "   Please run this script from the workspace root"
    exit 1
fi

# Backup existing config if it exists
if [ -f ".vscode/mcp.json" ]; then
    echo "📋 Backing up existing MCP configuration..."
    cp .vscode/mcp.json ".vscode/mcp.json.backup.$(date +%Y%m%d_%H%M%S)"
fi

# Copy shareable config
echo "📋 Installing shareable MCP configuration..."
cp mcp.shareable.json .vscode/mcp.json

# Create .vscode directory if it doesn't exist
mkdir -p .vscode

echo "📦 Installing required packages..."

# Check if npm is available
if command -v npm &> /dev/null; then
    echo "   Installing npm packages..."
    npm install -g repomix mcp-remote perplexity-mcp 2>/dev/null || echo "   ⚠️  Some npm packages may have failed to install"
else
    echo "   ⚠️  npm not found - please install Node.js and run:"
    echo "      npm install -g repomix mcp-remote perplexity-mcp"
fi

# Check if pip/uv is available
if command -v uv &> /dev/null; then
    echo "   Installing Python packages with uv..."
    uv tool install upstage-mcp 2>/dev/null || echo "   ⚠️  upstage-mcp installation failed"
elif command -v pip &> /dev/null; then
    echo "   Installing Python packages with pip..."
    pip install upstage-mcp 2>/dev/null || echo "   ⚠️  upstage-mcp installation failed"
else
    echo "   ⚠️  Neither uv nor pip found - please install upstage-mcp manually"
fi

# Check for .env.local
if [ ! -f ".env.local" ]; then
    echo "📝 Creating .env.local template..."
    cat > .env.local << 'EOF'
# MCP API Keys - Replace with your actual keys
TAVILY_API_KEY=your_tavily_api_key_here
UPSTAGE_API_KEY=your_upstage_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Optional: Claude CLI path (uncomment and update if using claude-code server)
# CLAUDE_PATH=/path/to/claude
EOF
    echo "   ✅ Created .env.local - please edit with your API keys"
else
    echo "   ℹ️  .env.local already exists - please verify your API keys"
fi

echo ""
echo "🎉 MCP setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Edit .env.local with your actual API keys"
echo "   2. Restart VS Code or your MCP-compatible editor"
echo "   3. Check that MCP servers are connected"
echo ""
echo "📖 For detailed help, see:"
echo "   • MCP_QUICK_SHARE.md (quick reference)"
echo "   • MCP_SHARING_GUIDE.md (comprehensive guide)"
echo ""
echo "🔍 To verify installation:"
echo "   npx repomix --version"
echo "   npx mcp-remote --help"
echo "   uvx mcp-upstage --help 2>/dev/null || echo 'upstage not available'"</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/setup_mcp.sh
