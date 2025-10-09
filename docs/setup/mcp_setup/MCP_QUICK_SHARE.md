# MCP Quick Share Guide

## 🚀 One-Command Setup

```bash
# 1. Copy shareable config
cp mcp.shareable.json .vscode/mcp.json

# 2. Install dependencies
npm install -g repomix mcp-remote perplexity-mcp
pip install upstage-mcp  # or: uvx mcp-upstage

# 3. Set API keys
echo "TAVILY_API_KEY=your_key_here" > .env.local
echo "UPSTAGE_API_KEY=your_key_here" >> .env.local

# 4. Restart VS Code / Claude Desktop
```

## 📋 What's Included

### ✅ Ready to Use
- **GitHub MCP Server** - Repository management
- **Repomix** - Code analysis
- **Tavily Search** - Web search
- **Perplexity** - AI research
- **Upstage** - Document processing

### 🔧 Requires Configuration
- Replace `YOUR_TAVILY_API_KEY` with actual API key
- Install Claude CLI for `claude-code` server (optional)

## 🔍 Compatibility

| Application | Status | Notes |
|-------------|--------|-------|
| VS Code | ✅ | Requires MCP extension |
| Claude Desktop | ✅ | Native MCP support |
| Cursor | ✅ | MCP compatible |
| Other editors | ⚠️ | Check MCP support |

## 🛡️ Security

- ✅ No hardcoded secrets
- ✅ Environment variable usage
- ✅ Minimal required permissions
- ⚠️ Review API key access levels

## 📞 Need Help?

1. Check the full [MCP_SHARING_GUIDE.md](./MCP_SHARING_GUIDE.md)
2. Verify API keys are set correctly
3. Test individual servers: `npx repomix --help`
4. Check VS Code MCP extension status

---
*Generated from upstageailab-ocr-recsys-competition-ocr-2 workspace*</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/MCP_QUICK_SHARE.md
