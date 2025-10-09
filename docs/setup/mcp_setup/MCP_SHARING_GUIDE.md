# MCP Server Configuration Guide

## Overview

This workspace uses the **Model Context Protocol (MCP)** to connect AI assistants with external tools and services. MCP enables AI models to interact with various APIs, databases, and tools through a standardized protocol.

## Current Configuration

### Active Servers (8 total)

| Server | Purpose | Shareable | Status |
|--------|---------|-----------|--------|
| `github/github-mcp-server` | GitHub integration | ✅ Yes | Active |
| `repomix` | Codebase analysis | 🔧 Adaptable | Active |
| `tavily-remote` | Web search via Tavily | 🔧 Adaptable | Active |
| `perplexity` | AI-powered search | 🔧 Adaptable | Active |
| `upstage` | Upstage AI services | 🔧 Adaptable | Active |
| `claude-code` | Claude CLI integration | 🔧 Adaptable | Active |
| `seroost-search` | Semantic code search | ❌ Workspace-specific | Active |
| `system-monitor` | System monitoring | ❌ Workspace-specific | Active |

## Sharing MCP Configurations

### Quick Start for Sharing

1. **Copy the shareable template:**
   ```bash
   cp .vscode/mcp.shareable.json /path/to/other/workspace/.vscode/mcp.json
   ```

2. **Customize for your environment:**
   - Replace `YOUR_TAVILY_API_KEY` with actual API key
   - Update any workspace-specific paths
   - Configure environment variables

3. **Install dependencies:**
   ```bash
   # For repomix
   npm install -g repomix

   # For tavily-remote
   npm install -g mcp-remote

   # For perplexity
   npm install -g perplexity-mcp

   # For upstage
   pip install upstage-mcp  # or uvx mcp-upstage
   ```

## Server Details

### ✅ Fully Shareable Servers

#### GitHub MCP Server
```json
{
  "github/github-mcp-server": {
    "type": "http",
    "url": "https://api.githubcopilot.com/mcp/",
    "gallery": "https://api.mcp.github.com/v0/servers/ab12cd34-5678-90ef-1234-567890abcdef",
    "version": "0.13.0"
  }
}
```
**Purpose:** GitHub repository management and queries
**Requirements:** None (works out-of-the-box)

### 🔧 Adaptable Servers

#### Repomix
```json
{
  "repomix": {
    "type": "stdio",
    "command": "npx",
    "args": ["-y", "repomix", "--mcp"]
  }
}
```
**Purpose:** Codebase analysis and packaging
**Adaptation needed:** Remove workspace-specific config paths

#### Tavily Remote
```json
{
  "tavily-remote": {
    "type": "stdio",
    "command": "npx",
    "args": ["-y", "mcp-remote", "https://mcp.tavily.com/mcp/?tavilyApiKey=YOUR_API_KEY"]
  }
}
```
**Purpose:** Web search and information retrieval
**Adaptation needed:** Replace `YOUR_API_KEY` with actual Tavily API key

#### Perplexity
```json
{
  "perplexity": {
    "type": "stdio",
    "command": "npx",
    "args": ["-y", "perplexity-mcp"]
  }
}
```
**Purpose:** AI-powered research and search
**Adaptation needed:** Configure API credentials via environment

#### Upstage
```json
{
  "upstage": {
    "type": "stdio",
    "command": "uvx",
    "args": ["mcp-upstage"]
  }
}
```
**Purpose:** Upstage AI document processing services
**Adaptation needed:** Configure Upstage API credentials

#### Claude Code
```json
{
  "claude-code": {
    "command": "/path/to/claude",
    "args": ["mcp", "serve"]
  }
}
```
**Purpose:** Claude CLI integration
**Adaptation needed:** Update path to local Claude installation

### ❌ Workspace-Specific Servers

#### Seroost Search
```json
{
  "seroost-search": {
    "type": "stdio",
    "command": "node",
    "args": ["/path/to/semantic-search-mcp/build/index.js"]
  }
}
```
**Purpose:** Semantic search across codebase
**Why not shareable:** Hardcoded workspace paths

#### System Monitor
```json
{
  "system-monitor": {
    "type": "stdio",
    "command": "node",
    "args": ["/path/to/system_monitor_mcp/index.js"]
  }
}
```
**Purpose:** System resource monitoring
**Why not shareable:** Hardcoded workspace paths

## Cross-Platform Compatibility

### Supported Applications
- **VS Code** (with MCP extension)
- **Claude Desktop**
- **Cursor**
- **Other MCP-compatible editors**

### Environment Variables
Create a `.env.local` file for API keys:
```bash
# Example .env.local
TAVILY_API_KEY=your_tavily_api_key_here
UPSTAGE_API_KEY=your_upstage_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
```

## Security Considerations

### 🔐 API Keys
- Never commit API keys to version control
- Use environment variables for secrets
- Rotate keys regularly
- Use read-only keys when possible

### 🛡️ Workspace Paths
- Avoid sharing absolute paths
- Use relative paths when possible
- Document required directory structures
- Consider using environment variables for paths

### 🔍 Access Control
- Review what each MCP server can access
- Limit permissions to necessary operations
- Monitor server usage and logs

## Troubleshooting

### Common Issues

#### "Server not found" errors
- Check if required packages are installed
- Verify paths in configuration
- Ensure environment variables are set

#### "Permission denied" errors
- Check file permissions on MCP server scripts
- Verify user has access to required directories
- Check if services are running

#### "Connection failed" errors
- Verify network connectivity for remote services
- Check API key validity
- Review firewall settings

### Debugging Steps

1. **Check MCP extension status:**
   - VS Code: MCP extension should show connected servers
   - Claude Desktop: Check MCP server logs

2. **Validate configuration:**
   ```bash
   # Test JSON syntax
   python -m json.tool .vscode/mcp.json

   # Check for required files
   ls -la /path/to/mcp/servers/
   ```

3. **Monitor server processes:**
   ```bash
   # Check for running MCP processes
   ps aux | grep mcp

   # Monitor resource usage
   top -p $(pgrep -f mcp)
   ```

## Contributing

### Adding New Servers
1. Test server locally first
2. Add configuration to `.vscode/mcp.json`
3. Update this documentation
4. Test sharing capability
5. Add to shareable template if appropriate

### Updating Existing Servers
1. Test changes locally
2. Update both main config and shareable template
3. Update documentation
4. Test backward compatibility

## Resources

- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [MCP Server Registry](https://github.com/modelcontextprotocol/registry)
- [VS Code MCP Extension](https://marketplace.visualstudio.com/items?itemName=buildwithlayer.mcp-integration-expert-eligr)

## Version History

- **v1.0** - Initial consolidated configuration
- **v1.1** - Added shareable template and documentation</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/MCP_SHARING_GUIDE.md
