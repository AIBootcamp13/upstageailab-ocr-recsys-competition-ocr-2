# MCP Server Sharing Between AI Agents

## Overview

MCP servers can be shared between different AI agents (VS Code, Claude Desktop, Cursor, etc.), but each agent typically requires its own configuration file. However, there are several strategies to minimize duplication.

## Connection Types & Sharing Potential

### HTTP-Based Servers (High Sharing Potential)
```json
{
  "github/github-mcp-server": {
    "type": "http",
    "url": "https://api.githubcopilot.com/mcp/"
  }
}
```
**✅ Can be shared** - Single HTTP endpoint serves multiple agents
**📍 Configuration:** Each agent needs the HTTP URL in its config

### Stdio-Based Servers (Medium Sharing Potential)
```json
{
  "repomix": {
    "type": "stdio",
    "command": "npx",
    "args": ["-y", "repomix", "--mcp"]
  }
}
```
**⚠️ Per-agent launch** - Each agent starts its own process instance
**📍 Configuration:** Each agent needs the full command configuration

## Agent-Specific Configuration Locations

### VS Code
- **Location:** `.vscode/mcp.json` (workspace) or `~/.config/Code/User/mcp.json` (global)
- **Extension:** Requires MCP integration extension
- **Status:** ✅ Currently configured

### Claude Desktop
- **Location:** `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
- **Location:** `~/.config/Claude/claude_desktop_config.json` (Linux)
- **Status:** ❓ Not checked - may need separate config

### Cursor
- **Location:** `~/Library/Application Support/Cursor/User/mcp.json` (macOS)
- **Location:** `~/.config/Cursor/User/mcp.json` (Linux)
- **Status:** ❓ Not checked - may need separate config

## Sharing Strategies

### Strategy 1: HTTP Proxy (Recommended for High-Sharing Servers)

Convert stdio servers to HTTP endpoints that multiple agents can use:

```bash
# Example: Create HTTP proxy for repomix
npx mcp-remote http://localhost:3001 -- mcp-server-repomix
```

### Strategy 2: Shared Configuration Files

Create symlinks or shared config files:

```bash
# Create shared config
cp .vscode/mcp.json ~/shared-mcp-config.json

# Symlink for different applications
ln -s ~/shared-mcp-config.json ~/.config/Claude/claude_desktop_config.json
ln -s ~/shared-mcp-config.json ~/.config/Cursor/User/mcp.json
```

### Strategy 3: Environment Variables

Use environment variables for API keys and paths:

```bash
# .env file
MCP_TAVILY_API_KEY=your_key_here
MCP_UPSTAGE_API_KEY=your_key_here

# MCP config references env vars
{
  "tavily-remote": {
    "command": "mcp-remote",
    "args": ["https://mcp.tavily.com/mcp/?tavilyApiKey=$MCP_TAVILY_API_KEY"]
  }
}
```

## Current Setup Analysis

### Your VS Code Configuration
- **Location:** `.vscode/mcp.json`
- **Servers:** 7 active servers
- **Sharing Potential:**
  - `github/github-mcp-server`: ✅ High (HTTP-based)
  - Others: ⚠️ Medium (stdio-based, per-agent launch)

### Recommended Implementation

1. **Keep HTTP servers shared** (GitHub MCP)
2. **Use environment variables** for API keys
3. **Create agent-specific configs** that reference shared env vars
4. **Consider HTTP proxies** for frequently used stdio servers

## Example: Multi-Agent Setup

```bash
# 1. Create shared environment file
cat > ~/.mcp-env << EOF
export MCP_TAVILY_API_KEY="your_key"
export MCP_UPSTAGE_API_KEY="your_key"
EOF

# 2. Create VS Code config
cat > .vscode/mcp.json << EOF
{
  "servers": {
    "github": {"type": "http", "url": "https://api.githubcopilot.com/mcp/"},
    "tavily": {"command": "mcp-remote", "args": ["https://mcp.tavily.com/mcp/?tavilyApiKey=\$MCP_TAVILY_API_KEY"]},
    "repomix": {"command": "npx", "args": ["repomix", "--mcp"]}
  }
}
EOF

# 3. Create Claude Desktop config (if needed)
cat > ~/.config/Claude/claude_desktop_config.json << EOF
{
  "mcpServers": {
    "github": {"type": "http", "url": "https://api.githubcopilot.com/mcp/"},
    "tavily": {"command": "mcp-remote", "args": ["https://mcp.tavily.com/mcp/?tavilyApiKey=\$MCP_TAVILY_API_KEY"]},
    "repomix": {"command": "npx", "args": ["repomix", "--mcp"]}
  }
}
EOF
```

## Best Practices

### 🔐 Security
- Use environment variables for API keys
- Avoid hardcoding secrets in config files
- Use read-only API keys when possible

### 🔄 Maintenance
- Keep configs in sync across agents
- Use version control for MCP configurations
- Document server dependencies

### 🚀 Performance
- HTTP servers are more efficient for multiple agents
- Consider resource usage of stdio servers
- Monitor for resource conflicts

## Troubleshooting

### "Server not found" errors
- Check if server is installed: `which repomix`
- Verify environment variables: `echo $MCP_TAVILY_API_KEY`
- Check agent-specific config syntax

### "Connection failed" errors
- For HTTP servers: Check network connectivity
- For stdio servers: Verify command paths
- Check agent logs for detailed errors

### Resource conflicts
- Monitor process usage: `ps aux | grep mcp`
- Check memory usage: `top -p $(pgrep -f mcp)`
- Consider HTTP proxies for heavily used servers</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/MCP_AGENT_SHARING.md
