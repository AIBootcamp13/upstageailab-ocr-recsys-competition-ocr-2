# Tool Configurations

This directory contains configuration files for various development tools used in the project.

## Files

### `repomix.config.json`
Configuration for [Repomix](https://repomix.com), a tool that packages codebases into single files for AI analysis.

**Purpose:** Creates compressed XML representations of the codebase for better context management in AI conversations.

**Usage:**
```bash
# Via MCP server (configured in .vscode/mcp.json)
# Or directly:
npx repomix --config configs/tools/repomix.config.json
```

**Output:** `../workspace/repomix_output/repomix-output.xml`

### `seroost_config.json`
Configuration for [Seroost](https://github.com/Seroost/seroost), a semantic code search engine.

**Purpose:** Defines which files to include/exclude when building the semantic search index.

**Usage:**
```bash
python scripts/seroost/setup_seroost_indexing.py
```

**Key Settings:**
- **Include:** Source code, configs, documentation, scripts
- **Exclude:** Build artifacts, dependencies, data files, logs

## Organization

These configuration files are kept separate from the main codebase to:
- Reduce clutter in the project root
- Allow tool-specific configurations
- Enable easier maintenance and updates
- Prevent accidental commits of tool configs

## Related Scripts

- `scripts/seroost/` - Seroost indexing scripts
- `scripts/monitoring/` - System monitoring scripts
