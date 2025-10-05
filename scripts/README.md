# Scripts Directory

This directory contains various utility scripts and tools for the OCR project.

## Organization

### `agent_tools/`
Scripts and tools for AI agent integration and automation.

### `monitoring/`
System monitoring and resource management tools.
- `monitor.sh` - AI-powered system monitoring via Qwen MCP server
- `monitor_resources.sh` - Resource monitoring script

### `seroost/`
Seroost semantic search indexing tools and configuration.
- `SEROOST_INDEXING_SETUP.md` - Complete setup documentation
- `setup_seroost_indexing.py` - Python script for indexing setup
- `test_seroost_config.py` - Configuration validation script
- `run_seroost_indexing.sh` - Shell script for running indexing
- `install_and_run_seroost.sh` - Complete installation and setup script
- `seroost_indexing/` - Additional indexing documentation

### `setup/`
Project setup and configuration scripts.

## Usage

### System Monitoring
```bash
# AI-powered system monitoring
./scripts/monitoring/monitor.sh "Show system health status"

# Resource monitoring
./scripts/monitoring/monitor_resources.sh
```

### Seroost Indexing
```bash
# Complete setup (install + index)
./scripts/seroost/install_and_run_seroost.sh

# Just run indexing (if seroost is already installed)
./scripts/seroost/run_seroost_indexing.sh

# Manual setup
python ./scripts/seroost/setup_seroost_indexing.py
```

## Contributing

When adding new scripts:
1. Place them in the appropriate subdirectory
2. Update this README with documentation
3. Ensure scripts are executable (`chmod +x script.sh`)
4. Include usage examples and descriptions
