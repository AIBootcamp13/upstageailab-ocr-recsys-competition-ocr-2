# Monitoring Scripts

This directory contains system monitoring tools for the OCR project.

## monitor.sh

**Purpose**: AI-powered system monitoring using Qwen with MCP integration.

**Usage**:
```bash
./monitor.sh "your monitoring command"
```

**Examples**:
```bash
./monitor.sh "Monitor system resources"
./monitor.sh "Check for orphaned processes"
./monitor.sh "List top CPU consuming processes"
./monitor.sh "Show system health status"
```

**Features**:
- ✅ Proper process cleanup on exit
- ✅ 10-minute timeout to prevent hanging
- ✅ Signal handling (Ctrl+C) with cleanup
- ✅ Automatic termination of background processes
- ✅ Clear exit status reporting

**Safety**: The script now properly terminates all processes it creates and handles interruptions gracefully.

## monitor_resources.sh

**Purpose**: Traditional bash-based system resource monitoring.

**Usage**:
```bash
./monitor_resources.sh
```

**Features**:
- CPU usage statistics
- Memory usage statistics
- Orphaned process detection
- Zombie process detection
- Long-running process identification
- Disk usage information
- Memory information

This script runs quickly and exits cleanly without leaving any processes running.
