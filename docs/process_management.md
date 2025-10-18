# Streamlit Process Management

# Streamlit Process Management

This document explains how to prevent zombie processes when running Streamlit UI applications and how to view their logs for debugging.

## For Agents: Debugging Streamlit Apps

**⚠️ IMPORTANT:** When debugging Streamlit UI issues, follow the [Streamlit Debugging Protocol](../../ai_handbook/02_protocols/components/22_streamlit_debugging_protocol.md) for comprehensive context building. This protocol addresses the common problem of insufficient problem descriptions.

**Quick Agent Reference:**
```bash
# Start UI with full logging (MANDATORY for debugging)
make serve-inference-ui

# View logs immediately when errors occur
make logs-inference-ui

# Follow logs in real-time during reproduction
make logs-inference-ui -- --follow

# Document everything in context logs
make context-log-start LABEL="streamlit_debug"
```

**Common Agent Mistakes to Avoid:**
- Starting UI without logging enabled
- Not capturing logs immediately after errors
- Insufficient reproduction documentation
- Missing system/environment context

## The Problem

Zombie processes occur when:
1. A parent process (like `make`) starts a child process (like `streamlit`)
2. The parent exits before the child finishes
3. The child becomes "orphaned" and may not be properly reaped by the system

**Additional Problem**: When processes run in background, their output is lost, making debugging difficult.

## Solutions

### 1. Process Manager with Logging (Recommended)

The process manager now captures all stdout and stderr output to log files for debugging:

```bash
# Start UIs with automatic logging
make serve-inference-ui    # Logs to logs/ui/inference_8501.out/.err
make serve-evaluation-ui  # Logs to logs/ui/evaluation_viewer_8501.out/.err

# View logs for debugging
make logs-inference-ui     # View last 50 lines of logs
make logs-inference-ui -- --lines 100  # View last 100 lines
make logs-inference-ui -- --follow     # Follow logs in real-time

# Clear old logs
make clear-logs-inference-ui
```

**Log Locations:**
- `logs/ui/{ui_name}_{port}.out` - Standard output
- `logs/ui/{ui_name}_{port}.err` - Error output

### 2. Direct Process Manager Usage

```bash
# Start with logging (default)
uv run python scripts/process_manager.py start inference --port 8504

# Start without logging (old behavior)
uv run python scripts/process_manager.py start inference --port 8504 --no-logging

# View logs
uv run python scripts/process_manager.py logs inference --port 8504
uv run python scripts/process_manager.py logs inference --port 8504 --lines 100
uv run python scripts/process_manager.py logs inference --port 8504 --follow

# Clear logs
uv run python scripts/process_manager.py clear-logs inference --port 8504
```

### 3. Debugging Inference Failures

When you see "Inference failed: Inference engine returned no results", check the logs:

```bash
# View inference UI logs
make logs-inference-ui

# Or follow in real-time while reproducing the issue
make logs-inference-ui -- --follow
```

**Common Issues to Look For:**
- Model loading errors
- Checkpoint compatibility issues
- CUDA/ GPU memory errors
- Import errors
- Configuration validation failures

### 4. Alternative Solutions

#### Tmux Sessions (For Development)
```bash
tmux new -s streamlit-ui
make serve-inference-ui  # UI runs in tmux, output visible
# Ctrl+B, D to detach - UI keeps running
tmux attach -t streamlit-ui  # Reattach to see output
```

#### Nohup (Simple Alternative)
```bash
# Start with output to files
nohup uv run streamlit run ui/inference_ui.py --server.port=8501 > inference.log 2>&1 &
```

## Best Practices

1. **Always use the process manager** for production/development workflows
2. **Check logs immediately** when encountering errors
3. **Use `--follow`** when reproducing issues to see real-time output
4. **Clear logs periodically** to avoid disk space issues
5. **Use tmux** for interactive development sessions where you need to see output immediately

## Troubleshooting

### No Log Files Found
If logs don't exist:
```bash
# Check if process is running
make status-inference-ui

# Restart with logging if needed
make stop-inference-ui
make serve-inference-ui
```

### Process Started Without Logging
If you started a process without logging:
```bash
# Stop and restart with logging
make stop-inference-ui
make serve-inference-ui
```

### Log Files Too Large
```bash
# Clear old logs
make clear-logs-inference-ui

# Or manually
rm logs/ui/inference_*.log
```

### Permission Issues
```bash
# Ensure log directory is writable
ls -la logs/ui/
chmod 755 logs/ui/
```

## Known Issues & Fixes

### Pydantic V2 Configuration Warnings
**Issue:** `UserWarning: Valid config keys have changed in V2: 'allow_population_by_field_name' has been renamed to 'validate_by_name'`

**Root Cause:** Dependencies using old Pydantic v2 parameter names during inference engine initialization.

**Fix:** Added warning suppressions in:
- `ui/inference_ui.py` - Entry point suppressions
- `ui/utils/inference/engine.py` - Engine-level suppressions

**Status:** ✅ Resolved - Warnings no longer appear in logs

## The Problem

Zombie processes occur when:
1. A parent process (like `make`) starts a child process (like `streamlit`)
2. The parent exits before the child finishes
3. The child becomes "orphaned" and may not be properly reaped by the system

## Solutions

### 1. Process Manager (Recommended)

Use the new process manager for proper lifecycle management:

```bash
# Start UIs
make serve-inference-ui    # Starts on port 8501
make serve-evaluation-ui  # Starts on port 8501 (change PORT variable)

# Check status
make status-inference-ui

# Stop UIs
make stop-inference-ui

# List all running UIs
make list-ui-processes

# Stop all UIs
make stop-all-ui
```

**Features:**
- PID file management
- Proper process group handling with `os.setsid()`
- Graceful shutdown (SIGTERM then SIGKILL)
- Background execution with output redirection
- Port conflict detection

### 2. Direct Process Manager Usage

```bash
# Start specific UI on custom port
uv run python scripts/process_manager.py start inference --port 8504

# Check status
uv run python scripts/process_manager.py status inference --port 8504

# Stop UI
uv run python scripts/process_manager.py stop inference --port 8504

# List all managed processes
uv run python scripts/process_manager.py list

# Stop all processes
uv run python scripts/process_manager.py stop-all
```

### 3. Tmux Sessions (For Development)

Use tmux for persistent sessions that survive terminal disconnection:

```bash
# Create a new tmux session for UI
tmux new -s streamlit-ui

# Inside tmux, start the UI
make serve-inference-ui

# Detach from tmux (Ctrl+B, D)
# UI continues running

# Reattach later
tmux attach -t streamlit-ui

# Kill the session when done
tmux kill-session -t streamlit-ui
```

### 4. Nohup (Simple Alternative)

For simple cases, use `nohup` to detach processes:

```bash
# Start UI with nohup
nohup uv run streamlit run ui/inference_ui.py --server.port=8501 > /dev/null 2>&1 &

# Find and kill later
ps aux | grep streamlit
kill <PID>
```

## Best Practices

1. **Always use the process manager** for production/development workflows
2. **Check for running processes** before starting new ones
3. **Use tmux** for interactive development sessions
4. **Clean up PID files** if processes crash unexpectedly
5. **Monitor resource usage** with `make serve-resource-monitor`

## Troubleshooting

### Zombie Processes Still Occur

If you still see zombie processes:

1. Check if processes are started via the process manager
2. Look for orphaned processes: `ps aux | grep '<defunct>'`
3. Kill parent processes of zombies: `kill -9 <PPID>`
4. Clean up stale PID files in the project root

### Port Conflicts

If ports are in use:

```bash
# Check what's using the port
lsof -i :8501

# Use a different port
PORT=8502 make serve-inference-ui
```

### Process Won't Stop

If a process won't stop gracefully:

```bash
# Force kill
uv run python scripts/process_manager.py stop inference --port 8501

# Or manually
ps aux | grep streamlit
kill -9 <PID>
rm .inference_8501.pid  # Clean up PID file
```

## Configuration

The process manager creates PID files in the project root:
- `.inference_8501.pid`
- `.evaluation_viewer_8501.pid`
- etc.

These files are automatically cleaned up when processes stop normally.
